"""Pose estimation: a Gauss-Newton solver and an Unscented Kalman Filter.

Both invert `forward.predict`. The Gauss-Newton solver is memoryless -- it fits
one frame at a time -- and doubles as the UKF's initialiser and as the baseline
that says what the dynamics actually buy.

State is 12-dimensional, ``[t, theta, tdot, thetadot]``: pose (see `geometry` for
the rotation-vector convention) plus its velocity, under a constant-velocity
process model. Velocity is worth carrying because the sensors run at ~770 Hz
while the hand moves at a few Hz, so the filter has a lot of frames over which to
average, and a smoothed derivative is what makes the output feel continuous
rather than jittery.

Written for a later Rust port: plain arrays, no SciPy inside `update`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import forward, geometry
from .magnet import MagnetModel
from .params import CalibParams

N_STATE = 12
N_POSE = 6

#: Process noise as an acceleration std per DOF: mm/s^2 then rad/s^2.
#:
#: Tuned on a synthetic 0.5-2.3 Hz six-axis trajectory at realistic sensor
#: noise. Against per-frame Gauss-Newton on identical data this halves the
#: sample-to-sample jitter *and* roughly halves the tracking error:
#:
#:     per-frame solve      jitter 0.031   error 0.051 mm / 0.17 deg
#:     UKF, this value      jitter 0.016   error 0.027 mm / 0.10 deg
#:
#: Raising it converges back to the memoryless solve; lowering it another
#: decade starts to lag real motion (error climbs to 0.053 mm / 0.38 deg).
DEFAULT_ACCEL_STD = np.array([300.0, 300.0, 300.0, 30.0, 30.0, 30.0])


# --------------------------------------------------------------- Gauss-Newton


def solve_pose(
    counts: np.ndarray,
    params: CalibParams,
    model: MagnetModel | None = None,
    guess: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    """Least-squares pose from a single 9-vector of counts.

    Warm-starting from the previous pose (`guess`) makes this both faster and
    more robust; from a cold start it still converges, which is what makes it a
    usable initialiser for the filter.
    """
    model = model or forward.DEFAULT_MODEL
    counts = np.asarray(counts, dtype=float)
    w = 1.0 / (np.ones(geometry.N_CHANNELS) if sigma is None else np.asarray(sigma))

    def residual(pose: np.ndarray) -> np.ndarray:
        return (forward.predict(pose, params, model) - counts) * w

    x0 = np.zeros(N_POSE) if guess is None else np.asarray(guess, dtype=float)[:N_POSE]
    return least_squares(residual, x0, method="lm").x


# ------------------------------------------------------------------------ UKF


def _process_matrix(dt: float) -> np.ndarray:
    """Constant-velocity transition for the 12-state."""
    f = np.eye(N_STATE)
    f[:N_POSE, N_POSE:] = dt * np.eye(N_POSE)
    return f


def _process_noise(dt: float, accel_std: np.ndarray) -> np.ndarray:
    """Discrete white-noise-acceleration covariance.

    The standard piecewise-constant-acceleration form: an unmodelled
    acceleration over one step displaces by dt^2/2 and changes velocity by dt.
    """
    q = np.zeros((N_STATE, N_STATE))
    var = np.asarray(accel_std, dtype=float) ** 2
    pp = (dt**4) / 4.0
    pv = (dt**3) / 2.0
    vv = dt**2
    idx = np.arange(N_POSE)
    q[idx, idx] = pp * var
    q[idx, idx + N_POSE] = pv * var
    q[idx + N_POSE, idx] = pv * var
    q[idx + N_POSE, idx + N_POSE] = vv * var
    return q


@dataclass
class UKF:
    """Scaled-sigma-point UKF over the 12-state.

    Args:
        params: calibration, as fitted by `calibrate`.
        sigma: per-channel measurement noise in counts; take it from
            `calibrate.estimate_noise` rather than guessing.
        accel_std: process noise, as an acceleration std per DOF
            (mm/s^2 for translation, rad/s^2 for rotation). This is the one
            genuinely free tuning knob -- raise it to track faster and jitter
            more, lower it to smooth and lag. See `DEFAULT_ACCEL_STD`.
        model: magnet model; defaults to the finite-size one.
    """

    params: CalibParams
    sigma: np.ndarray
    accel_std: np.ndarray = field(default_factory=lambda: DEFAULT_ACCEL_STD.copy())
    model: MagnetModel | None = None
    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    x: np.ndarray = field(default_factory=lambda: np.zeros(N_STATE))
    P: np.ndarray = field(default_factory=lambda: np.eye(N_STATE))

    def __post_init__(self) -> None:
        self.model = self.model or forward.DEFAULT_MODEL
        self.sigma = np.asarray(self.sigma, dtype=float).reshape(geometry.N_CHANNELS)
        self.accel_std = np.asarray(self.accel_std, dtype=float).reshape(N_POSE)
        self.R = np.diag(self.sigma**2)

        lam = self.alpha**2 * (N_STATE + self.kappa) - N_STATE
        self._lambda = lam
        self._gamma = np.sqrt(N_STATE + lam)

        wm = np.full(2 * N_STATE + 1, 1.0 / (2.0 * (N_STATE + lam)))
        wc = wm.copy()
        wm[0] = lam / (N_STATE + lam)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)
        self.wm, self.wc = wm, wc

        # A pose we have not measured yet is unknown at roughly full travel.
        self.P = np.diag(
            np.concatenate([np.full(N_POSE, 1.0), np.full(N_POSE, 100.0)])
        )

    # -------------------------------------------------------------- internals

    def _sigma_points(self) -> np.ndarray:
        """(2n+1, n) scaled sigma points about the current mean."""
        try:
            chol = np.linalg.cholesky((N_STATE + self._lambda) * self.P)
        except np.linalg.LinAlgError:
            # Nudge back to positive-definite rather than dying mid-stream.
            jitter = 1e-9 * np.eye(N_STATE)
            chol = np.linalg.cholesky((N_STATE + self._lambda) * self.P + jitter)

        pts = np.empty((2 * N_STATE + 1, N_STATE))
        pts[0] = self.x
        pts[1 : N_STATE + 1] = self.x + chol.T
        pts[N_STATE + 1 :] = self.x - chol.T
        return pts

    def _measure(self, points: np.ndarray) -> np.ndarray:
        """h() applied to every sigma point at once -- one batched forward call."""
        return forward.predict(points[:, :N_POSE], self.params, self.model)

    # ------------------------------------------------------------------- API

    def initialize(self, counts: np.ndarray) -> None:
        """Seed from a Gauss-Newton solve. Sigma-point filters diverge from a
        bad start, so this is not optional in practice."""
        pose = solve_pose(counts, self.params, self.model, sigma=self.sigma)
        self.x = np.concatenate([pose, np.zeros(N_POSE)])
        self.P = np.diag(np.concatenate([np.full(N_POSE, 0.1), np.full(N_POSE, 10.0)]))

    def predict(self, dt: float) -> None:
        f = _process_matrix(dt)
        self.x = f @ self.x
        self.P = f @ self.P @ f.T + _process_noise(dt, self.accel_std)

    def update(self, counts: np.ndarray) -> None:
        counts = np.asarray(counts, dtype=float)
        pts = self._sigma_points()
        z = self._measure(pts)

        z_mean = self.wm @ z
        dz = z - z_mean
        dx = pts - self.x

        pzz = (dz * self.wc[:, None]).T @ dz + self.R
        pxz = (dx * self.wc[:, None]).T @ dz

        gain = np.linalg.solve(pzz.T, pxz.T).T
        self.x = self.x + gain @ (counts - z_mean)
        self.P = self.P - gain @ pzz @ gain.T
        # Keep it symmetric; asymmetry accumulates and eventually breaks Cholesky.
        self.P = 0.5 * (self.P + self.P.T)

    def step(self, counts: np.ndarray, dt: float) -> np.ndarray:
        """One predict/update cycle. Returns the current pose estimate."""
        self.predict(dt)
        self.update(counts)
        return self.pose

    @property
    def pose(self) -> np.ndarray:
        return self.x[:N_POSE].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[N_POSE:].copy()


def run(
    counts: np.ndarray,
    params: CalibParams,
    sigma: np.ndarray,
    dt: float,
    model: MagnetModel | None = None,
    **kwargs,
) -> np.ndarray:
    """Filter a whole recording offline. Returns (N, 6) poses."""
    counts = np.asarray(counts, dtype=float)
    filt = UKF(params=params, sigma=sigma, model=model, **kwargs)
    filt.initialize(counts[0])

    out = np.empty((len(counts), N_POSE))
    out[0] = filt.pose
    for i in range(1, len(counts)):
        out[i] = filt.step(counts[i], dt)
    return out


def solve_all(
    counts: np.ndarray,
    params: CalibParams,
    model: MagnetModel | None = None,
    sigma: np.ndarray | None = None,
) -> np.ndarray:
    """Per-frame Gauss-Newton over a whole recording, warm-started.

    The no-dynamics baseline: whatever the UKF achieves has to beat this to
    justify its complexity.
    """
    counts = np.asarray(counts, dtype=float)
    out = np.empty((len(counts), N_POSE))
    guess = None
    for i, frame in enumerate(counts):
        guess = solve_pose(frame, params, model, guess=guess, sigma=sigma)
        out[i] = guess
    return out
