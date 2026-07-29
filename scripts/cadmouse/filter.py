"""Recursive pose estimation: a UKF and an iterated EKF behind one interface.

Two implementations on purpose, and the reason is the port rather than
indecision. The RP2350 filter will be hand-written -- there is no mature
allocation-free ``no_std`` Rust UKF, and the one filter that looks likely to
win here, the iterated EKF, is not offered by any crate at all. A hand-written
filter needs something independent to be checked against, so the UKF here wraps
**FilterPy**: mature, widely used, and with a published derivation, so a
disagreement can be adjudicated instead of guessed at.

Which one should ship is an empirical question this module is meant to settle.
The prior is that the IEKF wins: the measurement Jacobian's singular values
span only a factor of 2.5, and the posterior is about 6 micrometres wide, over
which ``h`` is essentially linear. The nonlinearity in this problem lives at the
millimetre scale, not at the scale the filter actually explores. The IEKF also
costs roughly a third as much, which decides whether velocity states fit in the
2 kHz budget.

**There is no ground truth in any recorded session**, so accuracy cannot be
measured here -- only consistency. Tuning is against the normalised innovation
squared, which should follow a chi-squared distribution with nine degrees of
freedom if the filter's own uncertainty estimate is honest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .dataset import HELDOUT_SEGMENT, Run, load_session
from .magnet import FieldTable, build_table
from .model import MEAS_DIM, POSE_DIM, forward, forward_and_jac_vector, solve_pose
from .params import CalibParams

#: Process noise, as a random-walk power spectral density on the pose. Units
#: are mm^2/s and rad^2/s, so the variance added over one step is this times dt.
#:
#: Tuned on the held-out ``free`` segment against the normalised innovation
#: squared, which is the only thing available: with no ground truth, a filter
#: can be checked for honesty about its own uncertainty but not for accuracy.
#: These give a mean NIS of about 9.7 against a target of 9.
#:
#: Getting this wrong is not a mild mistuning. Three orders of magnitude too
#: large puts the UKF's sigma points several millimetres from the mean --
#: outside anything the mechanism can reach, and outside the field table --
#: and the filter diverges. The IEKF conceals the same error by linearising at
#: the mean, which is worth knowing before trusting it.
DEFAULT_Q_POS = 0.02
DEFAULT_Q_ROT = 3.0e-4

#: Sigma-point spread. Emphatically *not* the alpha = 1e-3 that gets copied
#: from the literature: at n = 6 that puts lambda at -5.999994, so the mean
#: weight becomes -1e6 and the covariance stops being positive definite in
#: f32 long before it does in f64. With alpha = 1, kappa = 0 the weights are
#: lambda = 0 and 1/12 each, which is well conditioned and -- since h is very
#: nearly linear over the posterior -- loses nothing.
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 2.0
DEFAULT_KAPPA = 0.0

#: Initial pose uncertainty. Modest, because the filter is started from a
#: direct Gauss-Newton solve of the first frame, which is good to a few
#: micrometres. A generous initial covariance is not the harmless conservatism
#: it looks like: the UKF spreads its sigma points by sqrt((n + lambda) P), so
#: an initial variance of 1 mm^2 places them 2.4 mm out on the very first step.
INITIAL_POS_VAR = 0.05**2
INITIAL_ROT_VAR = np.deg2rad(0.2) ** 2


@dataclass
class FilterConfig:
    q_pos: float = DEFAULT_Q_POS
    q_rot: float = DEFAULT_Q_ROT
    #: Multiplies the measurement covariance, to cover error the sensor noise
    #: model does not.
    #:
    #: What that error actually is, measured rather than assumed: on the
    #: held-out segment the innovations stay autocorrelated at rho ~ 0.45 out
    #: past 100 frames, while at rest they are white with an rms of 0.99
    #: counts against a sensor sigma of 1.08. So the sensor model is right and
    #: the excess -- sqrt(1.44^2 - 1.08^2) = 0.95 counts -- is systematic and
    #: pose-dependent, matching the calibration's own held-out residual of
    #: 0.89. It is residual geometry error, and inflating R only papers over
    #: it; the real fix is a better model.
    #:
    #: It is *not* the sequential I2C readout, despite the obvious suspicion:
    #: NIS correlates with knob speed at r = 0.03, and at the measured peak of
    #: 19.5 mm/s the ~333 us of skew is worth under 1.5 counts.
    r_inflation: float = 1.0
    iterations: int = 3  # IEKF only

    def process_noise(self, dt: float) -> np.ndarray:
        return np.diag([self.q_pos * dt] * 3 + [self.q_rot * dt] * 3)


class IteratedEkf:
    """Iterated EKF on the six-element pose vector.

    The iteration is Gauss-Newton on the maximum-a-posteriori estimate: each
    pass relinearises about the current estimate while keeping the prior fixed,
    which is what separates it from a plain EKF taking one linearised step from
    a stale point.
    """

    def __init__(
        self,
        params: CalibParams,
        table: FieldTable,
        sigma: np.ndarray,
        config: FilterConfig,
        initial_pose: np.ndarray | None = None,
    ):
        self.params = params
        self.table = table
        self.config = config
        self.R = np.diag((sigma * np.sqrt(config.r_inflation)) ** 2)
        self.x = np.zeros(POSE_DIM) if initial_pose is None else np.asarray(initial_pose, float).copy()
        self.P = np.diag([INITIAL_POS_VAR] * 3 + [INITIAL_ROT_VAR] * 3)
        self.innovation = np.zeros(MEAS_DIM)
        self.innovation_cov = self.R.copy()

    def predict(self, dt: float) -> None:
        # Random walk: the state is unchanged, only its uncertainty grows.
        self.P = self.P + self.config.process_noise(dt)

    def update(self, z: np.ndarray) -> None:
        prior_x = self.x.copy()
        prior_P = self.P
        x = prior_x.copy()
        gain = None
        jac = None

        for iteration in range(max(1, self.config.iterations)):
            predicted, jac = forward_and_jac_vector(x, self.params, self.table)
            innovation_cov = jac @ prior_P @ jac.T + self.R
            gain = np.linalg.solve(innovation_cov, jac @ prior_P).T
            # The (prior_x - x) term is what makes this iterated rather than
            # merely repeated: it keeps the update anchored to the prior mean
            # while the linearisation point moves.
            residual = z - predicted - jac @ (prior_x - x)
            x = prior_x + gain @ residual
            if iteration == 0:
                # The innovation is a *prior* quantity: measurement minus what
                # the prediction expected, before any of it is absorbed.
                # Recording the last iteration instead yields a post-fit
                # residual, which is small by construction, makes the NIS
                # meaningless, and flatters the IEKF against a UKF that
                # reports the real thing.
                self.innovation = z - predicted
                self.innovation_cov = innovation_cov

        eye = np.eye(POSE_DIM)
        self.x = x
        # Joseph form: stays symmetric and positive definite under f32, which
        # the simple (I - KH)P does not reliably do.
        self.P = (eye - gain @ jac) @ prior_P @ (eye - gain @ jac).T + gain @ self.R @ gain.T


class FilterPyUkf:
    """FilterPy's UKF, wrapped to the same interface. The oracle, not the port."""

    def __init__(
        self,
        params: CalibParams,
        table: FieldTable,
        sigma: np.ndarray,
        config: FilterConfig,
        initial_pose: np.ndarray | None = None,
    ):
        from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

        self.params = params
        self.table = table
        self.config = config

        points = MerweScaledSigmaPoints(
            POSE_DIM, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, kappa=DEFAULT_KAPPA
        )
        self._ukf = UnscentedKalmanFilter(
            dim_x=POSE_DIM,
            dim_z=MEAS_DIM,
            dt=1.0,
            hx=lambda x: forward(x, params, table),
            fx=lambda x, dt: x,
            points=points,
        )
        self._ukf.x = (
            np.zeros(POSE_DIM) if initial_pose is None else np.asarray(initial_pose, float).copy()
        )
        self._ukf.P = np.diag([INITIAL_POS_VAR] * 3 + [INITIAL_ROT_VAR] * 3)
        self._ukf.R = np.diag((sigma * np.sqrt(config.r_inflation)) ** 2)

    @property
    def x(self) -> np.ndarray:
        return self._ukf.x

    @property
    def P(self) -> np.ndarray:
        return self._ukf.P

    @property
    def innovation(self) -> np.ndarray:
        return self._ukf.y

    @property
    def innovation_cov(self) -> np.ndarray:
        return self._ukf.S

    def predict(self, dt: float) -> None:
        self._ukf.Q = self.config.process_noise(dt)
        self._ukf.predict(dt=dt)

    def update(self, z: np.ndarray) -> None:
        self._ukf.update(z)


ESTIMATORS = {"iekf": IteratedEkf, "ukf": FilterPyUkf}


# --------------------------------------------------------------------------
# Replay and consistency
# --------------------------------------------------------------------------


@dataclass
class ReplayResult:
    kind: str
    poses: np.ndarray  # (n, 6)
    nis: np.ndarray  # (n,)
    innovations: np.ndarray  # (n, 9)
    seconds: float

    @property
    def mean_nis(self) -> float:
        """Should be 9 -- one per measurement channel -- if R and Q are honest."""
        return float(self.nis.mean())

    def inside_band(self, confidence: float = 0.95) -> float:
        lo, hi = chi2.ppf([(1 - confidence) / 2, 1 - (1 - confidence) / 2], MEAS_DIM)
        return float(np.mean((self.nis >= lo) & (self.nis <= hi)))

    def summary(self) -> str:
        return (
            f"{self.kind:5}  frames {len(self.nis):6d}  "
            f"mean NIS {self.mean_nis:7.2f} (target {MEAS_DIM})  "
            f"in 95% band {self.inside_band():5.1%}  "
            f"innov rms {np.sqrt((self.innovations**2).mean()):5.2f} counts  "
            f"{self.seconds:5.1f} s"
        )


def replay(
    run: Run,
    params: CalibParams,
    table: FieldTable,
    sigma: np.ndarray,
    config: FilterConfig,
    kind: str = "iekf",
    stride: int = 1,
) -> ReplayResult:
    """Run a filter over one recorded run, collecting consistency statistics."""
    import time

    counts = run.counts[::stride]
    times = run.t_us[::stride]

    # Start from the direct solution so the transient does not pollute the NIS.
    first, _ = solve_pose(counts[0], params, table, sigma=sigma)
    estimator = ESTIMATORS[kind](params, table, sigma, config, initial_pose=first)

    poses = np.zeros((len(counts), POSE_DIM))
    nis = np.zeros(len(counts))
    innovations = np.zeros((len(counts), MEAS_DIM))

    start = time.perf_counter()
    for k in range(len(counts)):
        dt = (times[k] - times[k - 1]) / 1e6 if k else 0.0
        if dt > 0:
            estimator.predict(dt)
        estimator.update(counts[k])
        poses[k] = estimator.x
        innovations[k] = estimator.innovation
        nis[k] = estimator.innovation @ np.linalg.solve(
            estimator.innovation_cov, estimator.innovation
        )
    seconds = time.perf_counter() - start

    return ReplayResult(kind, poses, nis, innovations, seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--segment", default=HELDOUT_SEGMENT)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--q-pos", type=float, default=DEFAULT_Q_POS)
    parser.add_argument("--q-rot", type=float, default=DEFAULT_Q_ROT)
    parser.add_argument("--r-inflation", type=float, default=1.0)
    args = parser.parse_args(argv)

    session = load_session(args.session)
    params = CalibParams.load(args.calibration)
    table = build_table()
    sigma = session.noise_sigma()
    run = session.by_segment(args.segment)[0]

    config = FilterConfig(
        q_pos=args.q_pos, q_rot=args.q_rot, r_inflation=args.r_inflation
    )
    for kind in ("iekf", "ukf"):
        print(
            replay(
                run, params, table, sigma, config, kind=kind, stride=args.stride
            ).summary()
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
