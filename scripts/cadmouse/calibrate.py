"""Ground-truth-free calibration by joint bundle adjustment.

There is no reference measurement of the knob's pose, so the per-sample poses are
solved *jointly* with the calibration parameters.

Known ring geometry alone does *not* make the result metric: pose scale trades
against gain almost freely (see `scale_residual`). Two priors close that. The
gauge, ``||m_1|| = 1``, fixes how scale is split between moments and gains; the
physical gain prior fixes the scale itself, and with it the pose's meaning in
millimetres.

Poses are not free per sample. Each guided segment gets one motion *direction*
plus a scalar amplitude per sample -- see `SegmentPoses` for why the free
version does not converge.

Residual blocks:

===================  ======  ==================================================
block                 rows   purpose
===================  ======  ==================================================
field                  9 N   ``(predict - measured) / sigma``
coupling prior           30  bound the per-segment cross-coupling ratios
gauge                     1  ``||m_1|| = 1``, fixing the moment/gain split
scale                     1  physical prior on gain, fixing the pose scale
===================  ======  ==================================================

Validated on synthetic sessions with known parameters: a blind fit recovers pose
to ~0.03 mm and ~0.07 degrees, which is the Cramer-Rao bound for this geometry
(`observability.py`), and recovers the metric travel to within 1 %.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from . import forward, geometry
from .magnet import MagnetModel
from .params import CalibParams

#: Segment label -> the one DOF the user was asked to move.
SEGMENT_DOF = {name: i for i, name in enumerate(geometry.DOF_NAMES)}

REST_SEGMENT = "rest"
HELDOUT_SEGMENT = "free"

#: Rough full-scale knob travel, in millimetres then radians. Not a constraint:
#: it only sets the search range for `_match_field_excursion` and makes
#: translation and rotation comparable in `CalibrationResult.cross_coupling`.
#: The actual travel is whatever the fit recovers.
POSE_SCALE = np.array([3.0, 3.0, 3.0, 0.15, 0.15, 0.15])

DEFAULT_WEIGHTS = {
    # Soft: the mechanism genuinely cross-couples, so this bounds the coupling
    # ratios rather than forbidding them.
    "coupling": 5.0,
    "gauge": 1e3,  # effectively hard; a pure gauge choice, not data
    # Physical, not a gauge: this one carries real prior information, so it is
    # weighted like a measurement rather than a constraint. See `scale_residual`.
    "scale": 30.0,
}


@dataclass
class CalibrationData:
    """A recorded session: raw counts plus the segment each sample belongs to."""

    counts: np.ndarray  # (N, 9) raw i16 counts
    segments: np.ndarray  # (N,) segment label per sample

    def __post_init__(self) -> None:
        self.counts = np.asarray(self.counts, dtype=float).reshape(-1, geometry.N_CHANNELS)
        self.segments = np.asarray(self.segments, dtype=object).reshape(-1)
        if len(self.counts) != len(self.segments):
            raise ValueError("counts and segments must have the same length")

    def mask(self, *labels: str) -> np.ndarray:
        return np.isin(self.segments, labels)

    def subset(self, mask: np.ndarray) -> CalibrationData:
        return CalibrationData(self.counts[mask], self.segments[mask])

    @property
    def fitted(self) -> CalibrationData:
        """Everything except the held-out `free` segment."""
        return self.subset(self.segments != HELDOUT_SEGMENT)

    @property
    def heldout(self) -> CalibrationData:
        return self.subset(self.segments == HELDOUT_SEGMENT)

    def subsample(self, n: int, rng: np.random.Generator | None = None) -> CalibrationData:
        """Evenly thin the data, preserving the proportion of each segment."""
        if n >= len(self.counts):
            return self
        rng = rng or np.random.default_rng(0)
        keep = []
        for label in np.unique(self.segments):
            idx = np.flatnonzero(self.segments == label)
            take = max(1, round(n * len(idx) / len(self.segments)))
            keep.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
        return self.subset(np.sort(np.concatenate(keep)))


def load_session(path: str | Path) -> tuple[CalibrationData, np.ndarray, np.ndarray]:
    """Read a CSV written by ``record.py``.

    Returns ``(data, seq, t_us)``. The sequence numbers are carried out
    separately so callers can audit frame loss without the model code having to
    care about it.
    """
    rows = np.genfromtxt(path, delimiter=",", dtype=None, encoding="utf-8", names=True)
    segments = np.array([str(s) for s in rows["segment"]], dtype=object)
    counts = np.stack([rows[name] for name in geometry.CHANNEL_NAMES], axis=-1)
    return CalibrationData(counts, segments), rows["seq"], rows["t_us"]


def estimate_noise(data: CalibrationData) -> np.ndarray:
    """Per-channel measurement sigma, in counts, from the `rest` segments.

    The knob is mechanically still during `rest`, so all remaining variation is
    sensor noise. This is also what seeds the UKF's R.
    """
    rest = data.counts[data.mask(REST_SEGMENT)]
    if len(rest) < 2:
        return np.ones(geometry.N_CHANNELS)
    return np.maximum(rest.std(axis=0), 1e-3)


def check_saturation(data: CalibrationData) -> np.ndarray:
    """Indices of frames where any channel railed the ADC. Those cannot be fitted."""
    return np.flatnonzero(geometry.is_saturated(data.counts))


# --------------------------------------------------------------------- residuals


class SegmentPoses:
    """Pose parameterisation: one direction per segment, one amplitude per sample.

    **Why not a free 6-vector per sample.** That was the original design and it
    does not work: 500 samples give 4500 field residuals against 45 + 3000
    unknowns, barely 1.5 per unknown. The fit then drives the residual *below*
    the noise floor by absorbing noise into the poses, and lands on a wrong
    calibration -- measured: correct gain magnitude but scrambled gain structure,
    inverted magnet polarities, and poses ten times too large.

    A guided sweep does not explore six dimensions. The knob traces a 1-D curve:
    the driven axis, plus whatever the mechanism cross-couples into the other
    five. So a segment needs *one* direction vector, shared by all its samples,
    and one scalar amplitude per sample. That is 45 + 30 + N unknowns instead of
    45 + 6N -- about ten residuals per unknown, and it states the physical truth
    that the earlier soft prior could only gesture at.

    The driven component is pinned to 1, so an amplitude is directly the driven
    DOF's displacement in millimetres or radians, and the remaining five
    components of each direction are pure cross-coupling ratios.

    `rest` samples are held at exactly zero -- they are what defines the pose
    origin, and they constrain the sensor biases directly.
    """

    #: Free direction components per segment: the five that are not the driven one.
    N_DIRECTION_FREE = 5

    def __init__(self, segments: np.ndarray):
        self.segments = segments
        self.seg_index = np.array(
            [SEGMENT_DOF.get(str(s), -1) for s in segments], dtype=int
        )
        self.active = self.seg_index >= 0
        self.n_active = int(self.active.sum())
        self.active_seg = self.seg_index[self.active]

        # Column indices of the five free components of each segment's direction.
        self._free_cols = np.array(
            [[j for j in range(6) if j != k] for k in range(6)], dtype=int
        )

    @property
    def n_free(self) -> int:
        return 6 * self.N_DIRECTION_FREE + self.n_active

    def directions(self, packed: np.ndarray) -> np.ndarray:
        """(6, 6) direction matrix, driven component pinned to 1."""
        free = packed[: 6 * self.N_DIRECTION_FREE].reshape(6, self.N_DIRECTION_FREE)
        dirs = np.eye(6)
        for k in range(6):
            dirs[k, self._free_cols[k]] = free[k]
        return dirs

    def amplitudes(self, packed: np.ndarray) -> np.ndarray:
        return packed[6 * self.N_DIRECTION_FREE :]

    def poses(self, packed: np.ndarray) -> np.ndarray:
        """(N, 6) poses for every sample, rest samples exactly zero."""
        dirs = self.directions(packed)
        out = np.zeros((len(self.segments), 6))
        out[self.active] = self.amplitudes(packed)[:, None] * dirs[self.active_seg]
        return out

    def pack(self, directions: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
        free = np.stack(
            [directions[k, self._free_cols[k]] for k in range(6)]
        ).ravel()
        return np.concatenate([free, amplitudes])


def gauge_residual(params: CalibParams) -> float:
    """Pin ||m_1|| = 1, removing the moment/gain split degeneracy.

    This fixes only *how the scale is shared* between moments and gains. It does
    not fix the scale itself -- that is `scale_residual`'s job.
    """
    return float(np.linalg.norm(params.moments[0]) - 1.0)


def scale_residual(params: CalibParams) -> float:
    """Pull the overall gain toward its physically predicted magnitude.

    Without this the calibration has a near-degenerate direction: scaling every
    pose by *s* and dividing every gain by *s* changes the prediction only
    through the model's curvature, which over a few millimetres of travel is a
    weak effect. Fits left free in this direction reach the noise floor while
    recovering poses wrong by more than the knob's entire travel -- they explain
    the data perfectly in the wrong units.

    `geometry.expected_gain_counts_per_unit` closes it using two numbers that
    owe nothing to the fit: the sensor's datasheet sensitivity and the magnet's
    moment. Expressed in log space so the prior is symmetric in *ratio*, which
    is how the uncertainty actually behaves.
    """
    expected = geometry.expected_gain_counts_per_unit()
    # `gain_magnitudes`, not `gain_scales`: the physical prediction knows nothing
    # about which axis convention the fit happens to be carrying, and the signed
    # projection would cancel against a mismatched orientation.
    actual = float(params.gain_magnitudes.mean())
    if actual <= 0:
        return 10.0
    return float(np.log(actual / expected) / geometry.GAIN_PRIOR_REL_SIGMA)


@dataclass
class _Problem:
    """Everything the residual function needs, assembled once.

    Unknown vector layout: ``[calibration | segment directions | amplitudes]``.
    """

    counts: np.ndarray
    sigma: np.ndarray
    seg: SegmentPoses
    template: CalibParams
    model: MagnetModel
    weights: dict
    n_calib: int

    @property
    def n_samples(self) -> int:
        return len(self.counts)

    def split(self, x: np.ndarray) -> tuple[CalibParams, np.ndarray]:
        params = self.template.unpack(x[: self.n_calib])
        return params, self.seg.poses(x[self.n_calib :])

    def residuals(self, x: np.ndarray) -> np.ndarray:
        params, poses = self.split(x)
        pred = forward.predict(poses, params, self.model)
        field_res = ((pred - self.counts) / self.sigma).ravel()

        # Cross-coupling is real but small: a gimbal that bled 100 % of one axis
        # into another would not be a 6-DOF mechanism. This keeps the direction
        # vectors from wandering into implausible shapes without forbidding the
        # coupling that genuinely exists.
        packed = x[self.n_calib :]
        free = packed[: 6 * SegmentPoses.N_DIRECTION_FREE]
        coupling = self.weights["coupling"] * free

        gauge = np.array(
            [
                self.weights["gauge"] * gauge_residual(params),
                self.weights["scale"] * scale_residual(params),
            ]
        )
        return np.concatenate([field_res, coupling, gauge])

    def sparsity(self) -> sparse.csr_matrix:
        """Which residuals depend on which unknowns.

        Without this the Jacobian is dense and the solve is hopeless; with it,
        SciPy needs a fixed number of function evaluations per Jacobian
        regardless of how many samples there are.
        """
        n, nc = self.n_samples, self.n_calib
        n_dir = 6 * SegmentPoses.N_DIRECTION_FREE
        rows = 9 * n + n_dir + 2
        cols = nc + self.seg.n_free
        m = sparse.lil_matrix((rows, cols), dtype=np.int8)

        # Field block: every calibration parameter always matters.
        m[: 9 * n, :nc] = 1

        # ...plus, for an active sample, its own amplitude and its segment's
        # five direction components. Rest samples have a pose fixed at zero, so
        # they depend on the calibration alone -- which is what makes them such
        # a clean constraint on the biases.
        active_rows = np.flatnonzero(self.seg.active)
        amp_col0 = nc + n_dir
        for row_i, sample in enumerate(active_rows):
            seg = self.seg.seg_index[sample]
            block = slice(9 * sample, 9 * sample + 9)
            m[block, amp_col0 + row_i] = 1
            m[block, nc + seg * SegmentPoses.N_DIRECTION_FREE :
                     nc + (seg + 1) * SegmentPoses.N_DIRECTION_FREE] = 1

        # Coupling prior: diagonal in the direction parameters.
        base = 9 * n
        m[base + np.arange(n_dir), nc + np.arange(n_dir)] = 1

        # Gauge and scale priors depend on the calibration block only.
        m[-2:, :nc] = 1

        return m.tocsr()


@dataclass
class CalibrationResult:
    params: CalibParams
    poses: np.ndarray
    data: CalibrationData
    sigma: np.ndarray
    model: MagnetModel
    cost: float
    n_iterations: int
    elapsed_s: float
    #: Driven-axis displacement per active sample, in mm or radians.
    amplitudes: np.ndarray | None = None
    #: (6, 6) per-segment motion directions; row *k* has a 1 in column *k* and
    #: the fitted cross-coupling ratios elsewhere.
    directions: np.ndarray | None = None
    orientation: np.ndarray | None = None
    magnet_signs: np.ndarray | None = None
    log: list[str] = field(default_factory=list)

    def cross_coupling(self) -> np.ndarray:
        """Largest off-axis bleed per commanded axis, dimensionless. Shape (6,).

        How much the mechanism moves the other five DOFs when you drive one -- a
        measured version of the axis bleed the C++ firmware's README complains
        about, rather than something inferred from the output.

        Expressed as *fraction of the other axis's full travel per full travel of
        the driven one*. The raw direction components cannot be read as
        percentages: they mix millimetres with radians, so a tilt that lifts a
        magnet a millimetre at a 16 mm radius reads as "175 %" of nothing
        meaningful. Scaling by `POSE_SCALE` makes the axes comparable.
        """
        if self.directions is None:
            return np.zeros(6)
        off = self.directions.copy()
        np.fill_diagonal(off, 0.0)
        scaled = np.abs(off) * (POSE_SCALE[None, :] ** -1) * POSE_SCALE[:, None]
        return scaled.max(axis=1)

    def residual_counts(self) -> np.ndarray:
        """Per-sample field residual in raw counts. (N, 9)."""
        return forward.predict(self.poses, self.params, self.model) - self.data.counts

    def rms_per_channel(self) -> np.ndarray:
        return np.sqrt((self.residual_counts() ** 2).mean(axis=0))

    def rms(self) -> float:
        return float(np.sqrt((self.residual_counts() ** 2).mean()))


# ------------------------------------------------------------------- the solver


def initial_amplitudes(data: CalibrationData, signs: np.ndarray) -> np.ndarray:
    """Starting amplitudes from the segment structure alone.

    Within a single-axis segment the knob traces a 1-D path, so the leading
    principal component of the (rest-subtracted) counts is a monotone proxy for
    how far along that axis it is. Scaling to `POSE_SCALE` gives a starting
    amplitude in millimetres or radians.

    The sign of a principal component is arbitrary -- that is precisely the
    ambiguity `search_signs` resolves.

    **The amplitude scale is set by physics, not by a guess.** A generic
    full-travel constant is up to 3x wrong -- this knob moves ~1 mm in z where a
    nominal guess might say 3 -- and the resulting error propagates straight into
    the linear gain solve, which then fights the physical scale prior. Instead,
    each segment's amplitude is scaled so the field excursion the model predicts
    matches the excursion actually measured, using the datasheet-and-magnet gain
    from `geometry.expected_gain_counts_per_unit`.
    """
    seg = SegmentPoses(data.segments)
    rest_mean = data.counts[data.mask(REST_SEGMENT)].mean(axis=0)
    amplitudes = np.zeros(seg.n_active)

    probe = CalibParams.initial(
        gain_counts_per_unit=geometry.expected_gain_counts_per_unit()
    )
    active_positions = np.flatnonzero(seg.active)

    for label, dof in SEGMENT_DOF.items():
        sample_idx = np.flatnonzero(data.segments == label)
        if len(sample_idx) < 3:
            continue
        block = data.counts[sample_idx] - rest_mean
        block = block - block.mean(axis=0)
        _, _, vt = np.linalg.svd(block, full_matrices=False)
        proj = block @ vt[0]
        peak = np.abs(proj).max()
        if peak <= 0:
            continue

        # How far along this axis reproduces the observed field excursion?
        span = _match_field_excursion(dof, float(block.std()), probe)
        where = np.searchsorted(active_positions, sample_idx)
        amplitudes[where] = signs[dof] * span * proj / peak
    return amplitudes


def _match_field_excursion(
    dof: int, measured_std: float, probe: CalibParams, n_grid: int = 24
) -> float:
    """Amplitude along `dof` whose predicted field spread matches `measured_std`.

    A 1-D search along one axis with everything else nominal. It only has to be
    right to a factor of about two -- enough that the first linear gain solve is
    not badly scaled.
    """
    trials = np.linspace(0.05, 1.0, n_grid) * POSE_SCALE[dof]
    spreads = np.array(
        [
            forward.predict(
                np.stack([np.eye(6)[dof] * a, -np.eye(6)[dof] * a]), probe,
                forward.COARSE_MODEL,
            ).std()
            for a in trials
        ]
    )
    if not np.any(spreads > 0):
        return POSE_SCALE[dof]
    return float(trials[np.argmin(np.abs(spreads - measured_std))])


def refine(
    data: CalibrationData,
    initial: CalibParams,
    model: MagnetModel | None = None,
    sigma: np.ndarray | None = None,
    amplitudes: np.ndarray | None = None,
    directions: np.ndarray | None = None,
    weights: dict | None = None,
    max_nfev: int | None = 120,
    verbose: int = 0,
) -> CalibrationResult:
    """Run the joint bundle adjustment from a given starting point.

    Poses are parameterised by `SegmentPoses`, not one free 6-vector per sample;
    see that class for why the free version does not converge.

    `max_nfev` is capped by default. SciPy's own default is ``100 * n``, which
    here means an uncapped solve chases the last fraction of a count for hours.
    The residual is flat long before that.
    """
    model = model or forward.DEFAULT_MODEL
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    sigma = estimate_noise(data) if sigma is None else np.asarray(sigma, dtype=float)

    seg = SegmentPoses(data.segments)
    problem = _Problem(
        counts=data.counts,
        sigma=sigma,
        seg=seg,
        template=initial,
        model=model,
        weights=weights,
        n_calib=initial.n_free,
    )

    if amplitudes is None:
        amplitudes = np.zeros(seg.n_active)
    if directions is None:
        directions = np.eye(6)
    x0 = np.concatenate([initial.pack(), seg.pack(directions, amplitudes)])

    started = time.perf_counter()
    sol = least_squares(
        problem.residuals,
        x0,
        jac_sparsity=problem.sparsity(),
        method="trf",
        x_scale="jac",
        max_nfev=max_nfev,
        verbose=verbose,
    )
    elapsed = time.perf_counter() - started

    params, fitted_poses = problem.split(sol.x)
    packed = sol.x[problem.n_calib :]
    return CalibrationResult(
        params=params,
        poses=fitted_poses,
        data=data,
        sigma=sigma,
        model=model,
        cost=float(sol.cost),
        n_iterations=int(sol.nfev),
        elapsed_s=elapsed,
        amplitudes=seg.amplitudes(packed),
        directions=seg.directions(packed),
    )


# ------------------------------------------------------------------ bootstrap
#
# A cold joint solve does not converge: from a random start it plateaus around
# 24 counts RMS where the true parameters sit at 1.6. The way out is that the
# model is *linear* in the pieces we do not know, one piece at a time:
#
#   * given poses and moments, ``z = A B + b`` is linear in (A, b);
#   * given poses and gains, it is linear in the moments (see
#     `forward.moment_basis`);
#   * given the parameters, each sample's pose is a cheap 6-parameter solve.
#
# Alternating exact linear solves gets within the joint solver's basin without
# any nonlinear search, and -- importantly -- makes the sensor orientation fall
# out of the linear solve for A rather than needing a discrete search at all.


def linear_gains(
    counts: np.ndarray, board: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Exact least-squares (gains, biases) given the board-frame field.

    ``z_i = A_i B_i + b_i`` decouples by sensor *and* by output component, so
    this is nine independent 4-unknown regressions, not one 36-unknown one.
    """
    counts = counts.reshape(-1, geometry.N_SENSORS, 3)
    n = len(counts)
    gains = np.empty((geometry.N_SENSORS, 3, 3))
    biases = np.empty((geometry.N_SENSORS, 3))

    for i in range(geometry.N_SENSORS):
        design = np.concatenate([board[:, i, :], np.ones((n, 1))], axis=1)
        sol, *_ = np.linalg.lstsq(design, counts[:, i, :], rcond=None)  # (4, 3)
        gains[i] = sol[:3].T
        biases[i] = sol[3]
    return gains, biases


def linear_moments(
    counts: np.ndarray, basis: np.ndarray, params: CalibParams
) -> np.ndarray:
    """Exact least-squares magnet moments given gains, biases, and poses."""
    n = len(counts)
    # Map each basis field through the gains to get counts per unit moment.
    design = np.einsum("sij,nsjk->nsik", params.gains, basis)
    design = design.reshape(n * geometry.N_CHANNELS, geometry.N_MAGNETS * 3)
    target = (counts.reshape(-1, geometry.N_SENSORS, 3) - params.biases).reshape(-1)
    sol, *_ = np.linalg.lstsq(design, target, rcond=None)
    return sol.reshape(geometry.N_MAGNETS, 3)


def fit_segment_directions(
    seg: SegmentPoses, poses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Best rank-1 fit of free poses to ``amplitude * direction`` per segment.

    Returns ``(directions (6, 6), amplitudes (n_active,))``.

    Taking only the driven-axis component and dropping the rest would throw away
    the cross-coupling the bootstrap just worked out -- and the cross-coupling is
    most of what distinguishes a real mechanism from an ideal one. Doing that
    turned a bootstrap sitting at 44 counts RMS into a starting point costing
    7e9, from which the joint solve could not recover.

    With the driven component pinned to 1, the amplitudes *are* the driven
    column, and each remaining component is a plain projection -- so this is
    exact, not another optimisation.
    """
    directions = np.eye(6)
    amplitudes = np.zeros(seg.n_active)
    active_poses = poses[seg.active]

    for k in range(6):
        rows = seg.active_seg == k
        if not rows.any():
            continue
        block = active_poses[rows]
        a = block[:, k]
        amplitudes[rows] = a
        denom = float(a @ a)
        if denom <= 0:
            continue
        for j in range(6):
            if j != k:
                directions[k, j] = float(a @ block[:, j]) / denom
    return directions, amplitudes


def apply_gauge(params: CalibParams) -> CalibParams:
    """Rescale to ``||m_1|| = 1`` without changing any prediction.

    The field is linear in the moments, so dividing every moment by *s* and
    multiplying every gain by *s* is exactly prediction-preserving. Only the
    split between the two moves.

    This has to happen before `refine`, not during it. `linear_moments` has no
    reason to land on a unit moment, and the gauge prior carries a weight of
    1e3 -- so a starting ``||m_1||`` of 1.4 contributes a squared residual of
    160000, dwarfing several thousand field residuals of order one. The
    optimiser then spends its budget fixing the gauge by rescaling gains and
    moments against each other, and the field fit gets *worse* than the
    bootstrap it started from. Normalising up front makes the prior start
    satisfied, so it only has to hold the gauge rather than establish it.
    """
    scale = float(np.linalg.norm(params.moments[0]))
    if scale <= 0:
        return params
    return replace(
        params, moments=params.moments / scale, gains=params.gains * scale
    )


def bootstrap(
    data: CalibrationData,
    signs: np.ndarray,
    model: MagnetModel | None = None,
    sigma: np.ndarray | None = None,
    rounds: int = 3,
) -> tuple[CalibParams, tuple[np.ndarray, np.ndarray], float]:
    """Alternating linear solve for a starting calibration.

    Returns ``(params, (directions, amplitudes), rms)``. No nonlinear search happens here
    beyond the per-sample pose refinement, which is well conditioned once the
    calibration is roughly right.

    Pose refinement runs *unconstrained* (a free 6-vector per sample), then the
    result is projected onto each segment's driven axis. That is deliberate:
    letting the pose move freely here is what lets the linear solves see the
    cross-coupling, while the projection keeps the handoff to `refine` in the
    parameterisation that actually converges.
    """
    from .ukf import solve_pose  # local import: ukf imports forward, not calibrate

    model = model or forward.COARSE_MODEL
    sigma = estimate_noise(data) if sigma is None else sigma

    seg = SegmentPoses(data.segments)
    params = CalibParams.initial(gain_counts_per_unit=1.0)

    poses = np.zeros((len(data.counts), 6))
    poses[seg.active, seg.active_seg] = initial_amplitudes(data, signs)

    for round_index in range(rounds):
        board = forward.board_field(poses, params, model)
        params.gains, params.biases = linear_gains(data.counts, board)

        basis = forward.moment_basis(poses, params, model)
        params.moments = linear_moments(data.counts, basis, params)

        # Re-solving poses is what couples the two linear solves together; skip
        # it on the last round since nothing consumes the result.
        if round_index < rounds - 1:
            poses = np.array(
                [
                    solve_pose(z, params, model, guess=p, sigma=sigma)
                    for z, p in zip(data.counts, poses)
                ]
            )
            # Rest samples define the origin; do not let them drift.
            poses[~seg.active] = 0.0

    directions, amplitudes = fit_segment_directions(seg, poses)
    # Report the RMS of the *reconstructed* poses, not the free ones: that is
    # what `refine` will actually start from, so it is the number that predicts
    # how the handover goes.
    rebuilt = seg.poses(seg.pack(directions, amplitudes))
    residual = forward.predict(rebuilt, params, model) - data.counts
    return (
        apply_gauge(params),
        (directions, amplitudes),
        float(np.sqrt((residual**2).mean())),
    )


def search_signs(
    data: CalibrationData,
    n_probe: int = 200,
    model: MagnetModel | None = None,
    verbose: bool = False,
) -> list[tuple[float, np.ndarray]]:
    """Exhaustive search over the 64 per-DOF pose sign conventions.

    These are the only genuinely discrete unknowns left. The sensor orientation
    is *not* among them -- it comes out of `linear_gains` continuously -- and the
    magnet polarities come out of `linear_moments`. What no linear solve can fix
    is which direction along each axis counts as positive, because the field
    response to +x and -x displacement differ and only one of them is consistent
    with the known ring geometry.

    Returns ``(rms, signs)`` sorted best first.
    """
    probe = data.fitted.subsample(n_probe)
    sigma = estimate_noise(data)

    results = []
    for bits in range(64):
        signs = np.array([1.0 if (bits >> k) & 1 == 0 else -1.0 for k in range(6)])
        _, _, rms = bootstrap(probe, signs, model=model, sigma=sigma, rounds=2)
        results.append((rms, signs))
        if verbose:
            print(f"  signs={signs.astype(int)} rms={rms:.2f}")

    return sorted(results, key=lambda r: r[0])


def _gain_guess(data: CalibrationData, model: MagnetModel) -> float:
    """Order-of-magnitude counts-per-model-unit, from the data's own spread.

    The absolute scale is a gauge choice, but the *starting* value has to be
    within an order of magnitude or the first Gauss-Newton step is meaningless.
    Compare the measured spread against the model's spread over a plausible
    sweep of poses.
    """
    probe = CalibParams.initial(gain_counts_per_unit=1.0)
    sweep = np.zeros((13, 6))
    for k in range(6):
        sweep[2 * k + 1, k] = POSE_SCALE[k]
        sweep[2 * k + 2, k] = -POSE_SCALE[k]

    model_spread = forward.predict(sweep, probe, model).std()
    measured_spread = data.counts.std()
    if model_spread <= 0:
        return 1.0
    return float(measured_spread / model_spread)


def calibrate(
    data: CalibrationData,
    n_probe: int = 150,
    n_fit: int = 900,
    n_candidates: int = 3,
    fine_nfev: int = 40,
    fit_magnet_offsets: bool = False,
    verbose: bool = False,
) -> CalibrationResult:
    """Full pipeline: sign search, linear bootstrap, then two-stage polish.

    Stages, and why each exists:

    1. `search_signs` -- the 64 pose sign conventions, the only discrete
       unknowns left once the linear solves handle orientation and polarity.
    2. `bootstrap` -- alternating exact linear solves, because a cold nonlinear
       start does not converge (it plateaus around 24 counts RMS where the true
       parameters sit near the noise floor).
    3. Joint refine under `forward.COARSE_MODEL` -- cheap point dipoles, enough
       to settle the pose/parameter coupling.
    4. Joint refine under `forward.DEFAULT_MODEL` -- pays for finite-size
       magnets, worth ~30 % of the signal at this geometry.

    The top `n_candidates` sign vectors are all carried through stage 3 and
    ranked on the result, because the bootstrap's own ranking separates the
    leaders by only a few percent -- too little to trust on its own.
    """
    log: list[str] = []

    def say(msg: str) -> None:
        log.append(msg)
        if verbose:
            print(msg, flush=True)

    saturated = check_saturation(data)
    if len(saturated):
        raise ValueError(
            f"{len(saturated)} of {len(data.counts)} frames railed the ADC "
            f"(|count| >= {geometry.ADC_FULL_SCALE_COUNTS}). Re-record with a "
            "lower sensitivity; these samples carry no usable information."
        )

    fitted = data.fitted
    if not fitted.mask(REST_SEGMENT).any():
        raise ValueError(
            "no 'rest' samples: they define the pose origin and the noise model"
        )

    sigma = estimate_noise(data)
    say(f"noise floor: {sigma.mean():.2f} counts RMS (from {REST_SEGMENT} segments)")

    say("searching 64 pose sign conventions...")
    ranked = search_signs(data, n_probe=n_probe)
    say(
        "  best "
        + ", ".join(f"{r:.1f}" for r, _ in ranked[:n_candidates])
        + f" (worst {ranked[-1][0]:.1f})"
    )

    fit_data = fitted.subsample(n_fit)
    say(f"bootstrap + coarse joint fit on {len(fit_data.counts)} samples...")

    best: CalibrationResult | None = None
    best_signs: np.ndarray | None = None
    for rms0, signs in ranked[:n_candidates]:
        start, (dirs, amps), _ = bootstrap(fit_data, signs, sigma=sigma)
        candidate = refine(
            fit_data,
            start,
            model=forward.COARSE_MODEL,
            sigma=sigma,
            amplitudes=amps,
            directions=dirs,
        )
        say(
            f"  signs={signs.astype(int)} bootstrap {rms0:6.2f} -> "
            f"joint {candidate.rms():7.3f} counts ({candidate.elapsed_s:.0f} s)"
        )
        if best is None or candidate.cost < best.cost:
            best, best_signs = candidate, signs

    assert best is not None and best_signs is not None
    polish_start = best.params
    if fit_magnet_offsets:
        # Released only for the final polish: nine extra parameters are easy to
        # confuse with pose early on, and pointless until the fit is close.
        polish_start = replace(polish_start, fit_magnet_offsets=True)
        say("polishing with finite-size magnets and free magnet positions...")
    else:
        say("polishing with finite-size magnets...")
    # Fewer iterations than the coarse stage: this starts from a converged
    # point-dipole solution and only has to absorb the finite-size correction,
    # and each iteration costs ~25x more.
    fine = refine(
        fit_data,
        polish_start,
        model=forward.DEFAULT_MODEL,
        sigma=sigma,
        amplitudes=best.amplitudes,
        directions=best.directions,
        max_nfev=fine_nfev,
    )
    say(
        f"  RMS {fine.rms():.3f} counts vs {sigma.mean():.2f} noise floor "
        f"({fine.elapsed_s:.0f} s)"
    )
    if fine.rms() > 3.0 * sigma.mean():
        say(
            "  WARNING: residual well above the noise floor. Either the magnet "
            "model is wrong for this hardware or the session lacks motion on "
            "some axis -- check the per-channel RMS and the residual scatter."
        )

    aniso = fine.params.gain_anisotropy()
    say(
        "  per-sensor gain anisotropy: "
        + ", ".join(f"mag{i + 1} {d:.1%}" for i, d in enumerate(aniso))
    )
    if np.any(aniso > 0.35):
        say(
            "  NOTE: a sensor's gain is far from a scaled rotation, meaning real "
            "per-axis mismatch or cross-axis sensitivity. That is a hardware "
            "property worth knowing, not a fitting artefact."
        )

    coupling = fine.cross_coupling()
    say(
        "  mechanism cross-coupling per axis: "
        + ", ".join(
            f"{geometry.DOF_NAMES[k]} {c:.0%}" for k, c in enumerate(coupling)
        )
    )

    fine.magnet_signs = best_signs
    fine.log = log
    return fine
