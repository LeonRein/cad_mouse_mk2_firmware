"""Fitting a device's calibration from one recorded session.

The awkward part of this problem, and the reason it is a bundle adjustment
rather than a regression, is that **the poses are unknown too**. The recording
asks for one axis at a time, but a fit that assumed the hand succeeded would be
fitting a fiction: across the six motion segments, the first principal
component explains only 80-97 % of the variance, so the off-axis bleed is real
and large. The poses are therefore latent variables, solved jointly with the 27
calibration parameters.

What pins the solution down, in the absence of any ground truth:

* The ``rest`` blocks define pose zero. That is the datum, and the only
  absolute pose knowledge in the session; the eight interleaved blocks also
  stop slow bias drift from aliasing onto one axis.
* Sensor positions are held fixed, which breaks the remaining gauge freedom --
  translating every magnet is otherwise indistinguishable from translating
  every sensor the other way.
* Weak priors aim each motion segment at the axis it was asked for, without
  forcing the bleed to zero.
* The ``free`` segment is excluded throughout and scored at the end.

Run it directly:

    uv run python -m cadmouse.calibrate data/session1.csv -o calibration.json
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from .dataset import HELDOUT_SEGMENT, REST_SEGMENT, Frame, Session, load_session
from .geometry import MOMENT_N35
from .magnet import FieldTable, build_table
from .model import (
    MEAS_DIM,
    POSE_DIM,
    TESLA_TO_COUNTS,
    evaluate_batch,
    forward_batch,
    pose_jacobian_batch,
    solve_pose,
)
from .params import CalibParams

#: Prior widths, one per block of :meth:`CalibParams.pack`. These are
#: regularisers, not constraints: they keep the fit from wandering into
#: physically silly territory while leaving it free to move as far as the data
#: insists. The moment prior is deliberately loose -- nominal N35 is already
#: known to be some 25 % too strong, so a tight prior there would fight the
#: data rather than guide it.
#:
#: The magnet position prior is anisotropic, and deliberately so. Ring radius
#: and angle are set by the magnet holder and are well controlled, but the
#: 12 mm ring depth is a design value that was never measured, so a tight prior
#: in z would be fighting the data rather than guiding it.
PRIOR_SIGMA = {
    "magnet_pos": np.array([0.5, 0.5, 2.0]),  # mm, per axis
    "magnet_tilt": np.deg2rad(3.0),  # rad
    "magnet_moment": 0.5 * MOMENT_N35,  # A*m^2
    "sensor_offset": 46.0,  # counts, about 3 mT
    "sensor_gain": 0.03,  # dimensionless
}

#: How far the knob is assumed to stray from the axis it was asked to move
#: along. Weak on purpose: whatever the mechanism bleeds into the other five
#: degrees of freedom is a property of the hardware worth measuring, not
#: something to force to zero.
OFF_AXIS_SIGMA_MM = 0.3
OFF_AXIS_SIGMA_RAD = np.deg2rad(2.0)

#: Ceiling on how fast the knob is assumed to move between two decimated
#: frames. Only strong enough to stop a single bad frame from producing a wild
#: pose that then drags the geometry.
MAX_SPEED_MM_S = 20.0
MAX_SPEED_RAD_S = np.deg2rad(200.0)

#: Residuals are in units of sigma, so this downweights anything past three of
#: them.
#:
#: The outliers it is for are rare but severe: at the instant a hand releases
#: the knob, the three sequential sensor reads straddle a discontinuous change
#: of pose and the frame is not a rigid-body pose at all -- up to 138 counts
#: out. Most are removed by :data:`~cadmouse.dataset.SETTLE_S`, and this
#: catches the rest before squared loss lets one of them drag the geometry.
#:
#: Note this is *not* needed for ordinary motion. The same sequential readout
#: during a normal traverse is worth well under a count: at the fastest speed
#: in the recorded data the knob moves 6.5 um between the first and last read.
ROBUST_F_SCALE = 3.0

#: Convergence tolerance, deliberately looser than scipy's 1e-8 default.
#:
#: The trust-region step is solved inexactly by ``lsmr``, so chasing 1e-8 costs
#: four times the iterations to reach the same answer -- 473 evaluations
#: against 111, for a cost that differs in its fifth significant figure and a
#: held-out residual that differs by 0.007 counts. There is also a physical
#: floor: a change this small moves the parameters far less than their own
#: uncertainty, so continuing is refining noise.
SOLVER_TOL = 1e-6


# --------------------------------------------------------------------------
# Polarity
# --------------------------------------------------------------------------


def detect_moment_signs(session: Session) -> np.ndarray:
    """Which way each magnet is magnetised, read straight off the rest blocks.

    Worth doing rather than assuming, because getting it wrong does not
    degrade the fit -- it destroys it, silently. Starting the joint fit with
    magnet 3's sign flipped drives all three moments to zero within one
    iteration and parks there: the poses come out garbage, the best linear
    explanation of garbage poses is "there is no field", and once the moments
    are zero the pose Jacobian is zero too, so nothing can recover. The signed
    moment makes the sign *representable*, not *findable*.

    The measurement is unambiguous. At rest each sensor sits under its own
    magnet and reads some 500 counts of its axial field, against at most ~46
    counts of sensor offset and ~14 counts of cross-talk from the other two.
    """
    rest = session.rest_mean().reshape(3, 3)
    axial = rest[:, 2]
    if np.any(np.abs(axial) < 150.0):
        raise ValueError(
            f"rest axial fields {axial} are too weak to read polarity from; "
            "is the knob actually at rest, and are the magnets fitted?"
        )
    return np.sign(axial)


def initial_params(session: Session) -> CalibParams:
    """Nominal geometry, with the magnet polarities taken from the data."""
    params = CalibParams.nominal()
    signs = detect_moment_signs(session)
    params.magnet_moment = np.abs(params.magnet_moment) * signs
    return params


# --------------------------------------------------------------------------
# Problem assembly
# --------------------------------------------------------------------------


@dataclass
class Problem:
    """The frames being fitted and the bookkeeping the residual needs."""

    frames: list[Frame]
    sigma: np.ndarray  # (9,) counts
    measurements: np.ndarray  # (n, 9)
    free_pose: np.ndarray  # (n,) bool -- rest frames are pinned at zero
    pose_slot: np.ndarray  # (n,) index into the pose block, -1 if pinned
    n_free: int
    param_mask: np.ndarray  # (36,) bool
    prior_centre: np.ndarray  # (36,)
    prior_sigma: np.ndarray  # (36,)
    smooth_pairs: np.ndarray  # (m, 2) indices of consecutive free frames
    smooth_sigma: np.ndarray  # (m, 6)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_params(self) -> int:
        return int(self.param_mask.sum())

    @property
    def n_unknowns(self) -> int:
        return self.n_params + self.n_free * POSE_DIM


def build_problem(
    session: Session,
    reference: CalibParams,
    per_segment: int = 400,
    per_rest_run: int = 150,
    fit_gain: bool = False,
) -> Problem:
    """Decimate, and work out what is free, what is pinned and what is priored.

    Rest frames are sampled more sparingly than motion frames on purpose. They
    are all but identical to each other, so taking as many of them as of the
    motion frames would weight the datum by thousands of near-duplicate
    measurements and let it dominate the geometry.
    """
    frames = [
        f
        for f in session.decimate(per_segment=per_segment, per_rest_run=per_rest_run)
        if f.segment != HELDOUT_SEGMENT
    ]
    sigma = session.noise_sigma()

    free_pose = np.array([not f.is_rest for f in frames])
    pose_slot = np.full(len(frames), -1, dtype=np.intp)
    pose_slot[free_pose] = np.arange(int(free_pose.sum()))

    mask = CalibParams.free_mask(fit_gain=fit_gain)
    prior_sigma = np.concatenate(
        [
            np.tile(PRIOR_SIGMA["magnet_pos"], 3),
            np.full(6, PRIOR_SIGMA["magnet_tilt"]),
            np.full(3, PRIOR_SIGMA["magnet_moment"]),
            np.full(9, PRIOR_SIGMA["sensor_offset"]),
            np.full(9, PRIOR_SIGMA["sensor_gain"]),
        ]
    )

    # Consecutive free frames from the same run, for the smoothness prior.
    pairs = []
    sigmas = []
    for a in range(len(frames) - 1):
        b = a + 1
        if not (free_pose[a] and free_pose[b]):
            continue
        if frames[a].run_index != frames[b].run_index:
            continue
        dt = (frames[b].t_us - frames[a].t_us) / 1e6
        pairs.append((pose_slot[a], pose_slot[b]))
        sigmas.append([MAX_SPEED_MM_S * dt] * 3 + [MAX_SPEED_RAD_S * dt] * 3)

    return Problem(
        frames=frames,
        sigma=sigma,
        measurements=np.array([f.counts for f in frames]),
        free_pose=free_pose,
        pose_slot=pose_slot,
        n_free=int(free_pose.sum()),
        param_mask=mask,
        prior_centre=reference.pack(),
        prior_sigma=prior_sigma,
        smooth_pairs=np.array(pairs, dtype=np.intp).reshape(-1, 2),
        smooth_sigma=np.array(sigmas, dtype=float).reshape(-1, POSE_DIM),
    )


def _poses_from(problem: Problem, pose_block: np.ndarray) -> np.ndarray:
    poses = np.zeros((problem.n_frames, POSE_DIM))
    poses[problem.free_pose] = pose_block.reshape(-1, POSE_DIM)
    return poses


def _split(problem: Problem, x: np.ndarray, base: np.ndarray):
    packed = base.copy()
    packed[problem.param_mask] = x[: problem.n_params]
    params = CalibParams.unpack(packed)
    poses = _poses_from(problem, x[problem.n_params :])
    return params, poses, packed


# --------------------------------------------------------------------------
# Residual and Jacobian
# --------------------------------------------------------------------------


def _param_jacobian(ev, params: CalibParams, tesla: np.ndarray) -> np.ndarray:
    """``d(counts)/d(calibration parameters)``, shape (n, 9, 36).

    All analytic. The field is evaluated at unit moment and scaled afterwards,
    which is what makes the moment column simply the unit field rather than
    something requiring a division by a parameter the fit might have driven
    towards zero.
    """
    n = ev.field.shape[0]
    moment = params.magnet_moment[None, None, :, None, None]
    grad_delta = ev.grad_delta * moment
    grad_axis = ev.grad_axis * moment

    out = np.zeros((n, 3, 3, 36))

    # Magnet positions: delta = sensor - (R p + t), so d(delta)/dp = -R.
    for j in range(3):
        out[:, :, :, 3 * j : 3 * j + 3] = -np.einsum(
            "nsoi,nic->nsoc", grad_delta[:, :, j], ev.rotation
        )

    # Magnet tilts, through the world-frame axis.
    d_axis_d_tilt = params.magnet_axes_jacobian()  # (j, 3, 2)
    for j in range(3):
        world = np.einsum("nab,bc->nac", ev.rotation, d_axis_d_tilt[j])  # (n, 3, 2)
        out[:, :, :, 9 + 2 * j : 9 + 2 * j + 2] = np.einsum(
            "nsoi,nic->nsoc", grad_axis[:, :, j], world
        )

    # Moments: the field is linear in them, so the derivative is the unit field.
    out[:, :, :, 15:18] = np.transpose(ev.field, (0, 1, 3, 2))

    # Everything above is a field derivative and shares the output scaling.
    out[:, :, :, 0:18] *= (params.sensor_gain * TESLA_TO_COUNTS)[None, :, :, None]

    # The sensor block does not: an offset adds straight onto its own channel,
    # and a gain multiplies the field already on it.
    channel = np.eye(MEAS_DIM).reshape(3, 3, MEAS_DIM)
    out[:, :, :, 18:27] = channel[None]
    out[:, :, :, 27:36] = channel[None] * (tesla * TESLA_TO_COUNTS)[:, :, :, None]
    return out.reshape(n, MEAS_DIM, 36)


def residual(problem: Problem, base: np.ndarray, table: FieldTable, x: np.ndarray):
    params, poses, packed = _split(problem, x, base)

    predicted = forward_batch(poses, params, table)
    measurement = ((problem.measurements - predicted) / problem.sigma).ravel()

    prior = ((packed - problem.prior_centre) / problem.prior_sigma)[problem.param_mask]

    free_poses = poses[problem.free_pose]
    axes = np.array([f.axis if f.axis is not None else -1 for f in problem.frames])[
        problem.free_pose
    ]
    off_sigma = np.array([OFF_AXIS_SIGMA_MM] * 3 + [OFF_AXIS_SIGMA_RAD] * 3)
    off_axis = free_poses / off_sigma
    rows = np.arange(len(axes))
    keep = axes >= 0
    off_axis[rows[keep], axes[keep]] = 0.0  # no prior on the commanded axis

    if problem.smooth_pairs.size:
        a, b = problem.smooth_pairs[:, 0], problem.smooth_pairs[:, 1]
        smooth = ((free_poses[b] - free_poses[a]) / problem.smooth_sigma).ravel()
    else:
        smooth = np.zeros(0)

    return np.concatenate([measurement, prior, off_axis.ravel(), smooth])


def jacobian(problem: Problem, base: np.ndarray, table: FieldTable, x: np.ndarray):
    params, poses, _ = _split(problem, x, base)
    n, p, f = problem.n_frames, problem.n_params, problem.n_free

    ev = evaluate_batch(poses, params, table)
    tesla = (ev.field * params.magnet_moment[None, None, :, None]).sum(axis=2)
    d_pose = pose_jacobian_batch(ev, params, poses)  # (n, 9, 6)
    d_param = _param_jacobian(ev, params, tesla)[:, :, problem.param_mask]  # (n, 9, p)

    inv_sigma = (1.0 / problem.sigma)[None, :, None]
    d_pose = -d_pose * inv_sigma
    d_param = -d_param * inv_sigma

    blocks = []

    # Measurement rows: dense in the parameters, block diagonal in the poses.
    meas_param = sparse.csr_matrix(d_param.reshape(n * MEAS_DIM, p))
    slots = problem.pose_slot
    rows, cols, vals = [], [], []
    for k in range(n):
        if slots[k] < 0:
            continue
        r = np.repeat(np.arange(k * MEAS_DIM, (k + 1) * MEAS_DIM), POSE_DIM)
        c = np.tile(
            np.arange(slots[k] * POSE_DIM, (slots[k] + 1) * POSE_DIM), MEAS_DIM
        )
        rows.append(r)
        cols.append(c)
        vals.append(d_pose[k].ravel())
    meas_pose = sparse.coo_matrix(
        (
            np.concatenate(vals) if vals else np.zeros(0),
            (
                np.concatenate(rows) if rows else np.zeros(0, int),
                np.concatenate(cols) if cols else np.zeros(0, int),
            ),
        ),
        shape=(n * MEAS_DIM, f * POSE_DIM),
    )
    blocks.append([meas_param, meas_pose])

    # Prior rows: parameters only.
    prior_diag = sparse.diags(1.0 / problem.prior_sigma[problem.param_mask])
    blocks.append([prior_diag, sparse.csr_matrix((p, f * POSE_DIM))])

    # Off-axis rows: poses only, diagonal, with the commanded axis zeroed.
    off_sigma = np.array([OFF_AXIS_SIGMA_MM] * 3 + [OFF_AXIS_SIGMA_RAD] * 3)
    weights = np.tile(1.0 / off_sigma, (f, 1))
    axes = np.array([fr.axis if fr.axis is not None else -1 for fr in problem.frames])[
        problem.free_pose
    ]
    keep = axes >= 0
    weights[np.arange(f)[keep], axes[keep]] = 0.0
    blocks.append(
        [
            sparse.csr_matrix((f * POSE_DIM, p)),
            sparse.diags(weights.ravel()),
        ]
    )

    # Smoothness rows: +1/-1 on a pair of pose blocks.
    if problem.smooth_pairs.size:
        m = problem.smooth_pairs.shape[0]
        inv = 1.0 / problem.smooth_sigma  # (m, 6)
        r = np.repeat(np.arange(m * POSE_DIM), 2)
        a = problem.smooth_pairs[:, 0][:, None] * POSE_DIM + np.arange(POSE_DIM)[None]
        b = problem.smooth_pairs[:, 1][:, None] * POSE_DIM + np.arange(POSE_DIM)[None]
        c = np.stack([b.ravel(), a.ravel()], axis=1).ravel()
        v = np.stack([inv.ravel(), -inv.ravel()], axis=1).ravel()
        smooth = sparse.coo_matrix(
            (v, (r, c)), shape=(m * POSE_DIM, f * POSE_DIM)
        )
        blocks.append([sparse.csr_matrix((m * POSE_DIM, p)), smooth])

    return sparse.bmat(blocks, format="csr")


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------


def initial_poses(
    problem: Problem, params: CalibParams, table: FieldTable
) -> np.ndarray:
    """Per-frame Gauss-Newton against the starting calibration.

    Cheap and safe: nine measurements for six unknowns, with the Jacobian's
    singular values spanning only a factor of 2.5, so this converges in a
    handful of iterations from the nominal pose.
    """
    poses = np.zeros((problem.n_frames, POSE_DIM))
    previous = np.zeros(POSE_DIM)
    for k, frame in enumerate(problem.frames):
        if frame.is_rest:
            previous = np.zeros(POSE_DIM)
            continue
        poses[k], _ = solve_pose(
            frame.counts, params, table, initial=previous, sigma=problem.sigma
        )
        previous = poses[k]
    return poses


def refit_linear(
    problem: Problem, poses: np.ndarray, params: CalibParams, table: FieldTable
) -> CalibParams:
    """Warm-up: moments and offsets only, with poses and geometry held.

    Linear in both, so this is a single least squares with no local minima,
    and it absorbs the bulk of the amplitude error -- nominal N35 geometry is
    about 25 % too strong, and 40 % too strong for the reversed third magnet --
    before anything nonlinear is asked to converge.
    """
    ev = evaluate_batch(poses, params, table, want_grad=False)
    n = problem.n_frames
    unit = ev.field * TESLA_TO_COUNTS * params.sensor_gain[None, :, None, :]

    design = np.zeros((n * MEAS_DIM, 12))
    design[:, 0:3] = np.transpose(unit, (0, 1, 3, 2)).reshape(n * MEAS_DIM, 3)
    design[:, 3:12] = np.tile(np.eye(MEAS_DIM), (n, 1))

    weight = np.tile(1.0 / problem.sigma, n)[:, None]
    target = problem.measurements.ravel()
    solution, *_ = np.linalg.lstsq(design * weight, target * weight[:, 0], rcond=None)

    out = params.copy()
    out.magnet_moment = solution[0:3]
    out.sensor_offset = solution[3:12].reshape(3, 3)
    return out


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


@dataclass
class SegmentScore:
    segment: str
    n_frames: int
    rms_counts: float
    max_counts: float
    rms_sigma: float


@dataclass
class CalibrationResult:
    params: CalibParams
    poses: np.ndarray
    problem: Problem
    scores: list[SegmentScore] = field(default_factory=list)
    heldout: SegmentScore | None = None
    seconds: float = 0.0
    status: int = 0
    n_eval: int = 0
    converged: bool = True

    def summary(self) -> str:
        lines = [
            f"{'segment':>10}  {'frames':>6}  {'rms':>8}  {'max':>8}  {'rms/sigma':>9}"
        ]
        for s in self.scores:
            lines.append(
                f"{s.segment:>10}  {s.n_frames:6d}  {s.rms_counts:8.2f}  "
                f"{s.max_counts:8.2f}  {s.rms_sigma:9.2f}"
            )
        if self.heldout is not None:
            h = self.heldout
            lines.append(
                f"{h.segment:>10}  {h.n_frames:6d}  {h.rms_counts:8.2f}  "
                f"{h.max_counts:8.2f}  {h.rms_sigma:9.2f}   <- held out"
            )
        return "\n".join(lines)


def score_frames(
    frames: list[Frame],
    poses: np.ndarray,
    params: CalibParams,
    table: FieldTable,
    sigma: np.ndarray,
    label: str,
) -> SegmentScore:
    predicted = forward_batch(poses, params, table)
    measured = np.array([f.counts for f in frames])
    err = measured - predicted
    return SegmentScore(
        segment=label,
        n_frames=len(frames),
        rms_counts=float(np.sqrt((err**2).mean())),
        max_counts=float(np.abs(err).max()),
        rms_sigma=float(np.sqrt(((err / sigma) ** 2).mean())),
    )


def score_heldout(
    session: Session,
    params: CalibParams,
    table: FieldTable,
    per_segment: int = 400,
) -> SegmentScore:
    """Fit poses to the held-out segment and report what is left over.

    The poses have to be solved for -- there is no ground truth anywhere in
    this session -- but the *calibration* has never seen these frames, and six
    pose freedoms cannot hide a wrong nine-channel model.
    """
    frames = [
        f
        for f in session.decimate(per_segment=per_segment)
        if f.segment == HELDOUT_SEGMENT
    ]
    sigma = session.noise_sigma()
    poses = np.zeros((len(frames), POSE_DIM))
    previous = np.zeros(POSE_DIM)
    for k, frame in enumerate(frames):
        poses[k], _ = solve_pose(
            frame.counts, params, table, initial=previous, sigma=sigma
        )
        previous = poses[k]
    return score_frames(frames, poses, params, table, sigma, HELDOUT_SEGMENT)


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------


def fit(
    session: Session,
    table: FieldTable | None = None,
    per_segment: int = 400,
    per_rest_run: int = 150,
    fit_gain: bool = False,
    max_nfev: int = 400,
    verbose: int = 1,
) -> CalibrationResult:
    """Fit one device's calibration.

    ``max_nfev`` is generous on purpose. Stopping early does not merely leave
    the fit a little short: at 60 evaluations this problem parks at a solution
    whose held-out residual is 1.49 counts and whose magnet depths are out by
    more than 2 mm from where they converge to, and nothing in the output says
    so. Check :attr:`CalibrationResult.converged`.
    """
    table = table or build_table()
    start = time.perf_counter()

    params = initial_params(session)
    problem = build_problem(
        session, params, per_segment, per_rest_run, fit_gain=fit_gain
    )
    if verbose:
        print(
            f"{problem.n_frames} frames "
            f"({problem.n_free} free poses, {problem.n_frames - problem.n_free} pinned), "
            f"{problem.n_params} parameters, {problem.n_unknowns} unknowns"
        )
        print(f"magnet polarity from data: {detect_moment_signs(session)}")

    poses = initial_poses(problem, params, table)
    params = refit_linear(problem, poses, params, table)
    if verbose:
        print(f"after warm-up: moments {np.array2string(params.magnet_moment, precision=4)}")

    base = params.pack()
    x0 = np.concatenate([base[problem.param_mask], poses[problem.free_pose].ravel()])

    # Without this the moments, which are ~0.05 in SI, would take steps
    # comparable to sensor offsets of tens of counts and be effectively frozen.
    pose_scale = np.tile([1.0, 1.0, 1.0, 0.05, 0.05, 0.05], problem.n_free)
    x_scale = np.concatenate([CalibParams.scales()[problem.param_mask], pose_scale])

    result = least_squares(
        lambda x: residual(problem, base, table, x),
        x0,
        jac=lambda x: jacobian(problem, base, table, x),
        x_scale=x_scale,
        loss="soft_l1",
        f_scale=ROBUST_F_SCALE,
        tr_solver="lsmr",
        ftol=SOLVER_TOL,
        xtol=SOLVER_TOL,
        gtol=SOLVER_TOL,
        max_nfev=max_nfev,
        verbose=2 if verbose > 1 else 0,
    )

    params, poses, _ = _split(problem, result.x, base)
    seconds = time.perf_counter() - start

    scores = []
    for segment in [REST_SEGMENT] + [s for s in session.segments if s not in (REST_SEGMENT, HELDOUT_SEGMENT)]:
        picks = [i for i, f in enumerate(problem.frames) if f.segment == segment]
        if picks:
            scores.append(
                score_frames(
                    [problem.frames[i] for i in picks],
                    poses[picks],
                    params,
                    table,
                    problem.sigma,
                    segment,
                )
            )

    return CalibrationResult(
        params=params,
        poses=poses,
        problem=problem,
        scores=scores,
        heldout=score_heldout(session, params, table, per_segment),
        seconds=seconds,
        status=result.status,
        n_eval=result.nfev,
        # status 0 is "max_nfev reached", i.e. stopped rather than converged.
        converged=result.status > 0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("session", type=Path, help="recorded CSV")
    parser.add_argument("-o", "--output", type=Path, default=Path("calibration.json"))
    parser.add_argument("--per-segment", type=int, default=400)
    parser.add_argument("--per-rest-run", type=int, default=150)
    parser.add_argument("--fit-gain", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=400)
    parser.add_argument(
        "--force",
        action="store_true",
        help="write the calibration even if the fit did not converge",
    )
    parser.add_argument("-v", "--verbose", action="count", default=1)
    args = parser.parse_args(argv)

    session = load_session(args.session)
    result = fit(
        session,
        per_segment=args.per_segment,
        per_rest_run=args.per_rest_run,
        fit_gain=args.fit_gain,
        max_nfev=args.max_nfev,
        verbose=args.verbose,
    )

    print()
    print(result.summary())
    print(f"\nfitted in {result.seconds:.1f} s, {result.n_eval} evaluations")
    print("magnet moments  ", np.array2string(result.params.magnet_moment, precision=4))
    print("magnet positions", np.array2string(result.params.magnet_pos, precision=3))
    print("sensor offsets  ", np.array2string(result.params.sensor_offset, precision=1))

    if not result.converged and not args.force:
        # Everything downstream trusts this file without re-checking it, and a
        # stopped fit looks plausible: the run that halted at 60 evaluations
        # still scored 1.49 counts on held-out data while placing the magnets
        # 2 mm from where they converge to.
        print(
            f"\nNOT WRITING {args.output}: the fit stopped at max_nfev "
            f"({result.n_eval}) rather than converging.\n"
            "Raise --max-nfev, or pass --force if you know what you are doing."
        )
        return 1

    result.params.save(args.output)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
