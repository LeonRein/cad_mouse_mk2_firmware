"""Synthetic calibration sessions from known parameters.

This exists so the calibration can be validated *before* any hardware is
involved. Without it, a bad fit is ambiguous between "the solver is broken" and
"the hardware is not what we assumed"; with it, those are separable.

The simulated session deliberately includes the things that make real data hard:
a scrambled sensor orientation, flipped and mismatched magnets, per-axis gain and
cross-axis error, large sensor biases, mechanical cross-coupling between the
"single-axis" segments, and ADC quantisation.
"""

from __future__ import annotations

import numpy as np

from . import forward, geometry
from .calibrate import CalibrationData, HELDOUT_SEGMENT, REST_SEGMENT
from .magnet import MagnetModel
from .params import CalibParams, signed_permutations_planar

#: Full-scale travel of the knob per DOF: millimetres then radians.
#:
#: Note how small the tz and tilt values are. Pressing the knob *down* closes
#: the 6 mm magnet-to-sensor gap, and the field rises steeply: at the upper end
#: of the gain prior the sensor rails past about 1.2 mm of tz or 7.4 degrees of
#: rx. Tilting counts too -- 0.13 rad at a 16 mm ring radius lifts one magnet
#: nearly 2 mm. These values stay inside that envelope so a simulated session is
#: usable; larger ones produce data the calibration correctly refuses as railed.
#:
#: That envelope is a real property of the build, not of this simulator. An N35
#: 6x3 mm disc reaches ~141 mT at a 4 mm gap, past the +/-133 mT the
#: TLI493D-A2B6 can represent at the `A2B6Sensitivity::Short` (2x) setting the
#: firmware currently uses (`src/sensors.rs`). Whether the real device clips
#: depends on the actual magnet grade -- the BOM does not state one -- and the
#: true ring heights. `record.py --check` reports peak counts and settles it in
#: seconds; the remedy is `A2B6Sensitivity::Full`, doubling the range to
#: +/-266 mT for half the resolution.
TRAVEL = np.array([2.5, 2.5, 1.0, 0.09, 0.09, 0.15])

#: How much a "pure" axis sweep bleeds into the other five. A real spring gimbal
#: cross-couples; a calibration that assumes otherwise bakes the error in.
CROSS_COUPLING = 0.12

#: Counts of white noise per channel. The TLI493D's own noise at 2x sensitivity
#: is a couple of LSB; this is deliberately pessimistic.
NOISE_COUNTS = 3.0


def random_truth(
    rng: np.random.Generator,
    gain_counts_per_unit: float | None = None,
    gain_error: float = 0.08,
    cross_axis: float = 0.04,
    bias_counts: float = 200.0,
    moment_spread: float = 0.25,
    moment_tilt_rad: float = 0.10,
) -> CalibParams:
    """A plausible 'true' calibration, with every defect the fit must absorb.

    The gain defaults to the physically predicted magnitude, offset by a random
    amount within the prior's width. Simulating a device whose true gain sat
    exactly at the prior's centre would flatter the fit -- the point is to check
    that the prior *locates* the scale, not that it happens to be handed the
    answer.
    """
    if gain_counts_per_unit is None:
        # Clipped at one sigma. An unclipped tail draw produces a magnet strong
        # enough to rail the ADC at the *rest* pose, where backing off the travel
        # cannot help -- a device that simply does not work at 2x sensitivity.
        # That is a real possibility on hardware but useless to simulate, since
        # there is no session to calibrate from.
        offset = np.clip(rng.normal(0.0, geometry.GAIN_PRIOR_REL_SIGMA),
                         -geometry.GAIN_PRIOR_REL_SIGMA, geometry.GAIN_PRIOR_REL_SIGMA)
        gain_counts_per_unit = geometry.expected_gain_counts_per_unit() * float(
            np.exp(offset)
        )
    orientation = signed_permutations_planar()[rng.integers(16)]

    gains = np.empty((geometry.N_SENSORS, 3, 3))
    for i in range(geometry.N_SENSORS):
        scale = np.diag(1.0 + rng.normal(0.0, gain_error, 3))
        skew = np.eye(3) + rng.normal(0.0, cross_axis, (3, 3)) * (1 - np.eye(3))
        gains[i] = gain_counts_per_unit * (skew @ scale @ orientation)

    biases = rng.normal(0.0, bias_counts, (geometry.N_SENSORS, 3))

    # Magnets: axial nominally, but flipped at random, with unequal strength and
    # a small glue-in tilt.
    moments = np.zeros((geometry.N_MAGNETS, 3))
    moments[:, 2] = rng.choice([-1.0, 1.0], geometry.N_MAGNETS)
    moments *= (1.0 + rng.normal(0.0, moment_spread, (geometry.N_MAGNETS, 1)))
    moments[:, :2] = rng.normal(0.0, moment_tilt_rad, (geometry.N_MAGNETS, 2))

    # Record the orientation that was used, so `gain_scales` and
    # `orientation_deviation` mean what they say when checked against truth.
    return CalibParams(
        gains=gains, biases=biases, moments=moments, orientation=orientation
    )


def session_poses(
    rng: np.random.Generator,
    n_rest: int = 120,
    n_sweep: int = 260,
    n_free: int = 400,
    cross_coupling: float = CROSS_COUPLING,
    travel: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Poses and segment labels for one guided capture, matching `record.py`.

    Sequence: rest, then a full-travel sweep of each DOF with a rest between,
    then a held-out block of arbitrary 6-DOF motion.
    """
    travel = TRAVEL if travel is None else np.asarray(travel, dtype=float)
    poses: list[np.ndarray] = []
    labels: list[str] = []

    def add(block: np.ndarray, label: str) -> None:
        poses.append(block)
        labels.extend([label] * len(block))

    def rest_block() -> np.ndarray:
        # Not exactly zero: the spring rest position wobbles a little.
        return rng.normal(0.0, 0.01, (n_rest, 6)) * travel

    add(rest_block(), REST_SEGMENT)
    for k, name in enumerate(geometry.DOF_NAMES):
        block = np.zeros((n_sweep, 6))
        ramp = np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, n_sweep))
        block[:, k] = travel[k] * ramp
        # Cross-coupling: the other DOFs follow the driven one, plus a little
        # independent slop.
        for j in range(6):
            if j == k:
                continue
            block[:, j] = travel[j] * (
                cross_coupling * rng.normal(0.0, 0.5) * ramp
                + rng.normal(0.0, 0.01, n_sweep)
            )
        add(block, name)
        add(rest_block(), REST_SEGMENT)

    if n_free:
        # Smooth random walk through the workspace, low-passed so it looks like
        # a hand rather than white noise.
        walk = rng.normal(0.0, 1.0, (n_free, 6))
        for i in range(1, n_free):
            walk[i] = 0.92 * walk[i - 1] + 0.08 * walk[i]
        walk /= np.abs(walk).max(axis=0, keepdims=True) + 1e-9
        add(walk * travel * 0.8, HELDOUT_SEGMENT)

    return np.concatenate(poses), np.array(labels, dtype=object)


def simulate(
    params: CalibParams,
    poses: np.ndarray,
    segments: np.ndarray,
    rng: np.random.Generator,
    noise_counts: float = NOISE_COUNTS,
    model: MagnetModel | None = None,
    quantize: bool = True,
) -> CalibrationData:
    """Render poses to sensor counts through the forward model."""
    model = model or forward.DEFAULT_MODEL
    counts = forward.predict(poses, params, model)
    counts = counts + rng.normal(0.0, noise_counts, counts.shape)
    if quantize:
        counts = np.rint(counts)
        counts = np.clip(
            counts, -geometry.ADC_FULL_SCALE_COUNTS, geometry.ADC_FULL_SCALE_COUNTS
        )
    return CalibrationData(counts, segments)


def make_session(
    seed: int = 0,
    noise_counts: float = NOISE_COUNTS,
    model: MagnetModel | None = None,
    max_backoff: int = 12,
    **truth_kwargs,
) -> tuple[CalibrationData, CalibParams, np.ndarray]:
    """One complete synthetic session. Returns ``(data, true_params, true_poses)``.

    If the generated motion rails the ADC, the travel is scaled down and the
    session regenerated. This is not a fudge -- it is the same choice the real
    build forces, and it keeps the simulator honest about it: a strong magnet
    leaves usable travel of barely a millimetre in z (see `TRAVEL`). Simultaneous
    multi-axis motion in the ``free`` block is usually what tips it over, since
    excursions stack.
    """
    rng = np.random.default_rng(seed)
    params = random_truth(rng, **truth_kwargs)

    scale = 1.0
    for _ in range(max_backoff):
        poses, segments = session_poses(rng, travel=TRAVEL * scale)
        data = simulate(
            params, poses, segments, rng, noise_counts=noise_counts, model=model
        )
        if not geometry.is_saturated(data.counts).any():
            return data, params, poses
        scale *= 0.85

    raise RuntimeError(
        "could not generate an unsaturated session: even at "
        f"{scale:.2f}x travel the ADC rails. The drawn magnet is too strong for "
        "the sensor's range -- exactly the condition that would force "
        "A2B6Sensitivity::Full on real hardware."
    )


def pose_errors(estimated: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """RMS pose error per DOF, in millimetres and degrees."""
    err = np.asarray(estimated) - np.asarray(truth)
    rms = np.sqrt((err**2).mean(axis=0))
    out = {name: float(rms[i]) for i, name in enumerate(geometry.DOF_NAMES[:3])}
    out.update(
        {name: float(np.rad2deg(rms[i + 3])) for i, name in enumerate(geometry.DOF_NAMES[3:])}
    )
    return out
