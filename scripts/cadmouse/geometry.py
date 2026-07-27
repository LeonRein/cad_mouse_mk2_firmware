"""Board geometry and the pose convention.

Board frame ``B``: origin at the knob's neutral (spring-rest) centre of rotation,
``+x`` right, ``+y`` toward the rear of the device, ``+z`` up.

A pose is a flat 6-vector ``[tx, ty, tz, rx, ry, rz]``: translation in
millimetres, rotation as a rotation *vector* (axis * angle) in radians. The knob
travels a few millimetres and a few degrees, so a rotation vector is nowhere near
its singularity at |r| = pi and gives an unconstrained parameterisation that a
UKF can average over directly -- no quaternion machinery needed.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Sensor ring. Positions are the real pick-and-place coordinates from
# `cad-mouse-mk2/pcbs/fab/sensor_board_CAMOutputs/Assembly/PnP_Sensors Board v80_front.txt`
# (MAG1 (0, -16.51), MAG2 (-14.30, 8.26), MAG3 (+14.30, 8.26)) rather than the
# nominal 16 mm design radius. The board sits 18 mm below the knob centre.
SENSOR_RADIUS_MM = 16.51
SENSOR_Z_MM = -18.0

# Magnet ring at rest. Nominal design values; the 12 mm depth in particular is
# not measured, so `magnet_positions()` takes overrides and the calibration can
# additionally fit a per-magnet position offset.
MAGNET_RADIUS_MM = 16.0
MAGNET_Z_MM = -12.0

# Ring angles, shared by the sensors and the magnets they sit above.
# MAG1 faces the user (-y), MAG2 is rear-left, MAG3 is rear-right.
RING_ANGLES_DEG = np.array([-90.0, 150.0, 30.0])

N_SENSORS = 3
N_MAGNETS = 3
N_CHANNELS = N_SENSORS * 3

#: Channel order of the 9-vector, matching the firmware's ``[i16; 9]``
#: (``src/sensors.rs`` ``read_raw``) and the C++ reference's ``readRaw``.
CHANNEL_NAMES = [f"mag{i + 1}{ax}" for i in range(N_SENSORS) for ax in "xyz"]

DOF_NAMES = ["tx", "ty", "tz", "rx", "ry", "rz"]

# TLI493D-A2B6 with `A2B6Sensitivity::Short` (2x): the driver divides raw counts
# by 7.7 * scale to get millitesla (`../tli493d/src/variant.rs`). The firmware
# streams raw counts, so the conversion lives here and only here.
TLI493D_SENSITIVITY_SCALE = 2.0
COUNTS_PER_MT = 7.7 * TLI493D_SENSITIVITY_SCALE

#: Raw counts are sign-extended 12-bit (`../tli493d/src/register.rs`), so a
#: reading at this magnitude means the ADC railed and the sample is unusable.
ADC_FULL_SCALE_COUNTS = 2047


def expected_gain_counts_per_unit() -> float:
    """Physically predicted sensor gain, in counts per unit of model field.

    **This is a gauge that matters, not a convenience.** Scaling every pose by
    *s* while dividing every gain by *s* leaves the prediction almost unchanged,
    because the field response is close to linear over the knob's small travel.
    Only the model's curvature breaks the tie, and weakly -- fitting with a free
    scale recovers poses that explain the data at the noise floor while being
    wrong by more than the full travel.

    Both factors are known independently of the fit: the TLI493D's sensitivity
    is a datasheet figure (`COUNTS_PER_MT`) and the magnet's moment follows from
    its size and grade (`magnet.physical_moment_am2`). Their product pins the
    scale, and with it the pose's metric meaning.

    Accurate to roughly +/-30 %, dominated by the unknown magnet grade, so
    `calibrate` applies it as a soft prior rather than a hard constraint.
    """
    from .magnet import model_units_to_mt, physical_moment_am2

    return COUNTS_PER_MT * model_units_to_mt(physical_moment_am2())


#: Fractional uncertainty on `expected_gain_counts_per_unit`, used as the width
#: of the prior. Magnet grade dominates; sensor sensitivity tolerance is small.
GAIN_PRIOR_REL_SIGMA = 0.30


def _ring(radius_mm: float, z_mm: float) -> np.ndarray:
    """Three points on a ring at `RING_ANGLES_DEG`, shape (3, 3)."""
    theta = np.deg2rad(RING_ANGLES_DEG)
    return np.stack(
        [radius_mm * np.cos(theta), radius_mm * np.sin(theta), np.full(3, z_mm)],
        axis=-1,
    )


def sensor_positions(
    radius_mm: float = SENSOR_RADIUS_MM, z_mm: float = SENSOR_Z_MM
) -> np.ndarray:
    """Sensor positions in the board frame, shape (3, 3), millimetres."""
    return _ring(radius_mm, z_mm)


def magnet_positions(
    radius_mm: float = MAGNET_RADIUS_MM, z_mm: float = MAGNET_Z_MM
) -> np.ndarray:
    """Magnet positions at the rest pose, board frame, shape (3, 3), millimetres."""
    return _ring(radius_mm, z_mm)


def counts_to_mt(counts: np.ndarray) -> np.ndarray:
    """Convert raw sensor counts to millitesla."""
    return np.asarray(counts, dtype=float) / COUNTS_PER_MT


def is_saturated(counts: np.ndarray) -> np.ndarray:
    """Per-sample mask of frames where any channel railed the 12-bit ADC."""
    return np.any(np.abs(np.asarray(counts)) >= ADC_FULL_SCALE_COUNTS, axis=-1)


def pose_to_rt(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a pose (or batch of poses) into rotation matrices and translations.

    Accepts shape (6,) or (N, 6); returns ``(R, t)`` with shapes
    ``((3, 3), (3,))`` or ``((N, 3, 3), (N, 3))`` respectively.
    """
    pose = np.asarray(pose, dtype=float)
    single = pose.ndim == 1
    flat = np.atleast_2d(pose)
    rot = Rotation.from_rotvec(flat[:, 3:6]).as_matrix()
    trans = flat[:, 0:3]
    if single:
        return rot[0], trans[0]
    return rot, trans


def transform_knob(
    pose: np.ndarray, points: np.ndarray, moments: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Move knob-fixed magnets to a pose.

    Positions rotate about the knob centre *and* translate; moments only rotate.
    Both ``points`` and ``moments`` are (M, 3) in the rest configuration.

    Returns ``(positions, moments)`` shaped (M, 3) for a single pose or
    (N, M, 3) for a batch of N poses.
    """
    rot, trans = pose_to_rt(pose)
    points = np.asarray(points, dtype=float)
    moments = np.asarray(moments, dtype=float)
    if rot.ndim == 2:
        return points @ rot.T + trans, moments @ rot.T
    # (N, 3, 3) x (M, 3) -> (N, M, 3)
    return (
        np.einsum("nij,mj->nmi", rot, points) + trans[:, None, :],
        np.einsum("nij,mj->nmi", rot, moments),
    )
