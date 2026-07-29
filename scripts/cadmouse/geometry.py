"""Fixed geometry of the knob and the sensor ring.

Board frame: origin at the knob's neutral centre of rotation, ``+x`` right,
``+y`` toward the rear, ``+z`` up. Millimetres throughout. The knob body frame
coincides with the board frame at pose zero, so a pose is just the rigid
transform that carries body coordinates into board coordinates.

Everything in here is either *measured* or *fixed by construction*, and the
distinction matters for calibration:

* Sensor positions are pick-and-place coordinates, so they are trusted and act
  as the datum that pins the rigid-body gauge freedom (translating every magnet
  is otherwise indistinguishable from translating every sensor the other way).
* Magnet positions are design values -- the 12 mm ring depth in particular was
  never measured -- so they are the block the fit is allowed to move.
* Magnet diameter and length are drawing values held fixed, because at this
  standoff they are not separable from the magnet's moment.

See ``scripts/README.md`` for where these numbers come from.
"""

from __future__ import annotations

import numpy as np

#: Channel order of the 9-vector, matching the firmware's ``[i16; 9]``
#: (``src/sensors.rs`` ``read_raw``) and hence the CSV column order.
CHANNEL_NAMES = [f"mag{i + 1}{ax}" for i in range(3) for ax in "xyz"]

#: Counts per millitesla at ``A2B6Sensitivity::Short`` (2x): the TLI493D-A2B6
#: gives 7.7 LSB/mT at full range, doubled by the X2 bit
#: (``../tli493d/src/variant.rs``). The X2 bit scales Bx, By and Bz alike
#: (user manual Table 3), so this is a scalar, not a per-axis vector.
COUNTS_PER_MT = 7.7 * 2.0

#: Raw counts are sign-extended 12-bit, so a reading here means the ADC railed.
ADC_FULL_SCALE_COUNTS = 2047

#: Sensor positions in the board frame, MAG1/2/3. Real pick-and-place
#: coordinates: a 16.51 mm ring at z = -18 mm, ring angles -90 deg, 150 deg,
#: 30 deg. MAG1 faces the user.
SENSOR_POS = np.array(
    [
        [0.00, -16.51, -18.0],
        [-14.30, 8.26, -18.0],
        [14.30, 8.26, -18.0],
    ]
)

#: Nominal magnet ring: radius 16 mm, same angular positions as the sensors so
#: each magnet sits directly above one sensor.
MAGNET_RING_RADIUS = 16.0
MAGNET_RING_ANGLES_DEG = np.array([-90.0, 150.0, 30.0])

#: Two 6 x 3 mm discs stacked, so one 6 mm diameter by 6 mm tall cylinder. The
#: lower face sits at z = -12 mm, putting the centre at z = -9 mm and leaving a
#: 6 mm gap to the sensor plane.
MAGNET_DIAMETER = 6.0
MAGNET_HEIGHT = 6.0
MAGNET_BOTTOM_Z = -12.0

#: Nominal magnet centres in the knob body frame.
MAGNET_POS = np.stack(
    [
        MAGNET_RING_RADIUS * np.cos(np.deg2rad(MAGNET_RING_ANGLES_DEG)),
        MAGNET_RING_RADIUS * np.sin(np.deg2rad(MAGNET_RING_ANGLES_DEG)),
        np.full(3, MAGNET_BOTTOM_Z + MAGNET_HEIGHT / 2.0),
    ],
    axis=1,
)

#: Vacuum permeability, SI.
MU0 = 4.0e-7 * np.pi

#: Remanence of an N35 disc. Only ever an *initial guess*: the fit replaces it
#: with a per-magnet moment, and the recorded data says the real magnets are
#: some 25 % weaker than this, with magnet 3 weaker still.
BR_N35 = 1.17

#: Nominal moment of one stacked pair at ``BR_N35``, in A*m^2.
MOMENT_N35 = BR_N35 / MU0 * (np.pi * (MAGNET_DIAMETER / 2e3) ** 2 * (MAGNET_HEIGHT / 1e3))

#: Magnet 3 is physically reversed. This is not an assumption: correlating the
#: model Jacobian per DOF against the first principal component of each motion
#: segment scores 0.949 for a reversed magnet against 0.834 and 0.796 for the
#: two ways sensor 3 could instead have been mounted upside down. The sign is
#: carried by the magnet's *moment*, which is therefore signed, rather than by
#: flipping its axis -- that keeps the field model linear in the moment, which
#: the calibration's warm-up phase exploits.
MOMENT_SIGN = np.array([1.0, 1.0, -1.0])


def rotation_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    """Rodrigues' formula, with the small-angle limit handled.

    Poses in this application never exceed a few degrees, so the series would
    do -- but the calibration's line search can step well outside that range,
    and a rotation that silently stops being orthonormal is a miserable thing
    to debug.
    """
    rotvec = np.asarray(rotvec, dtype=float)
    theta = float(np.linalg.norm(rotvec))
    if theta < 1e-12:
        return np.eye(3)
    k = rotvec / theta
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1.0 - np.cos(theta)) * (kx @ kx)


def skew(v: np.ndarray) -> np.ndarray:
    """Matrix ``[v]_x`` with ``[v]_x u == cross(v, u)``."""
    v = np.asarray(v, dtype=float)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def right_jacobian_so3(rotvec: np.ndarray) -> np.ndarray:
    """Right Jacobian of SO(3), relating the two rotation conventions.

    The measurement Jacobian is naturally written for a *local* perturbation,
    ``R <- R exp(delta)``, which avoids differentiating Rodrigues' formula. A
    general-purpose optimiser instead varies the rotation vector itself. The
    two are related by ``exp((theta + d)^) ~= exp(theta^) exp((Jr(theta) d)^)``,
    so a Jacobian in the local convention becomes one in the vector convention
    by right-multiplying the rotation block by this matrix.

    At the few degrees this mechanism reaches, ``Jr`` differs from the identity
    by under a percent -- but it is the difference between a fit that converges
    quadratically and one that limps, and it costs almost nothing.
    """
    rotvec = np.asarray(rotvec, dtype=float)
    theta = float(np.linalg.norm(rotvec))
    kx = skew(rotvec)
    if theta < 1e-8:
        # Series to second order; exact enough well past where it is used.
        return np.eye(3) - 0.5 * kx + (kx @ kx) / 6.0
    return (
        np.eye(3)
        - ((1.0 - np.cos(theta)) / theta**2) * kx
        + ((theta - np.sin(theta)) / theta**3) * (kx @ kx)
    )
