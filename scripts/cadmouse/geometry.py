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

#: Angular position of MAG1/2/3 around the ring, degrees in the board xy-plane
#: measured from ``+x``. Shared by both rings, because each magnet sits directly
#: above its own sensor -- that is a fact about the mechanism, so the two rings
#: are given one set of angles rather than two that have to be kept equal by
#: hand. MAG1 faces the user, and the other two are 120 deg either side of it.
RING_ANGLES_DEG = np.array([-90.0, 150.0, 30.0])


def ring(radius: float, z: float, angles_deg: np.ndarray = RING_ANGLES_DEG) -> np.ndarray:
    """``(3, 3)`` points on a horizontal ring, one row per angle."""
    a = np.deg2rad(np.asarray(angles_deg, float))
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(a.size, z)], axis=1)


#: Sensor ring: real pick-and-place coordinates, a 16.51 mm radius at
#: z = -18 mm. Trusted, and the datum that pins the rigid-body gauge freedom.
SENSOR_RING_RADIUS = 16.51
SENSOR_RING_Z = -18.0

#: Sensor positions in the board frame, MAG1/2/3.
SENSOR_POS = ring(SENSOR_RING_RADIUS, SENSOR_RING_Z)

#: Nominal magnet ring radius, 16 mm.
MAGNET_RING_RADIUS = 16.0

#: One 6 x 3 mm disc per position. The lower face sits at z = -12 mm, putting
#: the centre at z = -10.5 mm and leaving a 7.5 mm gap to the sensor plane.
MAGNET_DIAMETER = 6.0
MAGNET_HEIGHT = 3.0
MAGNET_BOTTOM_Z = -12.0

#: Nominal magnet centres in the knob body frame. Design values, and the block
#: the fit is allowed to move -- the ring depth in particular was never
#: measured.
MAGNET_POS = ring(MAGNET_RING_RADIUS, MAGNET_BOTTOM_Z + MAGNET_HEIGHT / 2.0)

#: Vacuum permeability, SI.
MU0 = 4.0e-7 * np.pi

#: Remanence of an N35 disc. Only ever an *initial guess*: the fit replaces it
#: with a per-magnet moment, and the recorded data says the real magnets are
#: some 25 % weaker than this, with magnet 3 weaker still.
BR_N35 = 1.17

#: Nominal moment of one disc at ``BR_N35``, in A*m^2. Unsigned: which way a
#: magnet is actually magnetised is a property of the assembled device, not of
#: the design, so it is measured per device by
#: :func:`~cadmouse.calibrate.detect_moment_signs` rather than written down
#: here.
#:
#: That polarity is carried by the *moment*, which is therefore signed, rather
#: than by flipping the magnetisation axis. Two reasons: it keeps the field
#: linear in the moment, which the calibration's warm-up phase exploits, and it
#: keeps every tilt angle small. It also belongs on the magnet rather than the
#: sensor -- on this device magnet 3 is reversed, and that is not an assumption:
#: correlating the model Jacobian per DOF against the first principal component
#: of each motion segment scores 0.949 for a reversed magnet against 0.834 and
#: 0.796 for the two ways sensor 3 could instead have been mounted upside down.
MOMENT_N35 = BR_N35 / MU0 * (np.pi * (MAGNET_DIAMETER / 2e3) ** 2 * (MAGNET_HEIGHT / 1e3))


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
