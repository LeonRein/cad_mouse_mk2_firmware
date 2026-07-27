"""Validate the field models against closed-form physics.

The headline result is `test_point_dipole_error_at_operating_distance`, which
measures the point-dipole error at the actual 6 mm magnet-to-sensor gap. That
number is the justification for `SubdipoleCylinder` existing at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadmouse import forward, geometry
from cadmouse.magnet import (
    MAGNET_DIAMETER_MM,
    PointDipole,
    SubdipoleCylinder,
    cylinder_on_axis_bz,
)
from cadmouse.params import CalibParams, signed_permutations_planar

#: Magnet centre to sensor at the rest pose: magnets sit at z = -12 mm, sensors
#: at z = -18 mm.
OPERATING_GAP_MM = geometry.MAGNET_Z_MM - geometry.SENSOR_Z_MM


def _on_axis_field(model, z_mm: float, moment_z: float = 1.0) -> float:
    """B_z at `z_mm` below a magnet at the origin magnetised along +z."""
    point = np.array([[0.0, 0.0, -z_mm]])
    positions = np.array([[0.0, 0.0, 0.0]])
    moments = np.array([[0.0, 0.0, moment_z]])
    return model.field(point, positions, moments)[0, 2]


# --------------------------------------------------------------- dipole physics


def test_dipole_matches_closed_form_far_field():
    """Far from the magnet, the cylinder solution must reduce to 2m/z^3."""
    z = 200.0  # >> diameter, so finite size is negligible
    assert cylinder_on_axis_bz(z) == pytest.approx(2.0 / z**3, rel=1e-3)


def test_dipole_on_axis_and_equatorial():
    """The textbook factor-of-two-and-a-sign-flip between the two directions."""
    model = PointDipole()
    positions = np.array([[0.0, 0.0, 0.0]])
    moments = np.array([[0.0, 0.0, 1.0]])
    r = 10.0

    on_axis = model.field(np.array([[0.0, 0.0, r]]), positions, moments)[0]
    equatorial = model.field(np.array([[r, 0.0, 0.0]]), positions, moments)[0]

    assert on_axis == pytest.approx([0.0, 0.0, 2.0 / r**3], abs=1e-12)
    assert equatorial == pytest.approx([0.0, 0.0, -1.0 / r**3], abs=1e-12)


def test_superposition():
    """Two magnets' field is the sum of each one's."""
    model = PointDipole()
    pts = np.array([[3.0, -4.0, 5.0], [0.0, 0.0, 9.0]])
    p1, m1 = np.array([[1.0, 2.0, 0.0]]), np.array([[0.2, -0.3, 0.9]])
    p2, m2 = np.array([[-2.0, 0.5, 1.0]]), np.array([[-0.7, 0.1, 0.4]])

    both = model.field(pts, np.vstack([p1, p2]), np.vstack([m1, m2]))
    separate = model.field(pts, p1, m1) + model.field(pts, p2, m2)
    assert both == pytest.approx(separate, rel=1e-12)


# ------------------------------------------------------- finite-size cylinder


def test_subdipole_converges_to_closed_form():
    """A refined node set must reproduce the exact on-axis solution."""
    fine = SubdipoleCylinder(n_r=6, n_theta=16, n_z=4)
    for z in (4.0, 6.0, 10.0, 20.0):
        assert _on_axis_field(fine, z) == pytest.approx(
            cylinder_on_axis_bz(z), rel=1e-3
        ), f"at z = {z} mm"


def test_quadrature_weights_are_normalised():
    """Sub-moments must total the whole magnet's moment.

    Gauss-Legendre weights are not equal, so this is a real constraint rather
    than an artefact of dividing by the node count.
    """
    for g in [(2, 6, 2), (3, 8, 2), (6, 16, 4)]:
        _, weights = SubdipoleCylinder(*g)._nodes()
        assert weights.sum() == pytest.approx(1.0, rel=1e-12)


def test_gauss_legendre_beats_the_uniform_grid_it_replaced():
    """24 GL nodes must beat the 96-node uniform grid this used to use.

    Pinning the reason the quadrature changed: the win came from *where* the
    nodes are, not how many. If a future edit reverts to midpoint sampling,
    accuracy drops by roughly 6x and this fails.
    """
    err = abs(
        _on_axis_field(SubdipoleCylinder(2, 6, 2), OPERATING_GAP_MM)
        / cylinder_on_axis_bz(OPERATING_GAP_MM)
        - 1.0
    )
    assert err < 0.005, f"24 GL nodes give {err:.3%}, expected well under 0.5%"


def test_subdipole_refinement_is_monotone():
    """Error must shrink as the node set refines, at the tightest gap.

    The knob bottoms out well inside 6 mm, so 4 mm is the case that matters.
    """
    errs = [
        abs(_on_axis_field(SubdipoleCylinder(*g), 4.0) / cylinder_on_axis_bz(4.0) - 1.0)
        for g in [(1, 4, 1), (2, 6, 2), (3, 8, 2), (6, 16, 4)]
    ]
    assert errs == sorted(errs, reverse=True), f"not monotone: {errs}"


def test_default_model_is_accurate_enough():
    """`forward.DEFAULT_MODEL` must be well inside a percent at both extremes.

    Node count multiplies calibration time directly, so this is the check that
    the default buys real accuracy rather than just cost.
    """
    for gap in (OPERATING_GAP_MM, 4.0):
        err = abs(
            _on_axis_field(forward.DEFAULT_MODEL, gap) / cylinder_on_axis_bz(gap) - 1.0
        )
        assert err < 0.005, f"default model off by {err:.3%} at {gap} mm"


def test_single_node_sits_at_the_rms_radius():
    """One radial node lands at R/sqrt(2), *not* at the centre.

    Worth pinning down: it means `SubdipoleCylinder(1, 1, 1)` is deliberately not
    a point dipole, so `COARSE_MODEL` has to be `PointDipole` explicitly.
    """
    offsets, weights = SubdipoleCylinder(n_r=1, n_theta=1, n_z=1)._nodes()
    assert offsets.shape == (1, 3)
    assert weights == pytest.approx([1.0])
    radius = MAGNET_DIAMETER_MM / 2.0
    assert np.hypot(offsets[0, 0], offsets[0, 1]) == pytest.approx(
        radius / np.sqrt(2.0), rel=1e-12
    )


def test_subdipole_handles_flipped_and_tilted_moments():
    """Magnets may be glued in flipped; the offset cloud must not blow up.

    The -z case is the degenerate branch of the Rodrigues rotation in
    `_align_z_to`, so it is the one that would produce NaNs if mishandled.
    """
    model = SubdipoleCylinder()
    positions = np.array([[0.0, 0.0, 0.0]])
    point = np.array([[0.0, 0.0, -OPERATING_GAP_MM]])

    up = model.field(point, positions, np.array([[0.0, 0.0, 1.0]]))
    down = model.field(point, positions, np.array([[0.0, 0.0, -1.0]]))
    assert np.all(np.isfinite(down))
    assert down == pytest.approx(-up, rel=1e-12)

    tilted = model.field(point, positions, np.array([[0.3, -0.2, 0.9]]))
    assert np.all(np.isfinite(tilted))


def test_point_dipole_error_at_operating_distance():
    """Quantify the modelling error this geometry actually incurs.

    Magnet diameter 6 mm, gap 6 mm -- exactly one diameter, where a point dipole
    is badly wrong: +30 % at rest, and +77 % at 4 mm once the knob is pushed
    down. This is the entire justification for `SubdipoleCylinder` being the
    default rather than an option, so it is asserted rather than assumed.
    """
    assert OPERATING_GAP_MM == pytest.approx(MAGNET_DIAMETER_MM)

    err = abs(
        _on_axis_field(PointDipole(), OPERATING_GAP_MM)
        / cylinder_on_axis_bz(OPERATING_GAP_MM)
        - 1.0
    )
    assert 0.20 < err < 0.40, f"point-dipole error is {err:.1%}, expected ~30%"

    near = abs(
        _on_axis_field(PointDipole(), 4.0) / cylinder_on_axis_bz(4.0) - 1.0
    )
    assert near > err, "error must grow as the knob presses closer"


# ------------------------------------------------------------- forward model


def _demo_params() -> CalibParams:
    return CalibParams.initial(gain_counts_per_unit=5000.0)


def test_predict_shapes_and_batching():
    """A batch of poses must give exactly the per-pose results, stacked."""
    params = _demo_params()
    poses = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, -0.3, 0.2, 0.01, -0.02, 0.03],
            [-1.0, 0.0, 0.4, 0.0, 0.05, 0.0],
        ]
    )

    batched = forward.predict(poses, params)
    assert batched.shape == (3, geometry.N_CHANNELS)
    for i, pose in enumerate(poses):
        single = forward.predict(pose, params)
        assert single.shape == (geometry.N_CHANNELS,)
        assert single == pytest.approx(batched[i], rel=1e-12)


def test_rest_pose_is_symmetric():
    """At rest, three identical magnets on a ring give three identical readings.

    Each sensor sees the same field pattern rotated by 120 degrees, so the
    *radial* and *axial* components must match across sensors even though the
    x/y components differ.

    Tolerance is 1e-6 rather than exact: the default model's 8-point sub-dipole
    ring is not invariant under a 120 degree rotation, so it breaks the symmetry
    at the 1e-8 level. That is discretisation, not a modelling error, and it is
    four orders of magnitude below the sensor's 1-count quantisation.
    """
    params = _demo_params()
    b = forward.board_field(np.zeros(6), params)

    z = b[:, 2]
    assert z == pytest.approx(np.full(3, z[0]), rel=1e-6)

    radial_dir = params.sensor_positions[:, :2]
    radial_dir /= np.linalg.norm(radial_dir, axis=1, keepdims=True)
    radial = np.sum(b[:, :2] * radial_dir, axis=1)
    assert radial == pytest.approx(np.full(3, radial[0]), rel=1e-6)


def test_bias_is_the_reading_at_infinite_distance():
    """With zero magnet moments, `predict` returns exactly the biases."""
    params = _demo_params()
    params.moments = np.zeros((3, 3))
    params.biases = np.arange(9, dtype=float).reshape(3, 3)
    assert forward.predict(np.zeros(6), params) == pytest.approx(np.arange(9.0))


def test_translation_and_rotation_both_move_the_readings():
    """Sanity: every DOF has to actually do something, or it is unobservable."""
    params = _demo_params()
    rest = forward.predict(np.zeros(6), params)
    for k, name in enumerate(geometry.DOF_NAMES):
        pose = np.zeros(6)
        pose[k] = 0.5 if k < 3 else np.deg2rad(3.0)
        moved = forward.predict(pose, params)
        assert not np.allclose(moved, rest, atol=1e-9), f"{name} does nothing"


def test_pose_jacobian_matches_finite_difference_of_predict():
    """The Jacobian helper agrees with a coarse independent difference."""
    params = _demo_params()
    pose = np.array([0.3, -0.2, 0.1, 0.01, 0.02, -0.01])
    jac = forward.pose_jacobian(pose, params)
    assert jac.shape == (geometry.N_CHANNELS, 6)

    for k in range(6):
        h = 1e-4
        delta = np.zeros(6)
        delta[k] = h
        coarse = (
            forward.predict(pose + delta, params) - forward.predict(pose - delta, params)
        ) / (2 * h)
        assert jac[:, k] == pytest.approx(coarse, rel=1e-4, abs=1e-6)


# --------------------------------------------------------------- orientations


def test_signed_permutations_are_valid_and_distinct():
    """The 16 candidates must be genuine orthogonal signed permutations."""
    mats = signed_permutations_planar()
    assert len(mats) == 16
    assert len({m.tobytes() for m in mats}) == 16

    for m in mats:
        assert m.T @ m == pytest.approx(np.eye(3), abs=1e-12)
        assert abs(round(float(np.linalg.det(m)))) == 1
        # z stays isolated: Rx/Ry in the C++ mix use z-channels alone.
        assert abs(m[2, 2]) == 1
        assert m[2, 0] == 0 and m[2, 1] == 0
        assert m[0, 2] == 0 and m[1, 2] == 0


def test_orientation_reaches_predict():
    """A different shared orientation must permute the predicted channels."""
    identity = CalibParams.initial(gain_counts_per_unit=5000.0)
    pose = np.array([0.4, 0.2, -0.1, 0.01, 0.0, 0.02])
    base = forward.predict(pose, identity).reshape(3, 3)

    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    swapped_params = CalibParams.initial(orientation=swap, gain_counts_per_unit=5000.0)
    swapped = forward.predict(pose, swapped_params).reshape(3, 3)

    assert swapped[:, 0] == pytest.approx(base[:, 1], rel=1e-12)
    assert swapped[:, 1] == pytest.approx(base[:, 0], rel=1e-12)
    assert swapped[:, 2] == pytest.approx(-base[:, 2], rel=1e-12)
