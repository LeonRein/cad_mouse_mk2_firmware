"""Gates for the measurement function.

Two kinds of check live here. The mechanical ones -- Jacobian against finite
differences, round-tripping a pose through ``solve_pose`` -- say the code does
what it claims. The others compare against the recorded session, and those are
the ones that would catch a sign error or a mislabelled channel, because
nominal geometry is already close enough to the truth that a wrong model looks
obviously wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadmouse import forward, forward_and_jac, solve_pose
from cadmouse.dataset import SEGMENT_AXIS
from cadmouse.model import MEAS_DIM, POSE_DIM, perturb_pose


def test_jacobian_matches_finite_differences(table, nominal, envelope_poses):
    """Perturbations must go through ``perturb_pose`` to match the convention."""
    step = 1e-6
    for pose in envelope_poses[:25]:
        _, jac = forward_and_jac(pose, nominal, table)
        for k in range(POSE_DIM):
            bump = np.zeros(POSE_DIM)
            bump[k] = step
            hi = forward(perturb_pose(pose, bump), nominal, table)
            lo = forward(perturb_pose(pose, -bump), nominal, table)
            numeric = (hi - lo) / (2 * step)
            assert np.allclose(numeric, jac[:, k], rtol=2e-4, atol=1e-4)


def test_forward_and_jac_agrees_with_forward(table, nominal, envelope_poses):
    for pose in envelope_poses[:20]:
        assert np.allclose(forward(pose, nominal, table), forward_and_jac(pose, nominal, table)[0])


def test_offsets_and_gains_enter_affinely(table, nominal):
    """The sensor block must be a pure output map, untouched by pose."""
    params = nominal.copy()
    params.sensor_offset = np.arange(9, dtype=float).reshape(3, 3)
    pose = np.array([0.3, -0.2, 0.15, 0.02, -0.01, 0.03])
    base = forward(pose, nominal, table)
    shifted = forward(pose, params, table)
    assert np.allclose(shifted - base, params.sensor_offset.ravel())

    params = nominal.copy()
    params.sensor_gain = np.full((3, 3), 2.0)
    assert np.allclose(forward(pose, params, table), 2.0 * base)


def test_cross_talk_is_not_negligible(table, nominal):
    """Dropping the two far magnets would inject ten times the noise floor.

    This is the assumption behind summing over all three magnets, and it is
    cheap enough to keep honest.
    """
    from cadmouse.magnet import field_and_grad
    from cadmouse.model import TESLA_TO_COUNTS, _geometry

    _, _, axes, delta = _geometry(np.zeros(POSE_DIM), nominal)
    b, _, _ = field_and_grad(
        table, delta, axes[None, :, :], nominal.magnet_moment[None, :], want_grad=False
    )
    for sensor in range(3):
        far = np.delete(b[sensor], sensor, axis=0).sum(axis=0)
        assert np.linalg.norm(far) * TESLA_TO_COUNTS > 5.0


def test_pose_round_trips_when_warm_started(table, nominal, envelope_poses):
    """The production path: every solve starts from a neighbouring pose.

    Both the calibration's pose initialisation and the filter's startup warm
    start, so this is the case that has to be watertight.
    """
    for pose in envelope_poses[:30]:
        measurement = forward(pose, nominal, table)
        nearby = pose + np.concatenate(
            [np.full(3, 0.15), np.full(3, np.deg2rad(1.0))]
        )
        recovered, converged = solve_pose(measurement, nominal, table, initial=nearby)
        assert converged
        assert np.allclose(forward(recovered, nominal, table), measurement, atol=1e-4)


def test_pose_mostly_round_trips_from_a_cold_start(table, nominal, envelope_poses):
    """Cold-started from pose zero, which only the very first frame ever is.

    Damping is what makes this work at all: undamped Gauss-Newton lands in the
    wrong basin for about a third of this envelope and more iterations do not
    help. Even damped it is not perfect, because the fixture samples all six
    degrees of freedom independently and its corners put every axis at maximum
    at once -- a far larger excursion than the mechanism produces, where the
    measured peaks are on different axes at different moments.

    The consequence for the device is worth knowing rather than hiding: if the
    knob is held well off centre at power-up, the estimator can latch onto the
    wrong pose. Starting from rest, which is the normal case, it cannot.
    """
    failures = 0
    for pose in envelope_poses[:80]:
        measurement = forward(pose, nominal, table)
        recovered, _ = solve_pose(measurement, nominal, table)
        if not np.allclose(forward(recovered, nominal, table), measurement, atol=1e-3):
            failures += 1
    assert failures <= 4, f"{failures}/80 cold starts landed in the wrong basin"


def test_pose_is_observable(table, nominal):
    """All six degrees of freedom, from one frame, with room to spare.

    The singular values of the Jacobian scaled to a (1 mm, 5 deg) envelope span
    a factor of about 2.5. A poorly observed direction would show up here long
    before it showed up as a filter that will not settle.
    """
    _, jac = forward_and_jac(np.zeros(POSE_DIM), nominal, table)
    envelope = np.array([1.0, 1.0, 1.0, np.deg2rad(5), np.deg2rad(5), np.deg2rad(5)])
    singular = np.linalg.svd(jac * envelope, compute_uv=False)
    assert singular[-1] > 100.0, "weakest direction should still move ~100 counts"
    assert singular[0] / singular[-1] < 5.0


# ---------------------------------------------------------------- vs. data


def test_nominal_model_has_the_right_shape_but_the_wrong_size(table, nominal, session):
    """Nominal geometry points the right way and is badly scaled.

    Both halves matter. The direction being right is what makes the fit
    converge from this starting point; the magnitude being wrong by a quarter,
    and by 40 % on the reversed third magnet, is what makes the fit necessary.
    """
    measured = session.rest_mean()
    predicted = forward(np.zeros(POSE_DIM), nominal, table)

    cosine = float(measured @ predicted / (np.linalg.norm(measured) * np.linalg.norm(predicted)))
    assert cosine > 0.97, "nominal geometry should already point the right way"

    z_channels = [2, 5, 8]
    ratio = np.abs(measured[z_channels] / predicted[z_channels])
    assert np.all(ratio < 0.9), "nominal magnets are stronger than the real ones"
    assert ratio[2] < ratio[0] - 0.1, "magnet 3 is weaker still"


def test_third_magnet_is_reversed(session):
    """A negative Bz on sensor 3 at rest, and only on sensor 3."""
    measured = session.rest_mean()
    assert measured[2] > 300.0
    assert measured[5] > 300.0
    assert measured[8] < -300.0


#: Sign of every measurement channel's response to every degree of freedom, at
#: pose zero, in counts per millimetre and per degree. Zero means "not
#: asserted": those entries are under 3.2 counts where the significant ones are
#: all above 12.1, so their sign is arithmetic noise rather than a convention.
#:
#: **What anchors this to physical reality is a person, not a derivation.** On
#: 2026-07-29 the knob was moved by hand while `view.py` named each direction,
#: and all six axes read correctly. Nothing in the recorded data can substitute
#: for that: the sessions contain motion in both directions on every axis --
#: even `tz`, which is 41 % positive -- so there is no one-sided excursion whose
#: physical direction is known, and the sign of a principal component is
#: arbitrary anyway.
#:
#: So this is a change-detector, and that is the point. It cannot tell you the
#: convention is right; it will tell you the moment it stops being what was
#: confirmed by eye. Mirror an axis, swap a channel pair, or flip a magnet's
#: polarity, and this fails while every residual- and consistency-based test in
#: the suite stays green.
EXPECTED_JACOBIAN_SIGNS = {
    "mag1x": (+1, 0, 0, 0, -1, +1),
    "mag1y": (0, +1, -1, +1, 0, 0),
    "mag1z": (0, -1, -1, +1, 0, 0),
    "mag2x": (+1, 0, -1, 0, -1, -1),
    "mag2y": (0, +1, +1, +1, 0, -1),
    "mag2z": (-1, +1, -1, -1, -1, 0),
    "mag3x": (-1, 0, -1, 0, +1, +1),
    "mag3y": (0, -1, -1, -1, 0, -1),
    "mag3z": (-1, -1, +1, +1, -1, 0),
}

#: Below this a Jacobian entry carries no convention worth freezing. Chosen in
#: the gap: the smallest asserted entry is 12.1, the largest ignored one 3.2.
SIGN_THRESHOLD = 10.0


def test_measurement_sign_convention_is_frozen(table, nominal):
    """Freeze the hand-verified sign convention against silent mirroring.

    See :data:`EXPECTED_JACOBIAN_SIGNS`. If this fails, do not adjust the table
    to match the code -- go and look at `view.py` again, because either the
    convention really has changed or something has been mirrored.
    """
    _, jac = forward_and_jac(np.zeros(POSE_DIM), nominal, table)
    per_unit = jac * np.array([1.0, 1.0, 1.0, *(np.deg2rad(1.0),) * 3])

    from cadmouse.geometry import CHANNEL_NAMES

    wrong = []
    for row, name in enumerate(CHANNEL_NAMES):
        for col, expected in enumerate(EXPECTED_JACOBIAN_SIGNS[name]):
            value = per_unit[row, col]
            if expected == 0:
                assert abs(value) < SIGN_THRESHOLD * 1.5, (
                    f"{name} vs dof {col} grew to {value:.1f}; it used to be negligible, "
                    "so the frozen pattern needs regenerating deliberately"
                )
            elif abs(value) <= SIGN_THRESHOLD:
                # A response that was meant to be strong going quiet is just as
                # much a broken convention as one that changed sign, and it is
                # easier to miss. Mirroring the magnet ring in z does exactly
                # this: the magnets end up 27 mm from the sensors instead of 9,
                # every entry collapses, and a sign-only check sees nothing.
                wrong.append(
                    f"{name} vs dof {col}: {value:+.1f}, expected a strong {expected:+d}"
                )
            elif np.sign(value) != expected:
                wrong.append(f"{name} vs dof {col}: {value:+.1f}, expected sign {expected:+d}")
    assert not wrong, "sign convention changed:\n  " + "\n  ".join(wrong)


def test_translation_moves_magnets_the_obvious_way(nominal):
    """The convention one level down: +x really is right in the body frame.

    Cheap, and it localises a failure. If this passes but the Jacobian signs
    change, the problem is in the field model rather than in the pose
    parameterisation.
    """
    from cadmouse.model import _geometry

    for axis in range(3):
        pose = np.zeros(POSE_DIM)
        pose[axis] = 1.0
        _, centres, _, _ = _geometry(pose, nominal)
        base = _geometry(np.zeros(POSE_DIM), nominal)[1]
        assert np.allclose(centres[:, axis] - base[:, axis], 1.0)


@pytest.mark.parametrize("segment", sorted(SEGMENT_AXIS))
def test_measured_response_direction_matches_the_model(table, nominal, session, segment):
    """Each motion segment moves the measurement along the axis the model says.

    Correlating the model's Jacobian column against the first principal
    component of the segment is what distinguished a reversed magnet 3 from an
    upside-down sensor 3, and it is a sharp test of channel order: a swapped
    pair of channels drops these cosines through the floor.

    It says nothing about *sign*, and the ``abs`` below is why. The sign of a
    principal component is arbitrary -- SVD is free to return either -- and the
    operator moved each axis both ways, so there is no direction to compare
    against. :func:`test_measurement_sign_convention_is_frozen` covers that.
    """
    rest = session.rest_mean()
    runs = session.by_segment(segment)
    assert runs, f"session has no '{segment}' frames"

    data = np.concatenate([r.counts for r in runs]) - rest
    centred = data - data.mean(axis=0)
    principal = np.linalg.svd(centred, full_matrices=False)[2][0]

    _, jac = forward_and_jac(np.zeros(POSE_DIM), nominal, table)
    column = jac[:, SEGMENT_AXIS[segment]]
    cosine = abs(float(principal @ column / np.linalg.norm(column)))
    assert cosine > 0.85, f"{segment}: model and data disagree on direction ({cosine:.3f})"


def test_measurement_dimensions(table, nominal):
    predicted, jac = forward_and_jac(np.zeros(POSE_DIM), nominal, table)
    assert predicted.shape == (MEAS_DIM,)
    assert jac.shape == (MEAS_DIM, POSE_DIM)
