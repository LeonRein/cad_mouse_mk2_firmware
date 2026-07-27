"""Can the calibration recover parameters it was never told?

The gate before any hardware. Synthetic data is generated from known
parameters, the calibration is run on it blind, and the result is checked
against the truth. Without this, a bad fit on real data is ambiguous between
"the solver is broken" and "the hardware is not what we assumed"; with it, those
are separable.

The pieces are tested at increasing difficulty: the linear solves exactly, the
bootstrap approximately, then the whole pipeline. The marked-slow tests run the
full joint optimisation and take minutes.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadmouse import calibrate, forward, geometry, simulate
from cadmouse.params import CalibParams, signed_permutations_planar


@pytest.fixture(scope="module")
def session():
    """One synthetic session, shared across tests -- generation is not free."""
    data, truth, poses = simulate.make_session(seed=1)
    return data, truth, poses


# ------------------------------------------------------- the linear machinery


def test_linear_gains_are_exact_given_poses_and_moments(session):
    """With poses and moments known, gains and biases follow in closed form.

    This is the property the whole bootstrap rests on, so it is checked to
    floating-point precision on noiseless data rather than approximately.
    """
    _, truth, poses = session
    clean = forward.predict(poses, truth, forward.COARSE_MODEL)
    board = forward.board_field(poses, truth, forward.COARSE_MODEL)

    gains, biases = calibrate.linear_gains(clean, board)
    assert gains == pytest.approx(truth.gains, rel=1e-6)
    assert biases == pytest.approx(truth.biases, abs=1e-6)


def test_linear_moments_are_exact_given_poses_and_gains(session):
    """The mirror property: the model is linear in the magnet moments."""
    _, truth, poses = session
    clean = forward.predict(poses, truth, forward.COARSE_MODEL)
    basis = forward.moment_basis(poses, truth, forward.COARSE_MODEL)

    moments = calibrate.linear_moments(clean, basis, truth)
    assert moments == pytest.approx(truth.moments, abs=1e-6)


def test_field_is_linear_in_moments_for_both_models():
    """Both magnet models must be linear in the moments, not just the dipole.

    `SubdipoleCylinder` orients its sub-dipole cloud by the magnet's *body* axis
    rather than its moment, precisely so this holds. If that ever regresses, the
    linear bootstrap silently starts returning nonsense.
    """
    params = CalibParams.initial(gain_counts_per_unit=1.0)
    pose = np.array([0.6, -0.4, 0.3, 0.02, 0.01, -0.03])

    for model in (forward.COARSE_MODEL, forward.DEFAULT_MODEL):
        doubled = CalibParams.initial(gain_counts_per_unit=1.0)
        doubled.moments = 2.0 * params.moments
        assert forward.board_field(pose, doubled, model) == pytest.approx(
            2.0 * forward.board_field(pose, params, model), rel=1e-10
        )


def test_moment_basis_reconstructs_the_field(session):
    """The 9 unit-moment columns must span what the magnets actually produce."""
    _, truth, poses = session
    sub = poses[::40]
    basis = forward.moment_basis(sub, truth, forward.COARSE_MODEL)
    rebuilt = basis @ truth.moments.ravel()
    assert rebuilt == pytest.approx(
        forward.board_field(sub, truth, forward.COARSE_MODEL), rel=1e-9
    )


# ----------------------------------------------------------- the scale gauge


def test_pose_scale_is_nearly_degenerate_with_gain():
    """Document the degeneracy that makes `scale_residual` necessary.

    Scaling every pose by *s* while dividing every gain by *s* leaves the
    prediction almost unchanged -- only the model's curvature over a few
    millimetres of travel distinguishes them. This test measures how weak that
    signal is, because it is the reason the calibration needs a physical prior
    on the gain rather than an arbitrary gauge.
    """
    params = CalibParams.initial(
        gain_counts_per_unit=geometry.expected_gain_counts_per_unit()
    )
    poses = np.zeros((40, 6))
    poses[:, 0] = np.linspace(-2.5, 2.5, 40)  # full travel along x

    s = 1.10
    scaled = CalibParams.initial(
        gain_counts_per_unit=geometry.expected_gain_counts_per_unit() / s
    )
    # Absorb the resulting DC shift into the bias, as the fit would.
    base = forward.predict(poses, params)
    alt = forward.predict(poses * s, scaled)
    alt = alt + (base.mean(axis=0) - alt.mean(axis=0))

    discrepancy = np.sqrt(((alt - base) ** 2).mean())
    signal = base.std()
    # A 10 % pose-scale error survives at well under a percent of the signal:
    # far too little for a noisy fit to reject on its own.
    assert discrepancy < 0.02 * signal, (
        f"degeneracy is weaker than assumed ({discrepancy / signal:.3%}) -- "
        "the scale prior may no longer be needed"
    )


def test_gain_magnitude_is_orientation_independent():
    """`gain_magnitudes` must not depend on which axis convention is stored.

    Regression test. `gain_scales` is a *signed* projection onto the stored
    orientation, so when that orientation differs from the one actually baked
    into the gains, the signs partially cancel and the magnitude reads far too
    low -- by a factor of 2.6 in the case that caught this. The physical scale
    prior compares against a datasheet quantity that knows nothing about axis
    conventions, so it has to use the orientation-free norm.
    """
    g = 1.5e5
    for orientation in signed_permutations_planar():
        params = CalibParams.initial(orientation=orientation, gain_counts_per_unit=g)
        assert params.gain_magnitudes == pytest.approx(np.full(3, g), rel=1e-9)

        # Same gains, but the stored orientation forgotten (the bug's shape).
        amnesiac = CalibParams(
            gains=params.gains, biases=params.biases, moments=params.moments
        )
        assert amnesiac.gain_magnitudes == pytest.approx(np.full(3, g), rel=1e-9)


def test_magnet_position_offsets_round_trip():
    """The optional per-magnet position offsets must actually be fittable.

    Off by default -- the plan called for enabling them only if residuals warrant
    it -- but a flag that was never exercised is a flag that does not work. This
    checks the extra nine parameters survive packing and reach the prediction.
    """
    base = CalibParams.initial(gain_counts_per_unit=1.0)
    assert base.n_free == 45

    with_offsets = CalibParams.initial(
        gain_counts_per_unit=1.0, fit_magnet_offsets=True
    )
    assert with_offsets.n_free == 54

    vec = with_offsets.pack()
    vec[-9:] = np.arange(9) * 0.01
    restored = with_offsets.unpack(vec)
    assert restored.magnet_offsets.ravel() == pytest.approx(np.arange(9) * 0.01)
    assert restored.rest_magnet_positions == pytest.approx(
        restored.magnet_positions + restored.magnet_offsets
    )

    # A displaced magnet must change the prediction, or the parameter is inert.
    pose = np.zeros(6)
    assert not np.allclose(
        forward.predict(pose, restored), forward.predict(pose, with_offsets)
    )


def test_expected_gain_matches_the_hardware_scale():
    """The physical prediction must be the right order for this build.

    Derived from the datasheet sensitivity and the magnet's size and grade, with
    no reference to any fit. If this drifts far from what real sessions produce,
    the prior is pulling the calibration somewhere wrong.
    """
    gain = geometry.expected_gain_counts_per_unit()
    assert 5e4 < gain < 3e5, f"{gain:.3e} counts/unit is not a plausible scale"

    # The rest-pose field must land inside the ADC range but not be tiny:
    # a design that clipped or barely registered would be a hardware problem.
    params = CalibParams.initial(gain_counts_per_unit=gain)
    peak = np.abs(forward.predict(np.zeros(6), params)).max()
    assert 100 < peak < geometry.ADC_FULL_SCALE_COUNTS, (
        f"rest-pose peak {peak:.0f} counts is outside the usable ADC range"
    )


# ------------------------------------------------------------------ bootstrap


def test_bootstrap_beats_a_cold_start(session):
    """The bootstrap must explain most of the signal before the joint solve runs.

    Measured against the signal's own spread rather than an absolute count,
    since the count scale depends on the magnet strength and would otherwise
    make this test a hostage to the simulator's gain.
    """
    data, _, _ = session
    probe = data.fitted.subsample(250)
    sigma = calibrate.estimate_noise(data)

    _, _, rms = calibrate.bootstrap(probe, np.ones(6), sigma=sigma)
    signal = probe.counts.std()
    assert rms < 0.25 * signal, (
        f"bootstrap leaves {rms:.1f} of {signal:.1f} counts unexplained -- "
        "no better than a cold start"
    )


def test_sign_search_discriminates(session):
    """The 64 sign conventions must not all score the same.

    If the spread collapses, the search is not identifying anything and the
    pipeline is relying on luck.
    """
    data, _, _ = session
    ranked = calibrate.search_signs(data, n_probe=120)

    assert len(ranked) == 64
    best, worst = ranked[0][0], ranked[-1][0]
    assert worst > 1.5 * best, f"no discrimination: best {best:.1f}, worst {worst:.1f}"


# ------------------------------------------------------------ the whole thing


def _blind_fit(seed: int, stride: int = 5):
    """Run the whole pipeline on a synthetic session, truth-aligned.

    Subsampling by stride rather than `calibrate`'s own random thinning, so the
    recovered poses line up row-for-row with the known truth.
    """
    data, truth, true_poses = simulate.make_session(seed=seed)
    keep = data.segments != calibrate.HELDOUT_SEGMENT
    idx = np.arange(0, int(keep.sum()), stride)
    sub = calibrate.CalibrationData(
        data.counts[keep][idx], data.segments[keep][idx]
    )
    result = calibrate.calibrate(sub, n_probe=150, n_fit=10**6, n_candidates=2)
    return result, truth, true_poses[keep][idx], sub


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 2])
def test_full_calibration_recovers_pose(seed):
    """End to end, blind: does the fit recover poses it was never told?

    **This is the test that matters**, and its absence hid a broken calibration
    for a long time: an earlier version asserted only that the residual was
    small, which a wrong fit satisfies easily -- it drove the residual *below*
    the noise floor while recovering poses ten times too large.

    Tolerances are set against the Cramer-Rao bound from `observability.py`
    (~0.03 mm, ~0.07 deg for this geometry). A joint fit with no ground truth
    has no right to beat that, so landing within a small factor of it means the
    calibration is extracting essentially all the available information.
    """
    result, _, truth_poses, sub = _blind_fit(seed)

    sigma = calibrate.estimate_noise(sub)
    assert result.rms() < 1.5 * sigma.mean(), (
        f"residual {result.rms():.2f} counts vs noise floor {sigma.mean():.2f}"
    )

    err = simulate.pose_errors(result.poses, truth_poses)
    for dof in ("tx", "ty", "tz"):
        assert err[dof] < 0.15, f"{dof} error {err[dof]:.3f} mm"
    for dof in ("rx", "ry", "rz"):
        assert err[dof] < 0.40, f"{dof} error {err[dof]:.3f} deg"


@pytest.mark.slow
def test_full_calibration_recovers_metric_scale():
    """The recovered travel must be in real millimetres, not arbitrary units.

    This is what the physical gain prior buys. Without it the fit is free to
    scale poses against gains almost without penalty, and it happily reports a
    2 mm press as 12 mm -- explaining the data perfectly in the wrong units.
    """
    result, _, truth_poses, _ = _blind_fit(1)

    for k, name in enumerate(geometry.DOF_NAMES):
        true_span = float(np.ptp(truth_poses[:, k]))
        fit_span = float(np.ptp(result.poses[:, k]))
        if true_span < 1e-6:
            continue
        ratio = fit_span / true_span
        assert 0.8 < ratio < 1.25, (
            f"{name} span {fit_span:.3f} vs true {true_span:.3f} ({ratio:.2f}x)"
        )


@pytest.mark.slow
def test_full_calibration_generalises_to_held_out_motion():
    """Score on the `free` segment, which the fit never saw.

    A low residual here cannot be overfitting, so it is the one number that
    distinguishes a correct model from a merely flexible one.
    """
    from cadmouse import ukf

    data, _, _ = simulate.make_session(seed=1)
    keep = data.segments != calibrate.HELDOUT_SEGMENT
    idx = np.arange(0, int(keep.sum()), 5)
    sub = calibrate.CalibrationData(data.counts[keep][idx], data.segments[keep][idx])
    result = calibrate.calibrate(sub, n_probe=150, n_fit=10**6, n_candidates=2)

    heldout = data.heldout.counts[::4]
    poses = ukf.solve_all(heldout, result.params, result.model, sigma=result.sigma)
    residual = forward.predict(poses, result.params, result.model) - heldout
    rms = float(np.sqrt((residual**2).mean()))
    assert rms < 3.0 * result.sigma.mean(), (
        f"held-out RMS {rms:.2f} vs noise floor {result.sigma.mean():.2f}"
    )
