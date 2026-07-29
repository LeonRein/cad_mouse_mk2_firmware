"""Gates for the calibration fit."""

from __future__ import annotations

import numpy as np
import pytest

from cadmouse import CalibParams
from cadmouse.calibrate import (
    build_problem,
    detect_moment_signs,
    fit,
    initial_params,
    jacobian,
    residual,
)
from cadmouse.dataset import REST_SEGMENT


def test_polarity_is_read_from_the_data(session):
    """Magnet 3 reversed, the other two not -- detected, not assumed."""
    assert np.array_equal(detect_moment_signs(session), np.array([1.0, 1.0, -1.0]))


def test_initial_params_carry_the_detected_polarity(session):
    params = initial_params(session)
    assert np.array_equal(np.sign(params.magnet_moment), np.array([1.0, 1.0, -1.0]))
    assert np.allclose(np.abs(params.magnet_moment), np.abs(CalibParams.nominal().magnet_moment))


def test_polarity_detection_rejects_a_dead_session(session, monkeypatch):
    """A knob with no magnets must fail loudly rather than fit nonsense."""
    monkeypatch.setattr(type(session), "rest_mean", lambda self: np.zeros(9))
    with pytest.raises(ValueError, match="too weak to read polarity"):
        detect_moment_signs(session)


def test_rest_frames_are_pinned_and_motion_frames_are_not(session):
    params = initial_params(session)
    problem = build_problem(session, params, per_segment=20, per_rest_run=10)
    for frame, free in zip(problem.frames, problem.free_pose):
        assert free == (frame.segment != REST_SEGMENT)
    assert problem.n_free * 6 + problem.n_params == problem.n_unknowns


def test_heldout_segment_is_never_fitted(session):
    params = initial_params(session)
    problem = build_problem(session, params, per_segment=20, per_rest_run=10)
    assert all(f.segment != "free" for f in problem.frames)


def test_analytic_jacobian_matches_finite_differences(session, table):
    """The whole fit rests on this: every one of the 27 columns is analytic.

    Checked on the assembled residual, so it covers the priors and the
    smoothness rows as well as the measurement block, and it covers the
    bookkeeping that maps frames onto sparse columns.
    """
    params = initial_params(session)
    problem = build_problem(session, params, per_segment=6, per_rest_run=4)

    # Poses stay inside the envelope the mechanism reaches. Outside the table
    # the interpolant is deliberately flat, so agreement there would be a test
    # of the clamp rather than of the Jacobian.
    base = params.pack()
    rng = np.random.default_rng(0)
    poses = np.concatenate(
        [
            rng.uniform(-0.3, 0.3, (problem.n_free, 3)),
            rng.uniform(-np.deg2rad(3), np.deg2rad(3), (problem.n_free, 3)),
        ],
        axis=1,
    )
    x = np.concatenate([base[problem.param_mask], poses.ravel()])

    analytic = jacobian(problem, base, table, x).toarray()
    step = 1e-6
    for c in rng.choice(x.size, size=min(40, x.size), replace=False):
        hi, lo = x.copy(), x.copy()
        hi[c] += step
        lo[c] -= step
        numeric = (
            residual(problem, base, table, hi) - residual(problem, base, table, lo)
        ) / (2 * step)
        scale = max(1e-6, np.abs(analytic[:, c]).max())
        assert np.allclose(numeric, analytic[:, c], atol=2e-4 * scale, rtol=2e-4), (
            f"column {c} disagrees"
        )


@pytest.mark.slow
def test_fit_reaches_the_gate(session, table):
    """Stage 3's gate, on the held-out segment.

    ``free`` is 20 s of arbitrary six-axis motion that the calibration never
    sees. Six pose freedoms per frame cannot hide a wrong nine-channel model,
    so this is the honest test that the geometry is right rather than merely
    flexible.
    """
    result = fit(session, table=table, per_segment=120, per_rest_run=50, verbose=0)

    assert result.converged, "stopped at max_nfev; the result is not trustworthy"
    assert result.heldout.rms_counts < 3.0, result.summary()

    rest = next(s for s in result.scores if s.segment == REST_SEGMENT)
    assert rest.rms_sigma < 2.0, f"rest should sit near the noise floor: {rest}"

    # Polarity must survive the fit, and the moments stay physically plausible.
    assert np.array_equal(np.sign(result.params.magnet_moment), [1.0, 1.0, -1.0])
    assert np.all(np.abs(result.params.magnet_moment) > 0.02)


@pytest.mark.slow
def test_fit_does_not_depend_on_the_position_prior(session, table):
    """A prior that changes the answer is a prior doing the data's job.

    The magnet ring depth was never measured, so if the fit only lands in the
    right place because the prior held it there, the calibration would be
    reporting the drawing back rather than the device.
    """
    from cadmouse import calibrate
    from cadmouse.model import forward_batch

    original = calibrate.PRIOR_SIGMA["magnet_pos"]
    results = []
    try:
        for z_sigma in (0.5, 6.0):
            calibrate.PRIOR_SIGMA["magnet_pos"] = np.array([0.5, 0.5, z_sigma])
            results.append(fit(session, table=table, per_segment=120, per_rest_run=50, verbose=0))
    finally:
        calibrate.PRIOR_SIGMA["magnet_pos"] = original

    rng = np.random.default_rng(0)
    poses = np.concatenate(
        [rng.uniform(-1.2, 1.2, (200, 3)), np.deg2rad(rng.uniform(-4, 4, (200, 3)))], axis=1
    )
    spread = np.abs(
        forward_batch(poses, results[0].params, table)
        - forward_batch(poses, results[1].params, table)
    )
    rms = float(np.sqrt((spread**2).mean()))
    assert rms < 1.0, f"a 12x change of prior moved predictions by {rms:.2f} counts rms"
