"""Pose recovery: the Gauss-Newton solver and the UKF.

These use a *known* calibration, so they test inversion only -- whether the
estimator can undo `forward.predict`. Recovering the calibration itself is
`test_roundtrip.py`'s job.
"""

from __future__ import annotations

import numpy as np
import pytest

from cadmouse import forward, geometry, ukf
from cadmouse.params import CalibParams
from cadmouse.simulate import TRAVEL


@pytest.fixture(scope="module")
def params() -> CalibParams:
    """A well-conditioned calibration: identity orientation, unequal magnets."""
    rng = np.random.default_rng(7)
    gains = np.broadcast_to(6.0e4 * np.eye(3), (3, 3, 3)).copy()
    moments = np.zeros((3, 3))
    moments[:, 2] = [1.0, 0.9, 1.1]
    return CalibParams(
        gains=gains,
        biases=rng.normal(0, 50, (3, 3)),
        moments=moments,
    )


def _trajectory(n: int, rate_hz: float = 200.0) -> np.ndarray:
    """A smooth multi-axis path, roughly hand-speed."""
    t = np.arange(n) / rate_hz
    poses = np.zeros((n, 6))
    for k in range(6):
        freq = 0.5 + 0.3 * k
        poses[:, k] = 0.7 * TRAVEL[k] * np.sin(2 * np.pi * freq * t + k)
    return poses


# ------------------------------------------------------------- Gauss-Newton


def test_solve_pose_is_exact_without_noise(params):
    """Noiseless inversion must return the pose that generated the data."""
    for pose in [
        np.zeros(6),
        np.array([1.0, -0.5, 0.3, 0.02, -0.03, 0.05]),
        np.array([-2.0, 1.5, -0.8, -0.06, 0.04, -0.09]),
    ]:
        counts = forward.predict(pose, params)
        assert ukf.solve_pose(counts, params) == pytest.approx(pose, abs=1e-4)


def test_solve_pose_converges_from_a_cold_start(params):
    """No warm start: this is what seeds the filter, so it must not need one."""
    pose = np.array([2.0, -2.0, 1.2, 0.08, -0.07, 0.10])
    counts = forward.predict(pose, params)
    assert ukf.solve_pose(counts, params, guess=None) == pytest.approx(pose, abs=1e-3)


def test_solve_pose_reaches_the_theoretical_bound(params):
    """Error under noise must match the Cramer-Rao bound, not merely be small.

    A single draw says nothing -- it is one sample of a distribution. What
    matters is that the solver is *efficient*: its spread over many draws should
    sit at the bound `observability.fisher_covariance` predicts from the
    geometry, meaning it extracts everything the measurement contains.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from observability import fisher_covariance

    rng = np.random.default_rng(3)
    pose = np.array([1.0, 0.5, -0.4, 0.03, 0.02, -0.04])
    sigma = np.full(9, 4.0)
    clean = forward.predict(pose, params)

    errors = np.array(
        [
            ukf.solve_pose(clean + rng.normal(0, sigma), params, guess=pose) - pose
            for _ in range(60)
        ]
    )
    empirical = errors.std(axis=0)
    predicted = np.sqrt(np.diag(fisher_covariance(pose, params, sigma)))

    # Within a factor of two of the bound in both directions: below it would
    # mean the bound is wrong, far above it that the solver is leaving
    # information on the table.
    ratio = empirical / predicted
    assert np.all(ratio < 2.0), f"solver is inefficient: {ratio}"
    assert np.all(ratio > 0.5), f"beating the CRB is impossible: {ratio}"


# --------------------------------------------------------------------- UKF


def test_ukf_tracks_a_trajectory(params):
    rng = np.random.default_rng(11)
    rate = 200.0
    truth = _trajectory(400, rate)
    counts = forward.predict(truth, params) + rng.normal(0, 4.0, (400, 9))

    poses = ukf.run(counts, params, np.full(9, 4.0), dt=1.0 / rate)

    # Skip the first samples: the filter needs a moment to settle its velocity.
    err = poses[50:] - truth[50:]
    rms_t = np.sqrt((err[:, :3] ** 2).mean())
    rms_r = np.rad2deg(np.sqrt((err[:, 3:] ** 2).mean()))
    assert rms_t < 0.10, f"translation RMS {rms_t:.4f} mm"
    assert rms_r < 0.60, f"rotation RMS {rms_r:.4f} deg"


def test_ukf_is_smoother_than_per_frame_solving(params):
    """The whole justification for the filter: it must beat the memoryless solve.

    Both see identical data. If the UKF is not smoother, its complexity is not
    earning anything and the Gauss-Newton baseline should be used instead.
    """
    rng = np.random.default_rng(5)
    rate = 200.0
    truth = _trajectory(300, rate)
    counts = forward.predict(truth, params) + rng.normal(0, 6.0, (300, 9))

    filtered = ukf.run(counts, params, np.full(9, 6.0), dt=1.0 / rate)[50:]
    per_frame = ukf.solve_all(counts, params)[50:]

    # Sample-to-sample jitter, the thing a user feels as noise.
    jitter_ukf = np.abs(np.diff(filtered, axis=0)).mean()
    jitter_gn = np.abs(np.diff(per_frame, axis=0)).mean()
    assert jitter_ukf < 0.75 * jitter_gn, (
        f"UKF jitter {jitter_ukf:.5f} vs per-frame {jitter_gn:.5f} -- "
        "not enough smoothing to justify the filter"
    )

    # Smoothing that costs accuracy would be a bad trade, so check both.
    err_ukf = np.sqrt(((filtered - truth[50:]) ** 2).mean())
    err_gn = np.sqrt(((per_frame - truth[50:]) ** 2).mean())
    assert err_ukf < err_gn, f"UKF error {err_ukf:.5f} vs per-frame {err_gn:.5f}"


def test_ukf_stays_finite_on_a_static_signal(params):
    """A long run at rest must not drift or break the covariance.

    Repeated updates with no excitation are where a sigma-point filter loses
    positive-definiteness, so this is the numerical-stability canary.
    """
    rng = np.random.default_rng(2)
    counts = forward.predict(np.zeros(6), params) + rng.normal(0, 4.0, (600, 9))
    poses = ukf.run(counts, params, np.full(9, 4.0), dt=1.0 / 770.0)

    assert np.all(np.isfinite(poses))
    assert np.all(np.abs(poses[-1, :3]) < 0.05)
    assert np.all(np.abs(np.rad2deg(poses[-1, 3:])) < 0.5)


def test_ukf_initialize_uses_the_first_frame(params):
    """A filter started at a displaced pose must not begin from zero."""
    pose = np.array([2.0, -1.0, 0.5, 0.05, -0.04, 0.06])
    counts = forward.predict(pose, params)

    filt = ukf.UKF(params=params, sigma=np.full(9, 4.0))
    filt.initialize(counts)
    assert filt.pose == pytest.approx(pose, abs=1e-3)
    assert filt.velocity == pytest.approx(np.zeros(6), abs=1e-9)


def test_sigma_point_weights_are_consistent():
    """Mean weights must sum to one, or the filter is biased by construction."""
    filt = ukf.UKF(
        params=CalibParams.initial(gain_counts_per_unit=1.0), sigma=np.ones(9)
    )
    assert filt.wm.sum() == pytest.approx(1.0)
    assert len(filt.wm) == 2 * ukf.N_STATE + 1
    assert filt._gamma > 0
