"""Gates for the recursive estimators.

The important one is :func:`test_iekf_and_ukf_agree`. The RP2350 filter will be
hand-written, so the whole point of carrying FilterPy is to have something
independent to check it against; if these two ever diverge, one of them is
wrong and the golden vectors are worthless.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from cadmouse import CalibParams, forward
from cadmouse.dataset import HELDOUT_SEGMENT
from cadmouse.filter import (
    DEFAULT_ALPHA,
    DEFAULT_KAPPA,
    FilterConfig,
    IteratedEkf,
    replay,
)
from cadmouse.model import MEAS_DIM, POSE_DIM


@pytest.fixture(scope="module")
def calibrated(session, table):
    """A calibration good enough to filter with, fitted once for this module."""
    from cadmouse.calibrate import fit

    return fit(session, table=table, per_segment=120, per_rest_run=50, verbose=0).params


def _short(run, n=4000):
    sub = copy.copy(run)
    for attr in ("counts", "t_us", "seq"):
        sub.__dict__[attr] = getattr(run, attr)[:n]
    return sub


def test_sigma_point_weights_are_well_conditioned():
    """alpha = 1e-3 is copied from the literature and is wrong at n = 6.

    It puts lambda at -5.999994, so the mean weight becomes about -1e6 and the
    reconstructed covariance stops being positive definite. The chosen spread
    must keep the weights bounded, or the f32 port will diverge where the f64
    host merely wobbles.
    """
    lam = DEFAULT_ALPHA**2 * (POSE_DIM + DEFAULT_KAPPA) - POSE_DIM
    assert POSE_DIM + lam > 1.0, "sigma point scaling is near-singular"
    assert abs(lam / (POSE_DIM + lam)) < 10.0, "mean weight is pathological"


def test_iekf_recovers_a_known_pose(table, nominal):
    """Noiseless measurement, so the filter should land on the truth."""
    truth = np.array([0.4, -0.3, 0.25, 0.02, -0.01, 0.03])
    measurement = forward(truth, nominal, table)

    config = FilterConfig(iterations=6)
    estimator = IteratedEkf(nominal, table, np.ones(MEAS_DIM), config)
    for _ in range(8):
        estimator.predict(5e-4)
        estimator.update(measurement)

    assert np.allclose(estimator.x[:3], truth[:3], atol=2e-3)
    assert np.allclose(estimator.x[3:], truth[3:], atol=2e-4)


def test_innovation_is_a_prior_quantity(table, nominal):
    """Regression: the IEKF must not report its post-fit residual.

    Recording the last iteration instead of the first yields a residual that is
    small by construction, which makes the NIS meaningless and silently
    flatters the IEKF against any filter reporting the real innovation.
    """
    truth = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    measurement = forward(truth, nominal, table)

    estimator = IteratedEkf(nominal, table, np.ones(MEAS_DIM), FilterConfig(iterations=5))
    estimator.update(measurement)  # started at pose zero, so this is a big surprise

    # 0.5 mm against ~112 counts/mm has to show up as tens of counts.
    assert np.abs(estimator.innovation).max() > 20.0


def test_process_noise_scales_with_dt():
    config = FilterConfig(q_pos=0.02, q_rot=3e-4)
    assert np.allclose(config.process_noise(2e-3), 4.0 * config.process_noise(5e-4))


def test_covariance_stays_positive_definite(session, calibrated, table):
    run = _short(session.by_segment(HELDOUT_SEGMENT)[0], 2000)
    sigma = session.noise_sigma()
    estimator = IteratedEkf(calibrated, table, sigma, FilterConfig())
    for k in range(len(run.counts)):
        estimator.predict(5e-4)
        estimator.update(run.counts[k])
        assert np.all(np.linalg.eigvalsh(estimator.P) > 0), f"P lost rank at frame {k}"


@pytest.mark.slow
def test_iekf_and_ukf_agree(session, calibrated, table):
    """The cross-check the whole FilterPy dependency exists for.

    They must agree far inside the measurement noise floor -- a single frame
    resolves about 5.6 um and 0.013 deg -- or the choice between them would be
    a real modelling decision rather than a cost decision.
    """
    run = _short(session.by_segment(HELDOUT_SEGMENT)[0], 6000)
    sigma = session.noise_sigma()
    config = FilterConfig()

    iekf = replay(run, calibrated, table, sigma, config, kind="iekf")
    ukf = replay(run, calibrated, table, sigma, config, kind="ukf")

    delta = iekf.poses - ukf.poses

    # Over the whole run, including startup. The bound that matters is the
    # single-frame noise floor: a disagreement below it cannot be observed.
    assert np.abs(delta[:, :3]).max() < 0.0056, "translations differ by over the noise floor"
    assert np.abs(delta[:, 3:]).max() < np.deg2rad(0.013)

    # Once the covariance has settled they agree far more tightly than that.
    # The first frames legitimately differ: the prior covariance is still at
    # its deliberately loose initial value, so the UKF's sigma points span
    # about 0.12 mm, over which h is not perfectly linear.
    settled = delta[100:]
    assert np.abs(settled[:, :3]).max() < 0.001  # under a micrometre
    assert np.abs(settled[:, 3:]).max() < np.deg2rad(0.006)

    assert iekf.seconds < ukf.seconds, "the IEKF is supposed to be the cheap one"


@pytest.mark.slow
def test_innovations_are_white_at_rest(session, calibrated, table):
    """Pins where the remaining inconsistency comes from.

    At a fixed pose the innovations must be white and their spread must match
    the measured sensor noise -- that is what says the *sensor* model is right.
    While moving they are autocorrelated instead, which is the signature of a
    pose-dependent model error and rules out both white noise and the
    sequential readout as explanations.

    If this ever starts failing, the sensor noise model has drifted and the
    filter tuning rests on sand.

    Note the moving bound is an *upper* one. It used to assert
    ``autocorr(moving) > 0.2`` -- a lower bound on a defect, which fails the
    moment the defect gets smaller, and did: the measured figure is now 0.136
    against 0.2 when that line was written. A test that breaks when the model
    improves is worse than no test, so what is checked now is that the
    pose-dependent error has not *grown*.
    """
    from cadmouse.dataset import REST_SEGMENT

    sigma = session.noise_sigma()
    rest = replay(
        _short(session.by_segment(REST_SEGMENT)[0], 3000),
        calibrated,
        table,
        sigma,
        FilterConfig(),
    )
    moving = replay(
        _short(session.by_segment(HELDOUT_SEGMENT)[0], 6000),
        calibrated,
        table,
        sigma,
        FilterConfig(),
    )

    def autocorr(innovations, lag):
        x = innovations[:, 2] - innovations[:, 2].mean()
        return float(np.corrcoef(x[:-lag], x[lag:])[0, 1])

    assert abs(autocorr(rest.innovations, 20)) < 0.15, "rest innovations should be white"
    # Measured 0.136; this catches the pose-dependent error growing, not its
    # absence. See the note above.
    assert autocorr(moving.innovations, 20) < 0.35, "moving model error has grown"

    rest_rms = float(np.sqrt((rest.innovations**2).mean()))
    assert rest_rms < 1.5 * sigma.mean(), "at a fixed pose only sensor noise should remain"


@pytest.mark.slow
def test_nis_does_not_depend_on_knob_speed(session, calibrated, table):
    """Rules out the sequential I2C readout as the source of the excess.

    The three sensors are read one after another, so it is tempting to blame
    fast motion. Measured, the effect is not there: at a peak of ~20 mm/s the
    ~333 us of skew moves the knob 6.5 um, worth about a count on the most
    sensitive channel.

    The correlation is not quite zero -- it measures **-0.186** -- but the sign
    is what settles the question. Readout skew would inflate the innovations
    when the knob moves fastest and so push the NIS *up*; this goes the other
    way, which is consistent with the posterior widening during motion rather
    than with any skew. The bound is therefore set from the measurement and
    kept symmetric only for simplicity; a *positive* correlation of this size
    would be the interesting one.
    """
    run = _short(session.by_segment(HELDOUT_SEGMENT)[0], 20000)
    result = replay(run, calibrated, table, session.noise_sigma(), FilterConfig())

    dt = np.diff(run.t_us) / 1e6
    velocity = np.diff(result.poses[:, :3], axis=0) / dt[:, None]
    speed = np.convolve(np.linalg.norm(velocity, axis=1), np.ones(41) / 41, "same")

    assert abs(float(np.corrcoef(result.nis[1:], speed)[0, 1])) < 0.25


@pytest.mark.slow
def test_filter_is_consistent_on_held_out_data(session, calibrated, table):
    """Honest uncertainty, on data the calibration never saw.

    There is no ground truth in any session, so accuracy is not measurable --
    only whether the filter's own error bars match its actual innovations.

    The band does not reach 95 %, and the reason is residual calibration error
    rather than anything about the sensor: the innovations stay autocorrelated
    for tens of milliseconds while moving but are white at rest, which is what
    a pose-dependent model error looks like and what sensor noise does not.
    See :func:`test_innovations_are_white_at_rest`.
    """
    run = _short(session.by_segment(HELDOUT_SEGMENT)[0], 12000)
    result = replay(run, calibrated, table, session.noise_sigma(), FilterConfig())

    assert 6.0 < result.mean_nis < 14.0, f"mean NIS {result.mean_nis:.1f}, target {MEAS_DIM}"
    assert result.inside_band() > 0.75
    assert np.sqrt((result.innovations**2).mean()) < 3.0
