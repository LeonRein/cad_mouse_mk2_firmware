"""Gates for the data layer, checked against the recorded session."""

from __future__ import annotations

import numpy as np

from cadmouse.dataset import HELDOUT_SEGMENT, REST_SEGMENT, SEGMENT_AXIS


def test_session_structure(session):
    """The recording plan in ``record.py``, as it came out the other end."""
    # `usage` was added after this test was written -- it is what
    # `USAGE_ENVELOPE` in the exported calibration is measured from, and it
    # brought a ninth `rest` block with it.
    assert session.segments == [
        "rest", "tx", "ty", "tz", "rx", "ry", "rz", "free", "usage",
    ]
    assert len(session.by_segment(REST_SEGMENT)) == 9
    for segment in SEGMENT_AXIS:
        runs = session.by_segment(segment)
        assert len(runs) == 1
        assert 5.0 < runs[0].duration_s < 7.0
    assert 19.0 < session.by_segment(HELDOUT_SEGMENT)[0].duration_s < 21.0


def test_frame_rate_is_about_2khz(session):
    """A sanity bound on the readout, not a specification.

    The band is wide because the rate is not a firmware constant: the readout
    loop is unpaced and self-clocks on the sensors' conversion time, and it
    also rises whenever core 1 spends less time holding the shared critical
    section. Optimising the estimator moved it from about 2 kHz to 2.4 kHz
    without anyone touching the readout, which is what retired the original
    ``< 2100`` bound.

    What this still catches is the thing worth catching: a recording made at
    half the rate, or one that stalled.
    """
    for run in session.runs:
        rate = (len(run) - 1) / run.duration_s
        assert 1800 < rate < 3200, f"{run.segment}: {rate:.0f} Hz"


def test_noise_floor_is_about_one_count(session):
    """What the whole error budget is measured against.

    Anything the model gets wrong by more than this is worth fixing; anything
    below it is invisible.
    """
    sigma = session.noise_sigma()
    assert sigma.shape == (9,)
    assert np.all(sigma > 0.4)
    assert np.all(sigma < 3.0)
    assert 0.8 < float(np.median(sigma)) < 1.6


def test_rest_is_repeatable_across_the_session(session):
    """Bias drift over 90 s, which is what the runtime re-zero has to cover.

    This is the test doing its job rather than a threshold that wanted moving.
    Measured on `session1.csv`, per channel::

        x   1.3   1.4   1.5
        y   5.7   5.0   3.4
        z  11.5   6.3   4.3

    Almost all of it is in `z`, which is also the axis that leaks the runtime
    deadzone (about 2 % of frames at rest, against 0 % for `x` and `y`). Eleven
    counts is roughly ten times the noise floor, so thermal drift is *not*
    negligible over a 90 s recording -- it is simply covered, because the
    device re-zeroes at every power-up and on demand.

    The bound below is set from the measurement, not from taste. If it fails
    again the drift has grown beyond what one recording session can absorb,
    and the bundle adjustment -- which uses the `rest` blocks as its datum --
    is the thing that suffers first.
    """
    means = np.stack([r.counts.mean(axis=0) for r in session.by_segment(REST_SEGMENT)])
    drift = means.max(axis=0) - means.min(axis=0)
    assert drift.max() < 15.0, f"rest drifted by {drift.max():.1f} counts"


def test_nothing_clipped(session):
    """At 2x sensitivity the range holds; if it stops holding, the fit is junk."""
    assert session.clipped_fraction() == 0.0


def test_runs_are_contiguous_in_sequence(session):
    for run in session.runs:
        assert np.all(np.diff(run.seq) == 1)


def test_decimation_keeps_every_run(session):
    frames = session.decimate(per_segment=50)
    assert len({f.run_index for f in frames}) == len(session.runs)
    assert len(frames) <= 50 * len(session.runs)
    assert any(f.is_rest for f in frames)
    assert {f.segment for f in frames} == set(session.segments)


def test_decimation_spreads_across_each_traverse(session):
    """Even sampling in time, so a paused hand does not dominate the fit."""
    frames = [f for f in session.decimate(per_segment=100) if f.segment == "tz"]
    times = np.array([f.t_us for f in frames], dtype=float)
    gaps = np.diff(times)
    assert gaps.std() / gaps.mean() < 0.1


def test_axis_labels(session):
    frames = session.decimate(per_segment=5)
    for frame in frames:
        if frame.segment in SEGMENT_AXIS:
            assert frame.axis == SEGMENT_AXIS[frame.segment]
        else:
            assert frame.axis is None
