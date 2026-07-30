"""Loading and slicing a recorded session.

The CSV that :mod:`record.py` writes is one row per frame:
``segment, seq, t_us, mag1x .. mag3z``. Two things in it carry meaning beyond
the raw counts, and both are handled here.

``seq`` is a wrapping counter the firmware increments on every *attempted*
read, so a gap is a genuinely lost sample rather than a pause. That lets a run
of frames be split where the stream actually broke instead of where the clock
happened to jump.

The ``rest`` label is doing double duty: it defines pose zero, and it is the
only place the sensor's noise can be measured with the pose known. The eight
interleaved rest blocks also bracket every motion segment, which is what keeps
slow bias drift from aliasing onto one axis.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import ADC_FULL_SCALE_COUNTS, CHANNEL_NAMES

#: Segment labels with meaning beyond "this is the axis I was asked to move".
#: Kept in step with ``record.py``.
REST_SEGMENT = "rest"
HELDOUT_SEGMENT = "free"

#: Motion recorded at the amplitude the operator actually works at, as opposed
#: to the deliberately generous excursions everywhere else.
#:
#: It exists because the two things a recording is asked for pull in opposite
#: directions. The *fit* wants large excursions: magnet moment and magnet
#: standoff are nearly degenerate, and only watching the field change over real
#: travel separates them -- a gentle session refits the same hardware with
#: moments 37 % different and does not generalise. The *HID sensitivity* wants
#: the opposite: how far the knob is pushed in ordinary use, which is the one
#: thing a deliberately hard push does not tell you.
#:
#: So they are recorded separately. Like ``free`` this is never fitted; unlike
#: ``free`` it is not a test of anything, it is a measurement of the operator.
USAGE_SEGMENT = "usage"

#: Segments that carry pose information but are excluded from the fit.
NON_FITTED_SEGMENTS = (HELDOUT_SEGMENT, USAGE_SEGMENT)

#: Which pose component each motion segment was nominally exercising. Used only
#: to aim a *weak* prior at the other five: the recording asks for one axis at a
#: time, but the mechanism bleeds into the rest, and that bleed is a property of
#: the hardware worth measuring rather than something to force to zero.
SEGMENT_AXIS = {"tx": 0, "ty": 1, "tz": 2, "rx": 3, "ry": 4, "rz": 5}

#: A sequence gap wider than this ends a run. One dropped frame is a gap of 2.
_MAX_SEQ_GAP = 1

#: Frames to drop from the start of every run, in seconds.
#:
#: The three sensors are read sequentially over one I2C bus, so a frame that
#: straddles a change of pose mixes readings taken microseconds apart at
#: different poses. That is harmless mid-traverse, where the pose barely moves
#: between reads, but at the instant the hand *releases* the knob it produces a
#: frame that is not a rigid-body pose at all -- in ``session1.csv`` the first
#: frame of three rest blocks is out by up to 138 counts across all nine
#: channels, with the second frame already normal.
#:
#: Those frames matter more than their number suggests: rest frames are pinned
#: at pose zero, so a single one at 138 counts cannot be absorbed by a pose and
#: instead pulls on the geometry, and it inflates the measured noise floor by
#: about a third. Trimming is uniform across runs rather than special-cased to
#: rest, because the artefact belongs to the transition, not to the label.
SETTLE_S = 0.05


@dataclass(frozen=True)
class Run:
    """A contiguous stretch of frames with one segment label and no drops."""

    segment: str
    seq: np.ndarray  # (n,) int
    t_us: np.ndarray  # (n,) int
    counts: np.ndarray  # (n, 9) float, raw ADC counts

    def __len__(self) -> int:
        return self.counts.shape[0]

    @property
    def duration_s(self) -> float:
        return float(self.t_us[-1] - self.t_us[0]) / 1e6

    @property
    def axis(self) -> int | None:
        """Index of the pose component this segment was meant to exercise."""
        return SEGMENT_AXIS.get(self.segment)


@dataclass
class Session:
    runs: list[Run]
    path: Path

    # ------------------------------------------------------------- selection

    def by_segment(self, *segments: str) -> list[Run]:
        wanted = set(segments)
        return [r for r in self.runs if r.segment in wanted]

    def excluding(self, *segments: str) -> list[Run]:
        unwanted = set(segments)
        return [r for r in self.runs if r.segment not in unwanted]

    @property
    def segments(self) -> list[str]:
        seen: list[str] = []
        for run in self.runs:
            if run.segment not in seen:
                seen.append(run.segment)
        return seen

    # ------------------------------------------------------------- diagnostics

    def noise_sigma(self) -> np.ndarray:
        """Per-channel measurement noise, (9,) counts, from the rest blocks.

        Taken about each run's own mean rather than a global one, so the few
        counts of bias drift between blocks are not mistaken for noise. Hands
        are off the knob during these blocks, so what is left is the sensor.
        """
        rest = self.by_segment(REST_SEGMENT)
        if not rest:
            raise ValueError(f"{self.path} has no '{REST_SEGMENT}' frames")
        centred = np.concatenate([r.counts - r.counts.mean(axis=0) for r in rest])
        dof = sum(len(r) for r in rest) - len(rest)
        return np.sqrt((centred**2).sum(axis=0) / dof)

    def rest_mean(self) -> np.ndarray:
        """(9,) counts at pose zero, pooled over every rest block."""
        rest = self.by_segment(REST_SEGMENT)
        return np.concatenate([r.counts for r in rest]).mean(axis=0)

    def clipped_fraction(self) -> float:
        """Share of samples at the ADC rail, where the reading is unusable."""
        counts = np.concatenate([r.counts for r in self.runs])
        return float((np.abs(counts) >= ADC_FULL_SCALE_COUNTS).any(axis=1).mean())

    # ------------------------------------------------------------- decimation

    def decimate(self, per_segment: int = 400, per_rest_run: int | None = None) -> list["Frame"]:
        """Thin to a manageable set of frames for the bundle adjustment.

        The full session is around 146 000 frames, which would make the fit
        880 000 unknowns for no benefit: 27 calibration parameters are
        over-determined by orders of magnitude long before that. Sampling
        evenly *within each run* keeps every segment and every rest block
        represented, and keeps the sampled poses spread across each traverse
        rather than clustered wherever the hand paused.

        ``per_rest_run`` exists because the rest blocks are nearly identical to
        each other, so taking as many of them as of the motion frames would
        weight the pose datum by thousands of near-duplicate measurements and
        let it dominate the geometry. There are eight of them and only six
        motion segments.
        """
        frames: list[Frame] = []
        for run_index, run in enumerate(self.runs):
            cap = per_segment
            if run.segment == REST_SEGMENT and per_rest_run is not None:
                cap = per_rest_run
            take = min(cap, len(run))
            picks = np.unique(np.linspace(0, len(run) - 1, take).round().astype(int))
            for i in picks:
                frames.append(
                    Frame(
                        segment=run.segment,
                        run_index=run_index,
                        t_us=int(run.t_us[i]),
                        counts=run.counts[i],
                    )
                )
        return frames


@dataclass
class Frame:
    """One decimated sample, carrying enough context to build its priors."""

    segment: str
    run_index: int
    t_us: int
    counts: np.ndarray  # (9,)

    @property
    def axis(self) -> int | None:
        return SEGMENT_AXIS.get(self.segment)

    @property
    def is_rest(self) -> bool:
        return self.segment == REST_SEGMENT


def load_session(path: str | Path, settle_s: float = SETTLE_S) -> Session:
    """Read a recorded CSV and split it into contiguous, drop-free runs.

    ``settle_s`` is trimmed from the start of each run; see :data:`SETTLE_S`
    for why. Pass zero to see the recording exactly as it was captured.
    """
    path = Path(path)
    segments: list[str] = []
    seqs: list[int] = []
    times: list[int] = []
    counts: list[list[float]] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(["segment", "seq", "t_us", *CHANNEL_NAMES]) - set(
            reader.fieldnames or []
        )
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            segments.append(row["segment"])
            seqs.append(int(row["seq"]))
            times.append(int(row["t_us"]))
            counts.append([float(row[c]) for c in CHANNEL_NAMES])

    if not segments:
        raise ValueError(f"{path} has no rows")

    seq_arr = np.array(seqs, dtype=np.int64)
    time_arr = np.array(times, dtype=np.int64)
    count_arr = np.array(counts, dtype=float)

    # The counter is 16-bit and wraps; unwrap before differencing so a wrap is
    # not mistaken for a very large gap.
    wraps = np.cumsum(np.diff(seq_arr, prepend=seq_arr[0]) < -32768)
    unwrapped = seq_arr + wraps * 65536

    label_change = np.array(
        [True] + [segments[i] != segments[i - 1] for i in range(1, len(segments))]
    )
    seq_break = np.diff(unwrapped, prepend=unwrapped[0]) > _MAX_SEQ_GAP
    starts = np.flatnonzero(label_change | seq_break)
    bounds = list(starts) + [len(segments)]

    runs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        start = a
        if settle_s > 0.0:
            settled = np.searchsorted(time_arr[a:b], time_arr[a] + settle_s * 1e6)
            start = a + int(min(settled, max(b - a - 1, 0)))
        runs.append(
            Run(
                segment=segments[a],
                seq=unwrapped[start:b],
                t_us=time_arr[start:b],
                counts=count_arr[start:b],
            )
        )
    return Session(runs=runs, path=path)
