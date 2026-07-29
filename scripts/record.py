#!/usr/bin/env python3
"""Guided capture of a calibration session.

Walks you through the motions a calibration needs and writes one CSV. The
prompts matter: a fit uses the segment label to know which DOF you were *asked*
to move, and it uses the ``rest`` blocks to define pose zero and to measure the
sensor noise.

    uv run record.py --check              # link health only, no file
    uv run record.py -o data/session1.csv

Move slowly -- several seconds per traverse. The three sensors are read
sequentially over one I2C bus (``src/sensors.rs`` ``read_raw``), so a fast motion
smears the three readings across a few hundred microseconds of different poses.
That is harmless for quasi-static calibration data and unfixable here.

This script is deliberately self-contained: stdlib plus pyserial, no shared
package. It is pure data capture and has no opinion about how the poses are
later estimated, so it survives rewrites of the estimator.
"""

from __future__ import annotations

import argparse
import csv
import glob
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# --------------------------------------------------------------------------
# Wire format
#
# Defined by `format_frame` in ``src/sensors.rs`` and must be kept in step with
# it:
#
#   offset  size  field
#   0          2  magic 0xA55A
#   2          2  seq, wrapping frame counter
#   4          4  t_us, device uptime in microseconds
#   8         18  nine int16 raw counts, MAG1/2/3 x,y,z
#
# The sequence number is the point of the whole exercise. The firmware
# increments it on every *attempted* read including failed ones, so a gap here
# means a sample was genuinely lost rather than the hand having paused -- which
# is the difference between a usable velocity estimate and a quietly wrong one.
# --------------------------------------------------------------------------

FRAME_MAGIC = 0xA55A
FRAME_LEN = 26
_FRAME = struct.Struct("<HHI9h")

assert _FRAME.size == FRAME_LEN, "struct format out of step with FRAME_LEN"

#: The firmware's USB identity (`src/main.rs`): VID 0xc0de, PID 0xcafe,
#: manufacturer "CAD Mouse", product "CAD Mouse MK2", serial "00000001".
DEVICE_GLOB = "/dev/serial/by-id/*CAD_Mouse*"

#: Channel order of the 9-vector, matching the firmware's ``[i16; 9]``
#: (``src/sensors.rs`` ``read_raw``) and the C++ reference's ``readRaw``.
CHANNEL_NAMES = [f"mag{i + 1}{ax}" for i in range(3) for ax in "xyz"]

#: Raw counts are sign-extended 12-bit (`../tli493d/src/register.rs`), so a
#: reading at this magnitude means the ADC railed and the sample is unusable.
ADC_FULL_SCALE_COUNTS = 2047

#: Segment labels with meaning beyond "this was the axis I asked for".
REST_SEGMENT = "rest"
HELDOUT_SEGMENT = "free"

#: (label, seconds, instruction). Every sweep is bracketed by a rest block: they
#: pin the pose origin, and interleaving them spreads any slow bias drift across
#: the whole session instead of letting it alias onto one axis.
DEFAULT_PLAN: list[tuple[str, float, str]] = [
    (REST_SEGMENT, 3.0, "Hands OFF the knob."),
    ("tx", 6.0, "Slide the knob LEFT and RIGHT, full travel, a few slow cycles."),
    (REST_SEGMENT, 2.0, "Hands off."),
    ("ty", 6.0, "Slide the knob FORWARD and BACK, full travel."),
    (REST_SEGMENT, 2.0, "Hands off."),
    ("tz", 6.0, "Press the knob DOWN and let it rise, full travel."),
    (REST_SEGMENT, 2.0, "Hands off."),
    ("rx", 6.0, "TILT the knob forward and back (nose down, nose up)."),
    (REST_SEGMENT, 2.0, "Hands off."),
    ("ry", 6.0, "TILT the knob left and right."),
    (REST_SEGMENT, 2.0, "Hands off."),
    ("rz", 6.0, "TWIST the knob clockwise and anticlockwise."),
    (REST_SEGMENT, 2.0, "Hands off."),
    (
        HELDOUT_SEGMENT,
        20.0,
        "Move the knob however you like -- all six axes, mixed. "
        "This block is NEVER fitted; it is the honest test of the model.",
    ),
    (REST_SEGMENT, 2.0, "Hands off. Done after this."),
]


def find_port() -> str:
    """Locate the device's CDC data interface.

    Prefers the stable ``by-id`` symlink over ``/dev/ttyACM*``, which renumbers
    between plug-ins. The firmware exposes one CDC class, so the first match is
    the data interface.
    """
    matches = sorted(glob.glob(DEVICE_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"no CAD Mouse found at {DEVICE_GLOB}. Is it plugged in and running "
            "firmware with the binary stream (see src/sensors.rs format_frame)?"
        )
    return matches[0]


@dataclass
class Frame:
    seq: int
    t_us: int
    counts: tuple[int, ...]  # nine raw counts, MAG1/2/3 x,y,z


@dataclass
class FrameDecoder:
    """Incremental byte-stream decoder with resynchronisation.

    USB CDC carries no record boundaries, so a reader that attaches mid-frame
    lands at an arbitrary offset. Scanning for the magic word recovers, and
    `resyncs` counts how often that was needed -- a nonzero value on a settled
    link means frames are being corrupted, not merely dropped.
    """

    buffer: bytearray = field(default_factory=bytearray)
    resyncs: int = 0

    def feed(self, chunk: bytes) -> Iterator[Frame]:
        self.buffer.extend(chunk)
        while len(self.buffer) >= FRAME_LEN:
            if not (
                self.buffer[0] == (FRAME_MAGIC & 0xFF)
                and self.buffer[1] == (FRAME_MAGIC >> 8)
            ):
                start = self.buffer.find(FRAME_MAGIC.to_bytes(2, "little"), 1)
                if start < 0:
                    # Keep one byte: the magic may straddle the next chunk.
                    del self.buffer[:-1]
                    return
                del self.buffer[:start]
                self.resyncs += 1
                continue

            magic, seq, t_us, *counts = _FRAME.unpack_from(self.buffer, 0)
            del self.buffer[:FRAME_LEN]
            if magic != FRAME_MAGIC:  # defensive; the scan above should prevent it
                self.resyncs += 1
                continue
            yield Frame(seq=seq, t_us=t_us, counts=tuple(counts))


@dataclass
class StreamStats:
    """Link health, accumulated as frames arrive."""

    received: int = 0
    dropped: int = 0
    resyncs: int = 0
    first_t_us: int | None = None
    last_t_us: int | None = None
    _last_seq: int | None = None

    def observe(self, frame: Frame) -> None:
        if self._last_seq is not None:
            gap = (frame.seq - self._last_seq - 1) & 0xFFFF
            # A huge "gap" is a rewind (device reset), not 65k lost frames.
            if gap < 0x8000:
                self.dropped += gap
        self._last_seq = frame.seq
        self.received += 1
        if self.first_t_us is None:
            self.first_t_us = frame.t_us
        self.last_t_us = frame.t_us

    @property
    def duration_s(self) -> float:
        if self.first_t_us is None or self.last_t_us is None:
            return 0.0
        return ((self.last_t_us - self.first_t_us) & 0xFFFFFFFF) / 1e6

    @property
    def rate_hz(self) -> float:
        return self.received / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def loss_fraction(self) -> float:
        total = self.received + self.dropped
        return self.dropped / total if total else 0.0

    def summary(self) -> str:
        return (
            f"{self.received} frames in {self.duration_s:.1f} s "
            f"({self.rate_hz:.0f} Hz), {self.dropped} dropped "
            f"({self.loss_fraction:.2%}), {self.resyncs} resyncs"
        )


def read_frames(
    port: str | None = None,
    timeout: float = 1.0,
    chunk: int = 4096,
) -> Iterator[tuple[Frame, StreamStats]]:
    """Yield frames from the device indefinitely, with running link stats."""
    import serial  # imported lazily so --help needs no pyserial

    port = port or find_port()
    decoder = FrameDecoder()
    stats = StreamStats()

    # Baud rate is ignored by USB CDC but pyserial requires one.
    with serial.Serial(port, baudrate=115200, timeout=timeout) as handle:
        handle.reset_input_buffer()
        while True:
            data = handle.read(max(1, min(chunk, handle.in_waiting or 1)))
            if not data:
                continue
            for frame in decoder.feed(data):
                stats.resyncs = decoder.resyncs
                stats.observe(frame)
                yield frame, stats


def _prompt(message: str) -> None:
    print(f"\n>>> {message}")
    input("    Press Enter when ready...")


def check_link(port: str | None, seconds: float = 3.0) -> StreamStats:
    """Report frame rate, loss, and clipping without recording anything."""
    print(f"Reading {seconds:.0f} s from {port or find_port()} ...")

    peak = 0
    saturated = 0
    stats = StreamStats()
    deadline = time.monotonic() + seconds
    for frame, stats in read_frames(port):
        largest = max(abs(c) for c in frame.counts)
        peak = max(peak, largest)
        if largest >= ADC_FULL_SCALE_COUNTS:
            saturated += 1
        if time.monotonic() >= deadline:
            break

    if not stats.received:
        print("  no frames received -- is the firmware streaming?")
        return stats

    print(f"  {stats.summary()}")
    print(
        f"  peak |count| = {peak} of {ADC_FULL_SCALE_COUNTS} "
        f"({peak / ADC_FULL_SCALE_COUNTS:.0%} of full scale)"
    )

    if stats.loss_fraction > 0.02:
        print("  WARNING: >2% frame loss. Check the USB link before recording.")
    if stats.resyncs:
        print("  WARNING: frame resyncs on a settled link means corruption.")
    if saturated:
        print(
            f"  ERROR: {saturated} frames railed the ADC. Those carry no "
            "information and a fit will refuse them."
        )
    return stats


def record_segment(
    label: str, seconds: float, port: str | None, writer: csv.writer
) -> int:
    """Record one prompted segment for `seconds`, streaming rows to `writer`.

    Timed, not counted. The instruction on screen asks a person to sweep an axis
    for a stated number of seconds, so the segment has to end when that many
    seconds have passed -- whatever rate the device happens to be running at.
    """
    written = 0
    started = time.monotonic()
    deadline = started + seconds
    for frame, stats in read_frames(port):
        writer.writerow([label, frame.seq, frame.t_us, *frame.counts])
        written += 1
        now = time.monotonic()
        if written % 128 == 0:
            pct = 100 * min(1.0, (now - started) / seconds)
            print(
                f"\r    {label:5s} {pct:5.1f}%  {stats.rate_hz:4.0f} Hz  "
                f"{stats.loss_fraction:.1%} lost",
                end="",
                flush=True,
            )
        if now >= deadline:
            break
    elapsed = time.monotonic() - started
    print(f"\r    {label:5s} done, {written} frames in {elapsed:.1f} s" + " " * 16)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--output", type=Path, help="CSV to write")
    ap.add_argument("-p", "--port", help="serial port (default: autodetect)")
    ap.add_argument(
        "--check", action="store_true", help="report link health and exit"
    )
    args = ap.parse_args(argv)

    try:
        port = args.port or find_port()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        check_link(port)
        return 0

    if args.output is None:
        ap.error("--output is required unless --check is given")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Recording to {args.output} from {port}")
    print("Move SLOWLY -- several seconds per traverse.")

    total = 0
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["segment", "seq", "t_us", *CHANNEL_NAMES])
        for label, seconds, instruction in DEFAULT_PLAN:
            _prompt(f"[{label}] {instruction}  ({seconds:.0f} s)")
            total += record_segment(label, seconds, port, writer)

    print(f"\nWrote {total} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
