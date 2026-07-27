#!/usr/bin/env python3
"""Guided capture of a calibration session.

Walks you through the motions the calibration needs and writes one CSV. The
prompts matter: the fit uses the segment label to know which DOF you were
*asked* to move, and it uses the ``rest`` blocks to define pose zero and to
measure the sensor noise.

    uv run record.py --check              # link health only, no file
    uv run record.py -o data/session1.csv

Move slowly -- several seconds per traverse. The three sensors are read
sequentially over one I2C bus (``src/sensors.rs`` ``read_raw``), so a fast motion
smears the three readings across a few hundred microseconds of different poses.
That is harmless for quasi-static calibration data and unfixable here.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from cadmouse import geometry
from cadmouse.calibrate import HELDOUT_SEGMENT, REST_SEGMENT
from cadmouse.stream import StreamStats, collect, find_port, read_frames

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

#: Nominal sample rate. The sensor's internal update rate caps around 770 Hz
#: (`../tli493d/src/driver.rs`), so this is what a healthy link looks like.
NOMINAL_RATE_HZ = 770.0


def _prompt(message: str) -> None:
    print(f"\n>>> {message}")
    input("    Press Enter when ready...")


def check_link(port: str | None, seconds: float = 3.0) -> StreamStats:
    """Report frame rate, loss, and clipping without recording anything."""
    print(f"Reading {seconds:.0f} s from {port or find_port()} ...")
    n = int(NOMINAL_RATE_HZ * seconds)
    counts, _, stats = collect(n, port=port)

    print(f"  {stats.summary()}")
    peak = int(np.abs(counts).max())
    print(
        f"  peak |count| = {peak} of {geometry.ADC_FULL_SCALE_COUNTS} "
        f"({peak / geometry.ADC_FULL_SCALE_COUNTS:.0%} of full scale)"
    )

    if stats.loss_fraction > 0.02:
        print("  WARNING: >2% frame loss. Check the USB link before recording.")
    if stats.resyncs:
        print("  WARNING: frame resyncs on a settled link means corruption.")
    saturated = int(geometry.is_saturated(counts).sum())
    if saturated:
        print(
            f"  ERROR: {saturated} frames railed the ADC. Those carry no "
            "information and the fit will refuse them."
        )
    return stats


def record_segment(
    label: str, seconds: float, port: str | None, writer: csv.writer
) -> int:
    n = int(NOMINAL_RATE_HZ * seconds)
    written = 0
    for frame, stats in read_frames(port):
        writer.writerow([label, frame.seq, frame.t_us, *frame.counts.tolist()])
        written += 1
        if written % 128 == 0:
            pct = 100 * written / n
            print(
                f"\r    {label:5s} {pct:5.1f}%  {stats.rate_hz:4.0f} Hz  "
                f"{stats.loss_fraction:.1%} lost",
                end="",
                flush=True,
            )
        if written >= n:
            break
    print(f"\r    {label:5s} done, {written} frames" + " " * 24)
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
        writer.writerow(["segment", "seq", "t_us", *geometry.CHANNEL_NAMES])
        for label, seconds, instruction in DEFAULT_PLAN:
            _prompt(f"[{label}] {instruction}  ({seconds:.0f} s)")
            total += record_segment(label, seconds, port, writer)

    print(f"\nWrote {total} frames to {args.output}")
    print(f"Next: uv run fit.py {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
