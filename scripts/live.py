#!/usr/bin/env python3
"""Stream from the device and print the live 6-DOF pose.

    uv run live.py --calib calib.json
    uv run live.py --calib calib.json --raw     # counts, no estimator

The qualitative check the residual tables cannot give: push the knob along one
axis and watch whether the other five stay near zero. That is the direct
comparison against the axis bleed documented in
``misc/original_firmware/README.md``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from cadmouse import calibrate, geometry, ukf
from cadmouse.params import CalibParams
from cadmouse.stream import find_port, read_frames

BAR_WIDTH = 21
#: Full-scale deflection for the bar display: millimetres, then degrees.
DISPLAY_SCALE = np.array([2.5, 2.5, 2.0, 7.0, 7.0, 9.0])


def bar(value: float, limit: float) -> str:
    """A centred text bar, so a glance shows sign and magnitude."""
    half = BAR_WIDTH // 2
    n = int(np.clip(value / limit, -1.0, 1.0) * half)
    cells = [" "] * BAR_WIDTH
    cells[half] = "|"
    for i in range(min(n, half) if n > 0 else 0):
        cells[half + 1 + i] = "#"
    for i in range(min(-n, half) if n < 0 else 0):
        cells[half - 1 - i] = "#"
    return "".join(cells)


def render(pose: np.ndarray, rate_hz: float, loss: float, extra: str = "") -> str:
    display = pose.copy()
    display[3:] = np.rad2deg(display[3:])
    units = ["mm", "mm", "mm", "deg", "deg", "deg"]

    lines = [f"  {rate_hz:5.0f} Hz   {loss:5.2%} lost   {extra}"]
    for k, name in enumerate(geometry.DOF_NAMES):
        lines.append(
            f"  {name:3s} [{bar(display[k], DISPLAY_SCALE[k])}] "
            f"{display[k]:+8.3f} {units[k]}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calib", type=Path, help="calib.json from fit.py")
    ap.add_argument("-p", "--port", help="serial port (default: autodetect)")
    ap.add_argument("--raw", action="store_true", help="show raw counts instead")
    ap.add_argument(
        "--gauss-newton",
        action="store_true",
        help="per-frame solve instead of the UKF (the no-dynamics baseline)",
    )
    ap.add_argument("--hz", type=float, default=20.0, help="display refresh rate")
    args = ap.parse_args(argv)

    try:
        port = args.port or find_port()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.raw and not args.calib:
        ap.error("--calib is required unless --raw is given")

    filt = None
    params = None
    if args.calib:
        params = CalibParams.load(args.calib)
        sigma = np.full(geometry.N_CHANNELS, 4.0)
        filt = ukf.UKF(params=params, sigma=sigma)

    print(f"Reading {port}. Ctrl-C to stop.\n")
    last_draw = 0.0
    last_t_us = None
    pose = np.zeros(6)
    lines_drawn = 0

    try:
        for frame, stats in read_frames(port):
            if args.raw:
                pose = np.zeros(6)
            elif filt is not None:
                if stats.received == 1:
                    filt.initialize(frame.counts.astype(float))
                    pose = filt.pose
                else:
                    dt = ((frame.t_us - last_t_us) & 0xFFFFFFFF) / 1e6
                    dt = float(np.clip(dt, 1e-5, 0.1))
                    if args.gauss_newton:
                        pose = ukf.solve_pose(
                            frame.counts.astype(float), params, guess=pose
                        )
                    else:
                        pose = filt.step(frame.counts.astype(float), dt)
            last_t_us = frame.t_us

            now = time.monotonic()
            if now - last_draw < 1.0 / args.hz:
                continue
            last_draw = now

            if args.raw:
                body = "  " + "  ".join(
                    f"{n}={v:6d}"
                    for n, v in zip(geometry.CHANNEL_NAMES, frame.counts)
                )
                text = f"  {stats.rate_hz:5.0f} Hz  {stats.loss_fraction:5.2%} lost\n{body}"
            else:
                mode = "gauss-newton" if args.gauss_newton else "ukf"
                text = render(pose, stats.rate_hz, stats.loss_fraction, mode)

            if lines_drawn:
                sys.stdout.write(f"\033[{lines_drawn}A")
            sys.stdout.write("\033[J" + text + "\n")
            sys.stdout.flush()
            lines_drawn = text.count("\n") + 1
    except KeyboardInterrupt:
        print("\n\nstopped.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
