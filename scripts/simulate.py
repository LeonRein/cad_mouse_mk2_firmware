#!/usr/bin/env python3
"""Write a synthetic calibration session, for testing without hardware.

    uv run simulate.py -o data/fake.csv
    uv run fit.py data/fake.csv

Useful for exercising the whole pipeline end to end, and for reproducing a fit
problem without needing the device. `tests/test_roundtrip.py` uses the same
generator to assert that the calibration actually recovers known parameters.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cadmouse import geometry, simulate as sim


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-o", "--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--noise", type=float, default=sim.NOISE_COUNTS, help="counts of noise"
    )
    ap.add_argument(
        "--truth", type=Path, help="also write the true parameters as JSON"
    )
    args = ap.parse_args(argv)

    data, params, poses = sim.make_session(seed=args.seed, noise_counts=args.noise)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["segment", "seq", "t_us", *geometry.CHANNEL_NAMES])
        for i, (segment, row) in enumerate(zip(data.segments, data.counts)):
            # 770 Hz to match the device's real cadence.
            writer.writerow([segment, i % 65536, int(i * 1e6 / 770), *row.astype(int)])

    print(f"Wrote {len(data.counts)} frames to {args.output}")
    if args.truth:
        params.save(args.truth)
        print(f"Wrote true parameters to {args.truth}")

    print("\nTrue magnet moments:")
    for j, m in enumerate(params.moments):
        print(f"  magnet{j + 1}  {np.round(m, 3)}  |m| = {np.linalg.norm(m):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
