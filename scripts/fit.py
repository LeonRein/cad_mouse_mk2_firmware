#!/usr/bin/env python3
"""Fit a calibration from a recorded session and report how well it fits.

    uv run fit.py data/session1.csv
    uv run fit.py data/session1.csv -o calib.json --plot

The residual report is the part to read, not the fact that it finished. A fit
whose RMS sits at the noise floor measured from the ``rest`` segments is good; a
fit several times above it means the model is wrong for this hardware, and the
residual-versus-pose plot says in which direction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cadmouse import calibrate, forward, geometry


def report(result: calibrate.CalibrationResult, data: calibrate.CalibrationData) -> None:
    sigma = result.sigma
    print("\n--- residuals ---")
    print(f"  overall RMS   {result.rms():8.3f} counts")
    print(f"  noise floor   {sigma.mean():8.3f} counts (from rest segments)")
    print(f"  ratio         {result.rms() / sigma.mean():8.2f}x")

    per_channel = result.rms_per_channel()
    print("\n  per channel (RMS counts / noise floor):")
    for i, name in enumerate(geometry.CHANNEL_NAMES):
        flag = "  <-- high" if per_channel[i] > 3 * sigma[i] else ""
        print(f"    {name:7s} {per_channel[i]:8.2f} / {sigma[i]:5.2f}{flag}")

    print("\n--- recovered parameters ---")
    print(f"  magnet moments (||m_1|| pinned to 1 by gauge):")
    for j, m in enumerate(result.params.moments):
        tilt = np.rad2deg(np.arctan2(np.linalg.norm(m[:2]), abs(m[2])))
        print(
            f"    magnet{j + 1}  |m| = {np.linalg.norm(m):.3f}  "
            f"polarity {'+z' if m[2] > 0 else '-z'}  tilt {tilt:.1f} deg"
        )

    print("\n  per-sensor gain magnitude and anisotropy:")
    mags = result.params.gain_magnitudes
    aniso = result.params.gain_anisotropy()
    expected = geometry.expected_gain_counts_per_unit()
    for i in range(geometry.N_SENSORS):
        print(
            f"    mag{i + 1}  {mags[i]:12.1f} counts/unit  anisotropy {aniso[i]:6.1%}"
        )
    print(f"    physically predicted: {expected:12.1f} counts/unit")

    print("\n  mechanism cross-coupling (off-axis motion per commanded axis):")
    for k, c in enumerate(result.cross_coupling()):
        print(f"    {geometry.DOF_NAMES[k]}  {c:6.1%}")

    print("\n  sensor biases (counts):")
    for i, b in enumerate(result.params.biases):
        print(f"    mag{i + 1}  " + "  ".join(f"{v:8.1f}" for v in b))

    print("\n--- pose range explored ---")
    units = ["mm", "mm", "mm", "deg", "deg", "deg"]
    poses = result.poses.copy()
    poses[:, 3:] = np.rad2deg(poses[:, 3:])
    for k, name in enumerate(geometry.DOF_NAMES):
        print(
            f"    {name}  {poses[:, k].min():+7.2f} .. {poses[:, k].max():+7.2f} "
            f"{units[k]}"
        )


def evaluate_heldout(
    result: calibrate.CalibrationResult, data: calibrate.CalibrationData
) -> None:
    """Score the calibration on data it never saw.

    This is the number that says whether the model is right rather than merely
    flexible: the `free` segment was excluded from the fit entirely, so a low
    residual here cannot be overfitting.
    """
    from cadmouse import ukf

    heldout = data.heldout
    if not len(heldout.counts):
        print("\n(no 'free' segment recorded -- no held-out check possible)")
        return

    print(f"\n--- held-out '{calibrate.HELDOUT_SEGMENT}' segment ---")
    stride = max(1, len(heldout.counts) // 300)
    counts = heldout.counts[::stride]
    poses = ukf.solve_all(counts, result.params, result.model, sigma=result.sigma)
    residual = forward.predict(poses, result.params, result.model) - counts
    rms = float(np.sqrt((residual**2).mean()))
    print(f"  {len(counts)} samples, RMS {rms:.3f} counts")
    print(f"  ratio to noise floor: {rms / result.sigma.mean():.2f}x")
    if rms > 3 * result.sigma.mean():
        print("  -> the model does not generalise; do not trust the calibration.")


def plot(result: calibrate.CalibrationResult, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residual = result.residual_counts()
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for i, ax in enumerate(axes.ravel()):
        ax.scatter(result.poses[:, i % 6], residual[:, i], s=2, alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(geometry.CHANNEL_NAMES[i], fontsize=9)
        ax.set_xlabel(geometry.DOF_NAMES[i % 6], fontsize=8)
    fig.suptitle(
        "Residual vs pose -- structure here means model error, not noise"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"\nWrote {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("session", type=Path, help="CSV from record.py")
    ap.add_argument("-o", "--output", type=Path, default=Path("calib.json"))
    ap.add_argument("--samples", type=int, default=900, help="samples to fit")
    ap.add_argument("--plot", action="store_true", help="write a residual plot")
    ap.add_argument(
        "--magnet-offsets",
        action="store_true",
        help="also fit per-magnet position offsets (9 extra parameters). "
        "Only worth enabling if the residual plot shows structure the sensor "
        "calibration cannot absorb.",
    )
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    data, seq, _ = calibrate.load_session(args.session)
    print(f"Loaded {len(data.counts)} frames from {args.session}")

    gaps = int(np.sum(np.diff(seq.astype(np.int64)) - 1))
    if gaps > 0:
        print(f"  {gaps} frames dropped during recording ({gaps / len(seq):.2%})")

    try:
        result = calibrate.calibrate(
            data,
            n_fit=args.samples,
            fit_magnet_offsets=args.magnet_offsets,
            verbose=not args.quiet,
        )
    except ValueError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1

    report(result, data)
    evaluate_heldout(result, data)

    result.params.save(args.output)
    print(f"\nWrote {args.output}")

    if args.plot:
        plot(result, args.output.with_suffix(".residuals.png"))

    print(f"\nNext: uv run live.py --calib {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
