#!/usr/bin/env python3
"""How well can each DOF be resolved, given the geometry and the sensor noise?

This is a property of the hardware, not of the estimator: it is the Cramer-Rao
bound on pose from a single frame. It answers a question the C++ firmware's
author raised but could not settle -- whether the axis bleed he documented in
``misc/original_firmware/README.md`` was an artefact of his linear mix or a
physical limit of the magnet layout.

    uv run observability.py                    # nominal geometry
    uv run observability.py --calib calib.json # as actually built

Read the output as: *no* estimator, however clever, beats the per-DOF sigma from
one frame. A filter improves on it only by averaging over time.
"""

from __future__ import annotations

import argparse

import numpy as np

from cadmouse import forward, geometry
from cadmouse.params import CalibParams

#: Plausible per-channel noise in counts when no recording is available. The
#: measured value from a session's rest segments is better -- pass --calib.
DEFAULT_SIGMA_COUNTS = 4.0


def fisher_covariance(
    pose: np.ndarray, params: CalibParams, sigma: np.ndarray
) -> np.ndarray:
    """Pose covariance from one frame: ``(J^T R^-1 J)^-1``.

    Singular Jacobians are reported rather than hidden -- an unobservable
    direction is the single most important thing this script can find.
    """
    jac = forward.pose_jacobian(pose, params)
    weighted = jac / sigma[:, None]
    fisher = weighted.T @ weighted
    return np.linalg.pinv(fisher)


def resolution(pose: np.ndarray, params: CalibParams, sigma: np.ndarray) -> np.ndarray:
    """Per-DOF 1-sigma resolution: millimetres for t, degrees for r."""
    std = np.sqrt(np.diag(fisher_covariance(pose, params, sigma)))
    return np.concatenate([std[:3], np.rad2deg(std[3:])])


def workspace_poses(n: int = 5) -> np.ndarray:
    """A sweep along each DOF, to check resolution is not only good at rest."""
    travel = np.array([2.5, 2.5, 2.0, 0.12, 0.12, 0.15])
    poses = [np.zeros(6)]
    for k in range(6):
        for a in np.linspace(-1.0, 1.0, n):
            pose = np.zeros(6)
            pose[k] = a * travel[k]
            poses.append(pose)
    return np.array(poses)


def report(params: CalibParams, sigma: np.ndarray) -> None:
    units = ["mm", "mm", "mm", "deg", "deg", "deg"]

    jac = forward.pose_jacobian(np.zeros(6), params)
    weighted = jac / sigma[:, None]
    svals = np.linalg.svd(weighted, compute_uv=False)

    print("At the rest pose")
    print("  singular values of the noise-weighted Jacobian:")
    print("   ", np.array2string(svals, precision=3, suppress_small=False))
    print(f"  condition number: {svals[0] / svals[-1]:.1f}")
    if svals[0] / svals[-1] > 100:
        print(
            "  -> ill-conditioned: the weakest direction is over 100x harder to\n"
            "     see than the strongest, so expect it to be noisy and to bleed."
        )

    res = resolution(np.zeros(6), params, sigma)
    print("\n  per-DOF 1-sigma resolution from a single frame:")
    for i, name in enumerate(geometry.DOF_NAMES):
        print(f"    {name}  {res[i]:9.4f} {units[i]}")

    worst = int(np.argmax(res[:3] / res[:3].min()))
    worst_rot = 3 + int(np.argmax(res[3:] / res[3:].min()))
    print(
        f"\n  weakest translation axis: {geometry.DOF_NAMES[worst]}"
        f"   weakest rotation axis: {geometry.DOF_NAMES[worst_rot]}"
    )

    print("\nAcross the workspace (worst case over a sweep of each DOF)")
    across = np.array([resolution(p, params, sigma) for p in workspace_poses()])
    for i, name in enumerate(geometry.DOF_NAMES):
        print(
            f"    {name}  best {across[:, i].min():8.4f}  "
            f"worst {across[:, i].max():8.4f} {units[i]}"
        )

    print(
        "\nNote: this is a single-frame bound. A filter averaging over ~770 Hz\n"
        "of samples improves it by roughly sqrt(N) for motion slower than its\n"
        "bandwidth -- so a 0.01 mm single-frame sigma is not the usable limit."
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calib", help="calib.json from fit.py (default: nominal)")
    ap.add_argument(
        "--sigma",
        type=float,
        default=DEFAULT_SIGMA_COUNTS,
        help="per-channel noise in counts, if not taken from --calib",
    )
    args = ap.parse_args(argv)

    if args.calib:
        params = CalibParams.load(args.calib)
        print(f"Using calibration from {args.calib}")
    else:
        # Nominal build: identity orientation, equal axial magnets. A gain of
        # 6e4 counts per model unit is the order the real hardware lands on.
        params = CalibParams.initial(gain_counts_per_unit=6.0e4)
        print("Using nominal geometry (no --calib given)")

    sigma = np.full(geometry.N_CHANNELS, args.sigma)
    print(f"Assuming {args.sigma:.2f} counts of noise per channel\n")
    report(params, sigma)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
