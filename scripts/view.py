#!/usr/bin/env python3
"""Live 3-D view of the estimated knob pose.

    uv run view.py calibration.json                    # from the device
    uv run view.py calibration.json --replay data/session1.csv
    uv run view.py calibration.json --replay data/session1.csv --save shot.png

This exists to catch what none of the automated checks can. Every quantitative
test in this project is *self-consistent*: the held-out residual, the NIS, the
innovation whiteness all compare the model against itself, so they would pass
just as happily if the board frame were mirrored or two axes were swapped. The
one test that ties segment labels to axes takes an absolute value of the
cosine, which confirms the response lies along the predicted axis but not which
way it points.

For a 3-D mouse that is the bug that ships -- push right, cursor goes left --
and a person moving the knob while watching the screen finds it in seconds.

What this cannot do is measure accuracy. There is still no ground truth, and
six micrometres is not visible. It is a check on sign, handedness, axis
coupling, gross scale, stability and latency; the numbers elsewhere cover the
rest.

Self-contained apart from the `cadmouse` package and `record.py`, which it
imports for the wire format so there is only one decoder in the tree.
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cadmouse import CalibParams, build_table
from cadmouse.dataset import CHANNEL_NAMES
from cadmouse.filter import FilterConfig, IteratedEkf
from cadmouse.geometry import SENSOR_POS, rotation_from_rotvec
from cadmouse.model import POSE_DIM, solve_pose

# `record.py` sits next to this file rather than in the package, and owns the
# wire format. Imported here so there is exactly one decoder in the tree.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from record import STATUS_IN_DEADZONE, describe_status, read_frames  # noqa: E402

#: Seconds of history in the traces.
HISTORY_S = 6.0

#: Redraw rate. The filter runs on every frame regardless -- it sustains over
#: 2.5 kHz in Python against the device's 2 kHz -- so this only limits drawing.
DISPLAY_HZ = 30.0

#: Points kept per trace. Six seconds at the full 2 kHz would be 12 000 samples
#: per trace, drawn into a few hundred pixels; profiling put that at 76 ms a
#: frame against 42 ms for the same axes holding 400 points. The history is
#: therefore decimated on the way in rather than on the way out.
TRACE_POINTS = 600

#: Fixed axis limits, from the envelope the mechanism actually reaches
#: (2.5 mm and 11 degrees, measured by filtering the recorded free motion).
#: Fixed rather than autoscaled because rescaling an axis invalidates the
#: blitting background, which is where nearly all the frame time went.
TRANSLATION_LIMIT_MM = 3.0
ROTATION_LIMIT_DEG = 12.0

#: The knob moves about a millimetre and a few degrees, against a body 33 mm
#: across. At true scale that is invisible, so the 3-D view exaggerates by this
#: factor. The traces are always in real units; only the picture is amplified.
DEFAULT_GAIN = 8.0

#: Which way is which. Derived from the board frame (+x right, +y toward the
#: rear, +z up) and the right-hand rule, and worth writing out because reading
#: a sign convention off a rotating wireframe is exactly the error this script
#: is meant to catch.
#:
#: +rx: about +x, so +y rotates toward +z -- the rear lifts, the nose dips.
#: +ry: about +y, so +z rotates toward +x -- the top leans right.
#: +rz: about +z, so +x rotates toward +y -- anticlockwise seen from above.
AXIS_LABELS = [
    ("tx", "mm", "LEFT", "RIGHT"),
    ("ty", "mm", "FRONT", "BACK"),
    ("tz", "mm", "DOWN", "UP"),
    ("rx", "deg", "NOSE UP", "NOSE DOWN"),
    ("ry", "deg", "LEAN LEFT", "LEAN RIGHT"),
    ("rz", "deg", "TWIST CW", "TWIST CCW"),
]

#: Below this the reading is noise, and naming a direction would be misleading.
DEADBAND = np.array([0.02, 0.02, 0.02, *(np.deg2rad(0.1),) * 3])


@dataclass
class LiveState:
    """Shared between the reader thread and the drawing thread.

    The trace history is a fixed-size ring decimated on the way *in*. The
    obvious alternative -- keep every sample in a deque and thin it when
    drawing -- means rebuilding a 12 000 x 6 array on every frame while holding
    the lock, which both costs time and stalls the reader.
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    pose: np.ndarray = field(default_factory=lambda: np.zeros(POSE_DIM))
    times: deque = field(default_factory=lambda: deque(maxlen=TRACE_POINTS))
    history: deque = field(default_factory=lambda: deque(maxlen=TRACE_POINTS))
    device_history: deque = field(default_factory=lambda: deque(maxlen=TRACE_POINTS))
    innovation_rms: float = 0.0
    nis: float = 0.0
    frames: int = 0
    lost: int = 0
    rate_hz: float = 0.0
    running: bool = True
    error: str | None = None
    #: What the *device* said, when the stream carries it. The firmware runs
    #: its own port of this filter on core 1, so comparing the two live is the
    #: only end-to-end check of that port there is -- the golden vectors check
    #: it against recorded data, this checks it against the board.
    device_pose: np.ndarray | None = None
    device_status: int = 0
    device_gap_um: float = 0.0
    #: Frames where the knob was moving, so the gap above means something.
    device_compared: int = 0
    _next_sample_t: float = -1e18

    def push(
        self,
        t: float,
        pose: np.ndarray,
        innovation: np.ndarray,
        nis: float,
        device: tuple[np.ndarray, int] | None = None,
    ) -> None:
        # Cheap running statistics on every frame; history only at the rate the
        # display can actually show.
        alpha = 0.002
        rms = float(np.sqrt((innovation**2).mean()))
        with self.lock:
            self.pose = pose.copy()
            self.innovation_rms += alpha * (rms - self.innovation_rms)
            self.nis += alpha * (nis - self.nis)
            self.frames += 1
            if device is not None:
                device_pose, status = device
                self.device_pose = device_pose
                self.device_status = status
                # Only while the knob is actually moving. The device reports a
                # pose that has been through its deadzone, so at rest it sends
                # exact zeros while the host's estimate still wanders by a few
                # micrometres -- comparing those two would show a permanent
                # ~10 um "disagreement" that is nothing of the kind. The
                # device's own IN_DEADZONE bit says when the comparison is
                # meaningful.
                if not status & STATUS_IN_DEADZONE:
                    gap = float(np.linalg.norm(pose[:3] - device_pose[:3])) * 1000.0
                    self.device_gap_um += alpha * (gap - self.device_gap_um)
                    self.device_compared += 1
            if t >= self._next_sample_t:
                self._next_sample_t = t + HISTORY_S / TRACE_POINTS
                self.times.append(t)
                self.history.append(pose.copy())
                # Kept in lockstep with `history` so the two traces share an
                # x-axis; NaN where the device said nothing, which matplotlib
                # draws as a gap rather than as a line to zero.
                self.device_history.append(
                    self.device_pose.copy()
                    if self.device_pose is not None
                    else np.full(POSE_DIM, np.nan)
                )

    def snapshot(self):
        with self.lock:
            if not self.times:
                return None
            return (
                self.pose.copy(),
                np.fromiter(self.times, dtype=float, count=len(self.times)),
                np.array(self.history),
                self.innovation_rms,
                self.nis,
                self.frames,
                self.lost,
                self.rate_hz,
                None if self.device_pose is None else self.device_pose.copy(),
                self.device_status,
                self.device_gap_um,
                self.device_compared,
                np.array(self.device_history),
            )


# --------------------------------------------------------------------------
# Frame sources
# --------------------------------------------------------------------------


#: CSV columns a recording carries if it was made against firmware that
#: reports its own estimate.
DEVICE_COLUMNS = ["dev_x", "dev_y", "dev_z", "dev_rx", "dev_ry", "dev_rz"]


def device_frames(port: str | None):
    """(t_seconds, counts, lost, device) from the hardware.

    Uses record.py's decoder, so there is only one description of the wire
    format in the tree.
    """
    previous_seq = None
    lost = 0
    for frame, _stats in read_frames(port=port):
        if previous_seq is not None:
            gap = (frame.seq - previous_seq) & 0xFFFF
            lost += max(0, gap - 1)
        previous_seq = frame.seq
        device = (np.array(frame.pose, dtype=float), frame.status)
        yield frame.t_us / 1e6, np.array(frame.counts, dtype=float), lost, device


def replay_frames(path: Path, speed: float = 1.0):
    """(t_seconds, counts, lost, device) from a recording, paced to wall clock.

    Pacing matters: a viewer that keeps up when fed as fast as the disk allows
    tells you nothing about whether it keeps up with the device.

    Recordings made before the firmware reported its own pose have no device
    columns; those replay with `device` as `None` and the comparison simply
    does not appear.
    """
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    has_device = all(c in rows[0] for c in DEVICE_COLUMNS)
    start_wall = time.perf_counter()
    t0 = int(rows[0]["t_us"]) / 1e6
    for row in rows:
        t = int(row["t_us"]) / 1e6
        if speed > 0:
            behind = (t - t0) / speed - (time.perf_counter() - start_wall)
            if behind > 0:
                time.sleep(behind)
        device = None
        if has_device:
            pose = np.array([float(row[c]) for c in DEVICE_COLUMNS])
            device = (pose, int(float(row.get("dev_status", 0))))
        yield t, np.array([float(row[c]) for c in CHANNEL_NAMES]), 0, device


def reader(state: LiveState, source, params: CalibParams, table, sigma, config) -> None:
    estimator = None
    last_t = None
    tick = time.perf_counter()
    counted = 0
    try:
        for t, counts, lost, device in source:
            if estimator is None:
                first, _ = solve_pose(counts, params, table, sigma=sigma)
                estimator = IteratedEkf(params, table, sigma, config, initial_pose=first)
            if last_t is not None:
                dt = t - last_t
                if 0.0 < dt < 1.0:
                    estimator.predict(dt)
            last_t = t
            estimator.update(counts)

            nis = float(
                estimator.innovation
                @ np.linalg.solve(estimator.innovation_cov, estimator.innovation)
            )
            state.push(t, estimator.x, estimator.innovation, nis, device)
            state.lost = lost

            counted += 1
            now = time.perf_counter()
            if now - tick >= 0.5:
                state.rate_hz = counted / (now - tick)
                counted, tick = 0, now
            if not state.running:
                return
    except Exception as exc:  # surfaced in the window rather than a dead thread
        state.error = f"{type(exc).__name__}: {exc}"
    finally:
        state.running = False


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def knob_wireframe():
    """Line segments of a simple knob body, in the body frame, millimetres.

    Two rings joined by struts, plus a nose marker so yaw is readable -- a
    rotationally symmetric shape would make `rz` invisible, which is the one
    axis a naive wireframe hides.
    """
    angles = np.linspace(0, 2 * np.pi, 49)
    radius = 18.0
    top = np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, 6.0)], 1)
    bottom = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.full_like(angles, -10.0)], 1
    )
    segments = [top, bottom]
    for a in np.linspace(0, 2 * np.pi, 9)[:-1]:
        p = np.array([radius * np.cos(a), radius * np.sin(a)])
        segments.append(np.array([[p[0], p[1], 6.0], [p[0], p[1], -10.0]]))
    # Nose: an arrow toward the front (-y), where MAG1 sits. Drawn last and
    # picked out in colour because a rotationally symmetric wireframe makes rz
    # invisible, and rz is one of the axes whose sign most needs checking.
    tip = np.array([0.0, -radius - 10.0, 6.0])
    segments.append(np.array([[0.0, -radius, 6.0], tip]))
    segments.append(np.array([[-5.0, -radius - 4.0, 6.0], tip, [5.0, -radius - 4.0, 6.0]]))
    return segments


#: Segments drawn in the accent colour: the nose arrow, appended last.
N_NOSE_SEGMENTS = 2


def build_figure(gain: float):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(13, 7.5))
    fig.canvas.manager.set_window_title("CAD Mouse MK2 - live pose")
    grid = fig.add_gridspec(6, 3, width_ratios=[1.5, 1.0, 1.0], hspace=0.55, wspace=0.3)

    ax3d = fig.add_subplot(grid[0:5, 0], projection="3d")
    ax3d.set_title(
        f"estimated pose  (motion exaggerated {gain:.0f}x)\n"
        "solid = host filter,  dashed = device filter",
        fontsize=10,
    )
    for setter, label in ((ax3d.set_xlabel, "x  right"), (ax3d.set_ylabel, "y  rear"), (ax3d.set_zlabel, "z  up")):
        setter(label, fontsize=8)
    span = 30.0
    ax3d.set_xlim(-span, span)
    ax3d.set_ylim(-span, span)
    ax3d.set_zlim(-span, span)
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.tick_params(labelsize=6)

    # Static reference: the sensor ring, which does not move with the knob.
    ax3d.scatter(
        SENSOR_POS[:, 0], SENSOR_POS[:, 1], SENSOR_POS[:, 2], c="tab:red", s=28, depthshade=False
    )
    for k, (x, y, z) in enumerate(SENSOR_POS):
        ax3d.text(x, y, z - 4, f"MAG{k + 1}", fontsize=7, color="tab:red", ha="center")

    body = knob_wireframe()
    lines = []
    for k in range(len(body)):
        nose = k >= len(body) - N_NOSE_SEGMENTS
        lines.append(
            ax3d.plot(
                [], [], [], lw=2.4 if nose else 1.0, color="tab:orange" if nose else "tab:blue"
            )[0]
        )

    # The same knob again, drawn from the *device's* pose.
    #
    # Dashed and on top, not a solid ghost underneath, and the reason is worth
    # stating: where the port is right the two poses are identical to within a
    # micrometre, so a solid overlay would be perfectly hidden by the host's
    # wireframe and there would be no way to tell "agrees" from "not drawn at
    # all". Dashes ride visibly on the solid line when they coincide, and
    # separate into two knobs when they do not.
    device_body_lines = []
    for _ in range(len(body)):
        device_body_lines.append(
            ax3d.plot([], [], [], lw=1.3, color="0.1", alpha=0.9, linestyle=(0, (4, 4)))[0]
        )

    # Two axes, not six. Profiling put the cost of a set of axis furniture --
    # ticks, grid, spines -- at about 7 ms a redraw regardless of how much data
    # it holds, so six of them cost 42 ms before a single sample was drawn.
    trace_axes = []
    trace_lines = []
    device_lines = []
    for block, (title, limit, names) in enumerate(
        [
            ("translation, mm", TRANSLATION_LIMIT_MM, AXIS_LABELS[:3]),
            ("rotation, deg", ROTATION_LIMIT_DEG, AXIS_LABELS[3:]),
        ]
    ):
        ax = fig.add_subplot(grid[3 * block : 3 * block + 3, 1])
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.axhline(0.0, color="0.6", lw=0.6)
        ax.set_xlim(-HISTORY_S, 0.05)
        ax.set_ylim(-limit, limit)
        if block == 1:
            ax.set_xlabel("seconds", fontsize=8)
        for name, *_ in names:
            trace_lines.append(ax.plot([], [], lw=1.2, label=name)[0])
        # The device's own estimate, drawn in black over the top of the host's.
        # One line per block rather than per axis: three more colours would
        # make the plot unreadable, and the question this answers is "do the
        # two agree", for which one visibly-different overlay per block is
        # enough. Where they agree it hides the host trace exactly.
        for k in range(3):
            device_lines.append(
                ax.plot(
                    [],
                    [],
                    lw=0.9,
                    color="black",
                    alpha=0.55,
                    label="device" if k == 0 else None,
                )[0]
            )
        ax.legend(loc="upper left", fontsize=7, ncols=4, framealpha=0.8)
        trace_axes.append(ax)

    ax_text = fig.add_subplot(grid[:, 2])
    ax_text.axis("off")
    readout = ax_text.text(
        0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=10, transform=ax_text.transAxes
    )
    return fig, ax3d, lines, device_body_lines, trace_axes, trace_lines, device_lines, readout


def format_readout(
    pose,
    innovation_rms,
    nis,
    frames,
    lost,
    rate_hz,
    draw_hz,
    error,
    device_pose=None,
    device_status=0,
    device_gap_um=0.0,
    device_compared=0,
):
    rows = ["MOVE THE KNOB AND CHECK", "each line names what it thinks", "you are doing.", ""]
    for k, (name, unit, negative, positive) in enumerate(AXIS_LABELS):
        value = pose[k] if k < 3 else np.rad2deg(pose[k])
        if abs(pose[k]) < DEADBAND[k]:
            word = "-"
        else:
            word = positive if pose[k] > 0 else negative
        rows.append(f"{name:>3} {value:+8.3f} {unit:<4} {word}")
    rows += [
        "",
        "FILTER (host)",
        f"  innovation {innovation_rms:6.2f} counts",
        f"  NIS        {nis:6.2f}  (target 9)",
    ]
    if device_pose is not None:
        gap = (
            f"  gap        {device_gap_um:6.1f} um vs host"
            if device_compared
            else "  gap        -- move the knob"
        )
        rows += [
            "",
            "FILTER (device)",
            gap,
            f"  {describe_status(device_status)}",
        ]
    rows += [
        "",
        "LINK",
        f"  frames     {frames:>8d}",
        f"  lost       {lost:>8d}",
        f"  filter     {rate_hz:7.0f} Hz",
        f"  display    {draw_hz:7.1f} fps",
    ]
    if error:
        rows += ["", "ERROR", f"  {error}"]
    return "\n".join(rows)


def run(
    state: LiveState, gain: float, save: Path | None, frames_to_draw: int, blit: bool = True
) -> int:
    import matplotlib

    if save is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    (
        fig,
        ax3d,
        lines,
        device_body_lines,
        trace_axes,
        trace_lines,
        device_lines,
        readout,
    ) = build_figure(gain)
    body = knob_wireframe()
    draw_times: deque = deque(maxlen=30)

    def draw_hz() -> float:
        """Measured redraw rate, reported next to the filter's own rate.

        Worth showing: the two are unrelated -- the filter runs on every frame
        at over 2 kHz while the display manages tens of frames a second -- and
        without both numbers on screen a slow *picture* looks like a slow
        *filter*.
        """
        if len(draw_times) < 2:
            return 0.0
        span = draw_times[-1] - draw_times[0]
        return (len(draw_times) - 1) / span if span > 0 else 0.0

    def update(_):
        draw_times.append(time.perf_counter())
        snap = state.snapshot()
        if snap is None:
            readout.set_text(
                format_readout(np.zeros(6), 0, 0, 0, 0, 0, 0, state.error or "waiting...")
            )
            return [*lines, *device_body_lines, *trace_lines, *device_lines, readout]

        (
            pose,
            times,
            history,
            innovation_rms,
            nis,
            frames,
            lost,
            rate_hz,
            device_pose,
            device_status,
            device_gap_um,
            device_compared,
            device_history,
        ) = snap

        def draw_body(target_lines, source_pose):
            shown = source_pose * gain
            rotation = rotation_from_rotvec(shown[3:])
            for line, segment in zip(target_lines, body):
                moved = segment @ rotation.T + shown[:3]
                line.set_data(moved[:, 0], moved[:, 1])
                line.set_3d_properties(moved[:, 2])

        draw_body(lines, pose)

        if device_pose is not None:
            draw_body(device_body_lines, device_pose)
        else:
            # No device in the stream -- an old recording, say. Leave the ghost
            # off rather than parked at zero, where it would read as "the
            # device says the knob is centred".
            for line in device_body_lines:
                line.set_data([], [])
                line.set_3d_properties([])

        # Limits are fixed at construction, so nothing here invalidates the
        # blitting background.
        t = times - times[-1]
        for k, line in enumerate(trace_lines):
            values = history[:, k]
            line.set_data(t, np.rad2deg(values) if k >= 3 else values)

        # The device's estimate over the top. Where the port is right these lie
        # exactly on the host traces and are invisible except at rest, where
        # the device's deadzone pins them to zero and the host's does not --
        # which is itself the deadzone being visible, and worth seeing.
        for k, line in enumerate(device_lines):
            values = device_history[:, k]
            line.set_data(t, np.rad2deg(values) if k >= 3 else values)

        readout.set_text(
            format_readout(
                pose,
                innovation_rms,
                nis,
                frames,
                lost,
                rate_hz,
                draw_hz(),
                state.error,
                device_pose,
                device_status,
                device_gap_um,
                device_compared,
            )
        )
        return [*lines, *device_body_lines, *trace_lines, *device_lines, readout]

    # Blitting is what makes this usable: redrawing everything costs 66 ms a
    # frame (15 fps), while blitting the artists over a cached background costs
    # 20 ms (49 fps). Of that 20 ms the 3-D wireframe is 1.2 ms and the text
    # readout is the single most expensive artist.
    anim = FuncAnimation(
        fig, update, interval=int(1000 / DISPLAY_HZ), blit=blit, cache_frame_data=False
    )
    if save is not None:
        # Let the reader reach the requested frame before drawing, with a
        # ceiling so a stalled source cannot hang the run.
        deadline = time.perf_counter() + 180.0
        while time.perf_counter() < deadline and state.frames < frames_to_draw:
            if not state.running:
                break
            time.sleep(0.2)
        update(0)
        fig.savefig(save, dpi=110)
        print(f"wrote {save} after {state.frames} frames")
    else:
        plt.show()
    del anim
    state.running = False
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--replay", type=Path, help="feed a recorded CSV instead of the device")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed multiplier")
    parser.add_argument("--port", help="serial device, default is autodetected")
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN, help="3-D exaggeration")
    parser.add_argument("--save", type=Path, help="render one frame headless and exit")
    parser.add_argument("--warmup", type=int, default=3000, help="frames to buffer before --save")
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument(
        "--no-blit",
        action="store_true",
        help="redraw everything each frame; slower, but a fallback if the "
        "blitted 3-D view leaves artefacts on your backend",
    )
    args = parser.parse_args(argv)

    params = CalibParams.load(args.calibration)
    table = build_table()
    config = FilterConfig(iterations=args.iterations)

    if args.replay:
        source = replay_frames(args.replay, args.speed)
        # The recording's own rest blocks are the best noise estimate available.
        from cadmouse.dataset import load_session

        sigma = load_session(args.replay).noise_sigma()
    else:
        source = device_frames(args.port)
        sigma = np.full(9, 1.1)  # measured rest noise; refined by a recording

    state = LiveState()
    thread = threading.Thread(
        target=reader, args=(state, source, params, table, sigma, config), daemon=True
    )
    thread.start()
    try:
        return run(state, args.gain, args.save, args.warmup, blit=not args.no_blit)
    finally:
        state.running = False


if __name__ == "__main__":
    raise SystemExit(main())
