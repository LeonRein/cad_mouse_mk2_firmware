#!/usr/bin/env python3
"""Watch the six HID axes the device is actually reporting.

    uv run hidmon.py

This sits at the far end of the whole chain — past the sensors, the filter,
the zeroing, the deadzone and the scaling — and shows what an application
would see. It exists for one question that nothing else in this repository can
answer: **are the signs right?**

Every other check in this project compares the device against itself or
against the Python. Both would be equally happy if two axes were swapped or an
axis pointed backwards, because both would be consistently wrong. The only
test for that is a person pushing the knob one way and reading which number
moves, which is what this prints.

    push the knob RIGHT   ->  X should go positive
    push it AWAY from you ->  Y should go positive
    press DOWN            ->  Z should go negative
    tilt, twist           ->  Rx, Ry, Rz

If one is backwards, flip its entry in `AXIS_SIGN` in
`crates/cadmouse-model/src/shaping.rs` and reflash. If two are swapped, the
board frame and the report order disagree and the fix belongs in the same
place.

Reads evdev directly rather than going through spacenavd, so it works before
the daemon is set up and shows the raw axis values rather than the daemon's
filtered and scaled ones.
"""

from __future__ import annotations

import argparse
import sys

import evdev

#: What the firmware enumerates as. See `src/main.rs`.
USB_VID = 0x1209
USB_PID = 0x0001

#: The six axes, in report order.
AXES = [
    ("X", evdev.ecodes.ABS_X),
    ("Y", evdev.ecodes.ABS_Y),
    ("Z", evdev.ecodes.ABS_Z),
    ("Rx", evdev.ecodes.ABS_RX),
    ("Ry", evdev.ecodes.ABS_RY),
    ("Rz", evdev.ecodes.ABS_RZ),
]

BUTTONS = [("btn1", evdev.ecodes.BTN_0), ("btn2", evdev.ecodes.BTN_1)]


def find_device(path: str | None) -> evdev.InputDevice:
    if path:
        return evdev.InputDevice(path)

    for candidate in evdev.list_devices():
        device = evdev.InputDevice(candidate)
        if device.info.vendor == USB_VID and device.info.product == USB_PID:
            return device
        device.close()

    raise FileNotFoundError(
        f"no device with USB ID {USB_VID:04x}:{USB_PID:04x} found. Is the "
        "firmware flashed and the cable plugged in? If reading /dev/input "
        "is denied, you are not in the 'input' group."
    )


def bar(value: int, limit: int = 350, width: int = 21) -> str:
    """A centred bar, so a sign error is visible without reading the number."""
    middle = width // 2
    cells = ["-"] * width
    cells[middle] = "|"
    offset = int(round(max(-1.0, min(1.0, value / limit)) * middle))
    cells[middle + offset] = "#"
    return "".join(cells)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--path", help="evdev node, default is autodetected")
    args = ap.parse_args(argv)

    try:
        device = find_device(args.path)
    except (FileNotFoundError, PermissionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"{device.name}  [{device.info.vendor:04x}:{device.info.product:04x}]  {device.path}")
    print("Push the knob and watch which axis moves. Ctrl-C to stop.\n")

    values = {code: 0 for _, code in AXES}
    buttons = {code: 0 for _, code in BUTTONS}

    try:
        for event in device.read_loop():
            if event.type == evdev.ecodes.EV_ABS:
                values[event.code] = event.value
            elif event.type == evdev.ecodes.EV_KEY and event.code in buttons:
                buttons[event.code] = event.value
            elif event.type == evdev.ecodes.EV_SYN:
                cells = "  ".join(
                    f"{name}{values[code]:+5d} {bar(values[code])}" for name, code in AXES
                )
                pressed = "".join(
                    f" [{name}]" for name, code in BUTTONS if buttons[code]
                )
                print(f"\r{cells}{pressed}   ", end="", flush=True)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
