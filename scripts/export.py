#!/usr/bin/env python3
"""Emit the field table and a fitted calibration for the firmware to build.

    uv run export.py calibration.json

The work lives in `cadmouse/export.py`; this is here so the four steps of the
pipeline sit side by side and in order:

    record.py     capture a session from the device
    calibrate.py  fit the 27 calibration parameters to it
    view.py       watch the result live and check the signs
    export.py     emit the table and calibration for the firmware  <-- here
"""

from __future__ import annotations

from cadmouse.export import main

if __name__ == "__main__":
    raise SystemExit(main())
