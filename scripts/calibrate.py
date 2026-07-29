#!/usr/bin/env python3
"""Fit a device's calibration from a recorded session.

    uv run calibrate.py data/session1.csv -o calibration.json

The work lives in `cadmouse/calibrate.py`; this is here so the four steps of
the pipeline sit side by side and in order:

    record.py     capture a session from the device
    calibrate.py  fit the 27 calibration parameters to it   <-- you are here
    view.py       watch the result live and check the signs
    export.py     emit the table and calibration for the firmware
"""

from __future__ import annotations

from cadmouse.calibrate import main

if __name__ == "__main__":
    raise SystemExit(main())
