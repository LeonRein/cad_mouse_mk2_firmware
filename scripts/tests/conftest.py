"""Shared fixtures.

The field table takes a few seconds to build, so it is session-scoped: every
test that touches the measurement function wants the same one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cadmouse import build_table  # noqa: E402
from cadmouse.dataset import load_session  # noqa: E402
from cadmouse.params import CalibParams  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "session1.csv"


@pytest.fixture(scope="session")
def table():
    return build_table()


@pytest.fixture(scope="session")
def nominal():
    return CalibParams.nominal()


@pytest.fixture(scope="session")
def session():
    if not DATA.exists():
        pytest.skip(f"no recorded session at {DATA}")
    return load_session(DATA)


@pytest.fixture(scope="session")
def envelope_poses():
    """Poses spanning what the mechanism actually reaches.

    Derived from the recorded spans divided by the model's sensitivity: about
    +-1.2 mm of travel and +-4 deg of tilt.
    """
    rng = np.random.default_rng(20260729)
    translations = rng.uniform(-1.2, 1.2, size=(200, 3))
    rotations = np.deg2rad(rng.uniform(-4.0, 4.0, size=(200, 3)))
    return np.concatenate([translations, rotations], axis=1)
