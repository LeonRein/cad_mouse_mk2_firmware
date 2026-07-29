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

    Measured, not assumed: filtering the ``free`` segment gives peaks of 1.24,
    2.47 and 0.84 mm and 7.6, 8.5 and 10.2 degrees. An earlier guess of
    +-1.2 mm and +-4 deg was inferred from segment spans divided by the model's
    sensitivity, and it under-tested the rotations by more than a factor of
    two -- so the accuracy gates were being checked over a smaller envelope
    than the device actually visits.
    """
    rng = np.random.default_rng(20260729)
    translations = rng.uniform(-2.5, 2.5, size=(200, 3))
    rotations = np.deg2rad(rng.uniform(-11.0, 11.0, size=(200, 3)))
    return np.concatenate([translations, rotations], axis=1)
