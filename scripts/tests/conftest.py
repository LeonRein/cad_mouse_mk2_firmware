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
from cadmouse.calibrate import initial_params  # noqa: E402
from cadmouse.dataset import load_session  # noqa: E402
from cadmouse.params import CalibParams  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

#: Preferred recording, then whatever else is there.
#:
#: Not a hardcoded filename, because these gates check facts about the
#: *device* -- polarity, nominal amplitude, response direction -- and any
#: recording of it will do. Pinning one name meant the whole data-backed half
#: of the suite went quiet the moment that file was not where it used to be,
#: which is a silent skip rather than a failure and easy to miss.
def _find_session() -> Path | None:
    preferred = DATA_DIR / "session1.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(DATA_DIR.glob("*.csv")) if DATA_DIR.is_dir() else []
    return candidates[0] if candidates else None


DATA = _find_session()


@pytest.fixture(scope="session")
def table():
    return build_table()


@pytest.fixture(scope="session")
def nominal():
    """Geometry from the drawing, with no device in the room.

    Moments are all positive here, so this is the right fixture for the purely
    mechanical checks -- Jacobians against finite differences, affine offsets,
    observability -- and the wrong one for anything compared against a
    recording. Use :func:`device_nominal` for those.
    """
    return CalibParams.nominal()


@pytest.fixture(scope="session")
def device_nominal(session):
    """The same geometry, carrying the polarity of the recorded device.

    Which way each magnet points is measured, not designed, so a test that
    compares the model against ``session1.csv`` has to take the signs from
    ``session1.csv`` too. This is exactly what the fit starts from.
    """
    return initial_params(session)


@pytest.fixture(scope="session")
def session():
    if DATA is None:
        pytest.skip(f"no recorded session (*.csv) in {DATA_DIR}")
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
