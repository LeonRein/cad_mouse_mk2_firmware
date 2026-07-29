"""Calibration parameters: what the fit is allowed to move, and what it is not.

The split between these two blocks is the load-bearing design decision of the
whole estimator, so it is worth stating plainly.

**Geometry** -- magnet positions, magnetisation tilts, magnet moments. These are
properties of the mechanism. Changing one changes the entire measurement
Jacobian, so a geometry fitted on the recorded motions stays right at poses
that were never recorded.

**Sensor offset** -- one additive constant per channel. This is a property of
the silicon: a pure output-side map that no pose can affect.

Nothing else is fitted, and in particular there is no free matrix mixing the
two. Such a matrix would happily absorb geometry error over the region that was
sampled and then be wrong everywhere else, which is exactly the failure the
held-out ``free`` segment exists to catch.

Held fixed, and why:

* Sensor positions -- pick-and-place coordinates, and the datum that breaks the
  rigid-body gauge freedom.
* Magnet diameter and length -- drawing values, not separable from the moment
  at this standoff.
Not modelled at all: **per-channel sensor gain**. It was, once, and the reason
it is gone is worth keeping rather than rediscovering.

The nine gains are very nearly redundant with the three moments. Scale every
moment by ``a`` and every gain by ``1/a`` and no measurement changes at all --
an exact degeneracy -- and the eight remaining directions are only weakly
determined, trading against magnet tilt and position. Meanwhile the datasheet's
gain tolerance is far tighter than the magnets' remanence spread, so the
amplitude error worth chasing belongs in the moment, which is where the fit puts
it.

Enabling the gains was tried and it overfits: the held-out residual moves the
wrong way, 0.886 to 1.001 counts, while the gains themselves wander 6-16 % from
unity, far outside anything the part could plausibly do. Nine free parameters
buying a worse prediction is the textbook signature. They were pinned at 1.0 for
a while and then removed outright, because a parameter that is always exactly 1
is a multiply on the firmware's hot path pretending to be a degree of freedom.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .geometry import MAGNET_POS, MOMENT_N35

#: Order of the packed parameter blocks, and how many scalars each contributes.
_BLOCKS = (
    ("magnet_pos", 9),
    ("magnet_tilt", 6),
    ("magnet_moment", 3),
    ("sensor_offset", 9),
)

#: Size of the packed parameter vector. Every one of them is fitted.
N_PARAMS = sum(size for _, size in _BLOCKS)

#: Characteristic size of each block, in its own units. Handed to the optimiser
#: as ``x_scale`` so a millimetre of magnet position and a count of sensor
#: offset take comparable steps; without it the moments, which are ~0.1 in SI,
#: would be effectively frozen next to offsets of tens of counts.
_SCALES = {
    "magnet_pos": 0.5,  # mm
    "magnet_tilt": np.deg2rad(3.0),  # rad
    "magnet_moment": 0.2 * MOMENT_N35,  # A*m^2
    "sensor_offset": 20.0,  # counts (~1.3 mT)
}


@dataclass
class CalibParams:
    """One device's calibration.

    ``magnet_tilt`` holds two angles per magnet describing how its
    magnetisation axis leans off the body ``+z``. A reversed magnet is carried
    by a *negative moment* rather than a flipped axis, which keeps the field
    linear in the moment and keeps the tilt angles small for every magnet.
    """

    magnet_pos: np.ndarray  # (3, 3) mm, knob body frame
    magnet_tilt: np.ndarray  # (3, 2) rad, (about body x, about body y)
    magnet_moment: np.ndarray  # (3,) A*m^2, signed
    sensor_offset: np.ndarray  # (3, 3) counts

    @staticmethod
    def nominal() -> "CalibParams":
        """Starting point from the drawing, before any data is seen.

        Good enough to initialise from: nominal geometry already reproduces the
        direction of the measured response to within a cosine of 0.89-0.99 per
        degree of freedom, which is well inside the basin of attraction. It is
        *not* good enough to use, being some 25 % off in amplitude and 40 % off
        for the third magnet.

        The moments come out **all positive**, which is the drawing's intent and
        not necessarily the device's. Polarity is a per-device fact; see
        :func:`~cadmouse.calibrate.initial_params`, which is what the fit
        actually starts from.
        """
        return CalibParams(
            magnet_pos=MAGNET_POS.copy(),
            magnet_tilt=np.zeros((3, 2)),
            magnet_moment=np.full(3, MOMENT_N35),
            sensor_offset=np.zeros((3, 3)),
        )

    # ---------------------------------------------------------------- axes

    def magnet_axes(self) -> np.ndarray:
        """(3, 3) unit magnetisation directions in the knob body frame.

        Built so that zero tilt gives exactly ``+z`` and the result is unit
        norm for any angles, rather than only for small ones.
        """
        tx = self.magnet_tilt[:, 0]
        ty = self.magnet_tilt[:, 1]
        return np.stack(
            [np.sin(ty), -np.sin(tx) * np.cos(ty), np.cos(tx) * np.cos(ty)], axis=1
        )

    def magnet_axes_jacobian(self) -> np.ndarray:
        """``d(axis)/d(tilt)``, shape (3 magnets, 3 components, 2 angles)."""
        tx = self.magnet_tilt[:, 0]
        ty = self.magnet_tilt[:, 1]
        d_tx = np.stack(
            [np.zeros(3), -np.cos(tx) * np.cos(ty), -np.sin(tx) * np.cos(ty)], axis=1
        )
        d_ty = np.stack(
            [np.cos(ty), np.sin(tx) * np.sin(ty), -np.cos(tx) * np.sin(ty)], axis=1
        )
        return np.stack([d_tx, d_ty], axis=2)

    # ---------------------------------------------------------------- packing

    def pack(self) -> np.ndarray:
        """Flatten to the optimiser's parameter vector."""
        parts = [np.asarray(getattr(self, name), float).ravel() for name, _ in _BLOCKS]
        return np.concatenate(parts)

    @staticmethod
    def unpack(vector: np.ndarray) -> "CalibParams":
        vector = np.asarray(vector, float)
        out = {}
        at = 0
        for name, size in _BLOCKS:
            out[name] = vector[at : at + size]
            at += size
        if at != vector.size:
            raise ValueError(f"expected {at} parameters, got {vector.size}")
        return CalibParams(
            magnet_pos=out["magnet_pos"].reshape(3, 3),
            magnet_tilt=out["magnet_tilt"].reshape(3, 2),
            magnet_moment=out["magnet_moment"].copy(),
            sensor_offset=out["sensor_offset"].reshape(3, 3),
        )

    @staticmethod
    def scales() -> np.ndarray:
        """Per-parameter step scale, aligned with :meth:`pack`."""
        return np.concatenate([np.full(size, _SCALES[name]) for name, size in _BLOCKS])

    # ---------------------------------------------------------------- io

    def to_dict(self) -> dict:
        return {
            "magnet_pos_mm": self.magnet_pos.tolist(),
            "magnet_tilt_rad": self.magnet_tilt.tolist(),
            "magnet_moment_am2": self.magnet_moment.tolist(),
            "sensor_offset_counts": self.sensor_offset.tolist(),
        }

    @staticmethod
    def from_dict(d: dict) -> "CalibParams":
        return CalibParams(
            magnet_pos=np.array(d["magnet_pos_mm"], float),
            magnet_tilt=np.array(d["magnet_tilt_rad"], float),
            magnet_moment=np.array(d["magnet_moment_am2"], float),
            sensor_offset=np.array(d["sensor_offset_counts"], float),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @staticmethod
    def load(path: str | Path) -> "CalibParams":
        return CalibParams.from_dict(json.loads(Path(path).read_text()))

    def copy(self) -> "CalibParams":
        return replace(
            self,
            magnet_pos=self.magnet_pos.copy(),
            magnet_tilt=self.magnet_tilt.copy(),
            magnet_moment=self.magnet_moment.copy(),
            sensor_offset=self.sensor_offset.copy(),
        )
