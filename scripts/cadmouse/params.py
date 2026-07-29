"""Calibration parameters: what the fit is allowed to move, and what it is not.

The split between these two blocks is the load-bearing design decision of the
whole estimator, so it is worth stating plainly.

**Geometry** -- magnet positions, magnetisation tilts, magnet moments. These are
properties of the mechanism. Changing one changes the entire measurement
Jacobian, so a geometry fitted on the recorded motions stays right at poses
that were never recorded.

**Affine sensor block** -- a per-sensor offset and optional per-axis gain. These
are properties of the silicon: a pure output-side map that no pose can affect.

Nothing else is fitted, and in particular there is no free matrix mixing the
two. Such a matrix would happily absorb geometry error over the region that was
sampled and then be wrong everywhere else, which is exactly the failure the
held-out ``free`` segment exists to catch.

Held fixed, and why:

* Sensor positions -- pick-and-place coordinates, and the datum that breaks the
  rigid-body gauge freedom.
* Magnet diameter and length -- drawing values, not separable from the moment
  at this standoff.
* Sensor gain, by default -- the datasheet's gain tolerance is far tighter than
  the magnet's remanence spread, so letting both float would only feed the
  global scale degeneracy (scale every moment by ``a``, every gain by ``1/a``,
  and no measurement changes).

  This was tried, and it overfits: enabling the nine gains moves the held-out
  residual the wrong way, 0.886 to 1.001 counts, while the gains themselves
  wander 6-16 % from unity -- far outside anything the part could plausibly do.
  Nine free parameters buying a worse prediction is the textbook signature, so
  the default stays off.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .geometry import MAGNET_POS, MOMENT_N35, MOMENT_SIGN

#: Order of the packed parameter blocks, and how many scalars each contributes.
_BLOCKS = (
    ("magnet_pos", 9),
    ("magnet_tilt", 6),
    ("magnet_moment", 3),
    ("sensor_offset", 9),
    ("sensor_gain", 9),
)

#: Characteristic size of each block, in its own units. Handed to the optimiser
#: as ``x_scale`` so a millimetre of magnet position and a count of sensor
#: offset take comparable steps; without it the moments, which are ~0.1 in SI,
#: would be effectively frozen next to offsets of tens of counts.
_SCALES = {
    "magnet_pos": 0.5,  # mm
    "magnet_tilt": np.deg2rad(3.0),  # rad
    "magnet_moment": 0.2 * MOMENT_N35,  # A*m^2
    "sensor_offset": 20.0,  # counts (~1.3 mT)
    "sensor_gain": 0.03,  # dimensionless
}


@dataclass
class CalibParams:
    """One device's calibration.

    ``magnet_tilt`` holds two angles per magnet describing how its
    magnetisation axis leans off the body ``+z``; the reversed third magnet is
    carried by a *negative moment* rather than a flipped axis, which keeps the
    field linear in the moment and keeps the tilt angles small for every
    magnet.
    """

    magnet_pos: np.ndarray  # (3, 3) mm, knob body frame
    magnet_tilt: np.ndarray  # (3, 2) rad, (about body x, about body y)
    magnet_moment: np.ndarray  # (3,) A*m^2, signed
    sensor_offset: np.ndarray  # (3, 3) counts
    sensor_gain: np.ndarray  # (3, 3) dimensionless
    fit_gain: bool = field(default=False)

    @staticmethod
    def nominal() -> "CalibParams":
        """Starting point from the drawing, before any data is seen.

        Good enough to initialise from: nominal geometry already reproduces the
        direction of the measured response to within a cosine of 0.89-0.99 per
        degree of freedom, which is well inside the basin of attraction. It is
        *not* good enough to use, being some 25 % off in amplitude and 40 % off
        for the third magnet.
        """
        return CalibParams(
            magnet_pos=MAGNET_POS.copy(),
            magnet_tilt=np.zeros((3, 2)),
            magnet_moment=MOMENT_N35 * MOMENT_SIGN.copy(),
            sensor_offset=np.zeros((3, 3)),
            sensor_gain=np.ones((3, 3)),
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
    def unpack(vector: np.ndarray, fit_gain: bool = False) -> "CalibParams":
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
            sensor_gain=out["sensor_gain"].reshape(3, 3),
            fit_gain=fit_gain,
        )

    @staticmethod
    def scales() -> np.ndarray:
        """Per-parameter step scale, aligned with :meth:`pack`."""
        return np.concatenate([np.full(size, _SCALES[name]) for name, size in _BLOCKS])

    @staticmethod
    def free_mask(fit_gain: bool = False) -> np.ndarray:
        """Boolean mask of parameters the fit may move.

        Gains are pinned unless explicitly enabled; see the module docstring on
        the global scale degeneracy.
        """
        mask = np.concatenate([np.full(size, True) for _, size in _BLOCKS])
        if not fit_gain:
            mask[-9:] = False
        return mask

    # ---------------------------------------------------------------- io

    def to_dict(self) -> dict:
        return {
            "magnet_pos_mm": self.magnet_pos.tolist(),
            "magnet_tilt_rad": self.magnet_tilt.tolist(),
            "magnet_moment_am2": self.magnet_moment.tolist(),
            "sensor_offset_counts": self.sensor_offset.tolist(),
            "sensor_gain": self.sensor_gain.tolist(),
            "fit_gain": self.fit_gain,
        }

    @staticmethod
    def from_dict(d: dict) -> "CalibParams":
        return CalibParams(
            magnet_pos=np.array(d["magnet_pos_mm"], float),
            magnet_tilt=np.array(d["magnet_tilt_rad"], float),
            magnet_moment=np.array(d["magnet_moment_am2"], float),
            sensor_offset=np.array(d["sensor_offset_counts"], float),
            sensor_gain=np.array(d["sensor_gain"], float),
            fit_gain=bool(d.get("fit_gain", False)),
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
            sensor_gain=self.sensor_gain.copy(),
        )
