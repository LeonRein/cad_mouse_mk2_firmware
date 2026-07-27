"""Calibration parameters: the container, its flat-vector packing, and JSON I/O.

45 free parameters with magnet position offsets disabled (the default):

===================  =====  =====================================================
group                count  note
===================  =====  =====================================================
``gains`` A_i          27   full 3x3 per sensor: solder orientation, per-axis
                            gain, and cross-axis sensitivity, all at once
``biases`` b_i          9   die offset + ambient field (Earth, nearby steel)
``moments`` m_j        27*  direction *and* magnitude, free and independent per
                            magnet -- both vary between units
``magnet_offsets``      9   optional, off by default
===================  =====  =====================================================

\\* 9, not 27 -- three magnets x a 3-vector each.

**Gauge.** Scaling every moment by *s* and every gain by 1/*s* leaves the
prediction unchanged, so the fit pins ||m_1|| = 1 (see `calibrate.gauge_residual`)
and lets the gains carry the absolute scale. The *pose* is still metric, because
the ring radii and the axial gap in `geometry` are known.

**Two gain modes, and why.** A free 3x3 gain matrix can represent *any*
orientation, so with ``gain_mode="full"`` the discrete orientation search is
vacuous -- every candidate can converge to the same answer, and the search just
burns time without discriminating. ``gain_mode="scalar"`` restricts each sensor
to a single scale factor on a *fixed* shared orientation (21 free parameters
instead of 45), which makes the orientation genuinely discrete and the search
meaningful. `calibrate` searches in scalar mode, then calls `relax` to release
the full matrices for the final polish.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import geometry

_N_GAIN = geometry.N_SENSORS * 9
_N_SCALE = geometry.N_SENSORS
_N_BIAS = geometry.N_SENSORS * 3
_N_MOMENT = geometry.N_MAGNETS * 3
_N_OFFSET = geometry.N_MAGNETS * 3

GAIN_FULL = "full"
GAIN_SCALAR = "scalar"


@dataclass
class CalibParams:
    """Sensor and magnet calibration.

    Attributes:
        gains: (3, 3, 3) -- ``gains[i]`` maps board-frame field in model units to
            sensor *i*'s raw counts. Always the *effective* matrices, whichever
            gain mode is in force.
        biases: (3, 3) -- additive per-sensor offset in counts.
        moments: (3, 3) -- magnet moment vectors at the rest pose, board frame.
        magnet_offsets: (3, 3) -- correction to the nominal magnet positions.
        fit_magnet_offsets: whether `magnet_offsets` is a free parameter.
        gain_mode: ``"full"`` (27 free gain parameters) or ``"scalar"``
            (3, one scale per sensor on the fixed shared `orientation`).
        orientation: (3, 3) -- the shared board->sensor signed permutation. Only
            constrains the fit in ``"scalar"`` mode; otherwise it is kept as a
            record of where the discrete search landed.
        magnet_positions: (3, 3) -- nominal rest positions, held fixed.
        sensor_positions: (3, 3) -- board-frame sensor positions, held fixed.
    """

    gains: np.ndarray
    biases: np.ndarray
    moments: np.ndarray
    magnet_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros((geometry.N_MAGNETS, 3))
    )
    fit_magnet_offsets: bool = False
    gain_mode: str = GAIN_FULL
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    magnet_positions: np.ndarray = field(default_factory=geometry.magnet_positions)
    sensor_positions: np.ndarray = field(default_factory=geometry.sensor_positions)

    def __post_init__(self) -> None:
        self.gains = np.asarray(self.gains, dtype=float).reshape(geometry.N_SENSORS, 3, 3)
        self.biases = np.asarray(self.biases, dtype=float).reshape(geometry.N_SENSORS, 3)
        self.moments = np.asarray(self.moments, dtype=float).reshape(geometry.N_MAGNETS, 3)
        self.magnet_offsets = np.asarray(self.magnet_offsets, dtype=float).reshape(
            geometry.N_MAGNETS, 3
        )
        self.orientation = np.asarray(self.orientation, dtype=float).reshape(3, 3)
        if self.gain_mode not in (GAIN_FULL, GAIN_SCALAR):
            raise ValueError(f"unknown gain_mode {self.gain_mode!r}")
        self.magnet_positions = np.asarray(self.magnet_positions, dtype=float).reshape(
            geometry.N_MAGNETS, 3
        )
        self.sensor_positions = np.asarray(self.sensor_positions, dtype=float).reshape(
            geometry.N_SENSORS, 3
        )

    # ------------------------------------------------------------------ derived

    @property
    def rest_magnet_positions(self) -> np.ndarray:
        """Magnet positions at pose zero, including any fitted offset."""
        return self.magnet_positions + self.magnet_offsets

    @property
    def gain_scales(self) -> np.ndarray:
        """Per-sensor scale factor against `orientation`, shape (3,).

        Projection rather than a stored field, so it stays consistent with
        `gains` however those were last set. In scalar mode it is exact; in full
        mode it is the least-squares scale, i.e. how much of each sensor's gain
        matrix the shared orientation explains.
        """
        return np.einsum("sij,ij->s", self.gains, self.orientation) / 3.0

    @property
    def gain_magnitudes(self) -> np.ndarray:
        """Per-sensor gain magnitude, independent of orientation. Shape (3,).

        For a gain of the form ``g * R`` with *R* orthogonal, this returns
        exactly ``g`` -- unlike `gain_scales`, whose signed projection partially
        cancels when the true orientation differs from the stored one. Anything
        comparing gain against a *physical* prediction must use this, since the
        physics knows nothing about which axis convention was assumed.
        """
        return np.linalg.norm(self.gains.reshape(geometry.N_SENSORS, -1), axis=1) / np.sqrt(3.0)

    @property
    def n_free(self) -> int:
        n = (_N_SCALE if self.gain_mode == GAIN_SCALAR else _N_GAIN)
        n += _N_BIAS + _N_MOMENT
        return n + _N_OFFSET if self.fit_magnet_offsets else n

    def relax(self) -> CalibParams:
        """Switch to full 3x3 gains, keeping the current effective matrices.

        The staged handoff: the discrete search runs in scalar mode where the
        orientation is a real constraint, then the polish runs here where
        per-axis gain and cross-axis sensitivity can be absorbed.
        """
        return replace(self, gain_mode=GAIN_FULL)

    def gain_anisotropy(self) -> np.ndarray:
        """How far each sensor's gain is from a pure scaled rotation. Shape (3,).

        ``(s_max - s_min) / s_mean`` over the singular values. Zero means the
        sensor is a clean rotation times a single scale; a large value means real
        per-axis gain mismatch or cross-axis sensitivity.

        This replaced a comparison against the stored `orientation`, which was
        meaningless once the orientation stopped being a fitted quantity -- it
        falls out of the linear gain solve in `calibrate`, so the stored value is
        only ever the identity placeholder and the "deviation" measured nothing.
        Singular values need no reference frame.
        """
        svals = np.linalg.svd(self.gains, compute_uv=False)
        return (svals[:, 0] - svals[:, 2]) / np.maximum(svals.mean(axis=1), 1e-12)

    # ------------------------------------------------------------- constructors

    @classmethod
    def initial(
        cls,
        orientation: np.ndarray | None = None,
        magnet_signs: np.ndarray | None = None,
        gain_counts_per_unit: float = 1.0,
        fit_magnet_offsets: bool = False,
        gain_mode: str = GAIN_FULL,
    ) -> CalibParams:
        """Starting point for the fit.

        `orientation` is the shared board->sensor map -- one signed permutation
        applied to all three sensors, which is all the C++ firmware's
        ``Tx = mean(mag*x)`` mix allows (see the plan's "Prior knowledge").
        `magnet_signs` are the +/-1 axial polarities, which no prior pins down.
        """
        if orientation is None:
            orientation = np.eye(3)
        if magnet_signs is None:
            magnet_signs = np.ones(geometry.N_MAGNETS)

        orientation = np.asarray(orientation, dtype=float).reshape(3, 3)
        gains = np.broadcast_to(
            gain_counts_per_unit * orientation, (geometry.N_SENSORS, 3, 3)
        ).copy()

        moments = np.zeros((geometry.N_MAGNETS, 3))
        moments[:, 2] = np.asarray(magnet_signs, dtype=float)

        return cls(
            gains=gains,
            biases=np.zeros((geometry.N_SENSORS, 3)),
            moments=moments,
            fit_magnet_offsets=fit_magnet_offsets,
            gain_mode=gain_mode,
            orientation=orientation,
        )

    # ------------------------------------------------------------------ packing

    def pack(self) -> np.ndarray:
        """Flatten the free parameters into a solver vector."""
        if self.gain_mode == GAIN_SCALAR:
            head = self.gain_scales
        else:
            head = self.gains.ravel()
        parts = [head, self.biases.ravel(), self.moments.ravel()]
        if self.fit_magnet_offsets:
            parts.append(self.magnet_offsets.ravel())
        return np.concatenate(parts)

    def unpack(self, vec: np.ndarray) -> CalibParams:
        """Rebuild a `CalibParams` from a solver vector, keeping fixed fields."""
        vec = np.asarray(vec, dtype=float)
        if vec.size != self.n_free:
            raise ValueError(f"expected {self.n_free} parameters, got {vec.size}")

        i = 0
        if self.gain_mode == GAIN_SCALAR:
            gains = vec[i : i + _N_SCALE, None, None] * self.orientation
            i += _N_SCALE
        else:
            gains = vec[i : i + _N_GAIN].reshape(geometry.N_SENSORS, 3, 3)
            i += _N_GAIN
        biases = vec[i : i + _N_BIAS].reshape(geometry.N_SENSORS, 3)
        i += _N_BIAS
        moments = vec[i : i + _N_MOMENT].reshape(geometry.N_MAGNETS, 3)
        i += _N_MOMENT
        if self.fit_magnet_offsets:
            offsets = vec[i : i + _N_OFFSET].reshape(geometry.N_MAGNETS, 3)
        else:
            offsets = self.magnet_offsets

        return replace(
            self, gains=gains, biases=biases, moments=moments, magnet_offsets=offsets
        )

    # --------------------------------------------------------------------- I/O

    def to_dict(self) -> dict:
        return {
            "gains": self.gains.tolist(),
            "biases": self.biases.tolist(),
            "moments": self.moments.tolist(),
            "magnet_offsets": self.magnet_offsets.tolist(),
            "fit_magnet_offsets": self.fit_magnet_offsets,
            "gain_mode": self.gain_mode,
            "orientation": self.orientation.tolist(),
            "magnet_positions": self.magnet_positions.tolist(),
            "sensor_positions": self.sensor_positions.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> CalibParams:
        return cls(
            gains=np.array(data["gains"]),
            biases=np.array(data["biases"]),
            moments=np.array(data["moments"]),
            magnet_offsets=np.array(
                data.get("magnet_offsets", np.zeros((geometry.N_MAGNETS, 3)))
            ),
            fit_magnet_offsets=bool(data.get("fit_magnet_offsets", False)),
            gain_mode=str(data.get("gain_mode", GAIN_FULL)),
            orientation=np.array(data.get("orientation", np.eye(3))),
            magnet_positions=np.array(
                data.get("magnet_positions", geometry.magnet_positions())
            ),
            sensor_positions=np.array(
                data.get("sensor_positions", geometry.sensor_positions())
            ),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> CalibParams:
        return cls.from_dict(json.loads(Path(path).read_text()))


def signed_permutations_planar() -> list[np.ndarray]:
    """The 16 candidate board->sensor orientations.

    The C++ reference constrains this hard. ``MotionController.cpp`` averages the
    three x-channels for Tx (so all sensors share one orientation), uses only
    z-channels for Rx/Ry (so sensor z maps to +/- board z), and treats (x, y) as a
    genuine in-plane vector for Rz. That leaves 8 in-plane signed permutations
    times 2 z-signs = 16 -- versus 48^3 for three unconstrained rotations.
    """
    mats = []
    for swap_xy in (False, True):
        for sx in (1, -1):
            for sy in (1, -1):
                for sz in (1, -1):
                    m = np.zeros((3, 3))
                    if swap_xy:
                        m[0, 1], m[1, 0] = sx, sy
                    else:
                        m[0, 0], m[1, 1] = sx, sy
                    m[2, 2] = sz
                    mats.append(m)
    return mats
