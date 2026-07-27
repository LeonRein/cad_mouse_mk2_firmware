"""The forward model: knob pose -> the nine raw sensor counts.

This is the single source of truth that everything else inverts. The calibration
(`calibrate`) fits its parameters, the UKF (`ukf`) uses it as its measurement
function, and `observability` differentiates it.

    for magnet j:   q_j = R(pose) @ q_j0 + t(pose)      # magnets ride the knob
                    m_j = R(pose) @ m_j0
    for sensor i:   B_i = sum_j field(p_i - q_j, m_j)   # board frame, model units
                    z_i = A_i @ B_i + b_i               # sensor frame, counts

Everything is vectorised over a leading batch axis of poses, because the
bundle adjustment evaluates thousands of poses per solver iteration.
"""

from __future__ import annotations

import numpy as np

from . import geometry
from .magnet import MagnetModel, PointDipole, SubdipoleCylinder
from .params import CalibParams

#: The finite-size model is the default, not the fallback. Measured on-axis
#: error at the 6 mm operating gap (`tests/test_forward.py`):
#:
#:     PointDipole                      +29.6 %
#:     SubdipoleCylinder(3,8,2)          -0.13 %
#:
#: 30 % is a systematic *shape* error, so it biases the recovered pose rather
#: than averaging out.
#:
#: 48 Gauss-Legendre nodes, which is both cheaper and ten times more accurate
#: than the 96-node uniform grid this replaced -- see `SubdipoleCylinder`. The
#: node count is a direct multiplier on calibration time, so it is not set
#: higher than the sensor's 1-count quantisation can justify.
DEFAULT_MODEL: MagnetModel = SubdipoleCylinder()

#: Cheap model for the coarse stage of the fit, where being 30 % off still lands
#: in the right basin. `calibrate` uses this before polishing with the default.
COARSE_MODEL: MagnetModel = PointDipole()


#: Nominal body axis of every magnet: the discs are axially magnetised and sit
#: flat in the knob, so their symmetry axis is vertical at rest. Unlike the
#: moment, this is *not* fitted -- a glue-in tilt changes where the magnetisation
#: points, not the fact that the part is a flat disc.
NOMINAL_MAGNET_AXIS = np.array([0.0, 0.0, 1.0])


def board_field(
    pose: np.ndarray, params: CalibParams, model: MagnetModel = DEFAULT_MODEL
) -> np.ndarray:
    """Board-frame field at each sensor, in model units. (S, 3) or (N, S, 3)."""
    axes = np.broadcast_to(NOMINAL_MAGNET_AXIS, (geometry.N_MAGNETS, 3))
    positions, moments = geometry.transform_knob(
        pose, params.rest_magnet_positions, params.moments
    )
    _, rotated_axes = geometry.transform_knob(pose, params.rest_magnet_positions, axes)
    return model.field(params.sensor_positions, positions, moments, rotated_axes)


def moment_basis(
    pose: np.ndarray, params: CalibParams, model: MagnetModel = DEFAULT_MODEL
) -> np.ndarray:
    """Board-frame field per unit of each magnet moment component. (N, S, 3, 9).

    Exploits the model's linearity in the moments: evaluating the nine unit
    moments spans everything the magnets can produce, so `calibrate` can solve
    for them exactly rather than iteratively.
    """
    from dataclasses import replace as _replace

    basis = np.eye(geometry.N_MAGNETS * 3).reshape(-1, geometry.N_MAGNETS, 3)
    columns = [
        board_field(pose, _replace(params, moments=unit), model) for unit in basis
    ]
    return np.stack(columns, axis=-1)


def predict(
    pose: np.ndarray, params: CalibParams, model: MagnetModel = DEFAULT_MODEL
) -> np.ndarray:
    """Predicted sensor counts. Shape (9,) for one pose, (N, 9) for a batch."""
    b = board_field(pose, params, model)
    counts = np.einsum("sij,...sj->...si", params.gains, b) + params.biases
    return counts.reshape(counts.shape[:-2] + (geometry.N_CHANNELS,))


def pose_jacobian(
    pose: np.ndarray,
    params: CalibParams,
    model: MagnetModel = DEFAULT_MODEL,
    step: np.ndarray | float = 1e-6,
) -> np.ndarray:
    """d(counts)/d(pose) by central differences. (9, 6) or (N, 9, 6).

    Central differences rather than analytic derivatives: the model is smooth and
    well-scaled, this is only used for the Gauss-Newton seed and the
    observability analysis, and an analytic dB/dpose is a lot of surface area to
    get subtly wrong. `calibrate` lets SciPy do its own differencing.
    """
    pose = np.asarray(pose, dtype=float)
    single = pose.ndim == 1
    flat = np.atleast_2d(pose)
    h = np.broadcast_to(np.asarray(step, dtype=float), (6,))

    cols = []
    for k in range(6):
        delta = np.zeros(6)
        delta[k] = h[k]
        plus = predict(flat + delta, params, model)
        minus = predict(flat - delta, params, model)
        cols.append((plus - minus) / (2.0 * h[k]))

    jac = np.stack(cols, axis=-1)  # (N, 9, 6)
    return jac[0] if single else jac
