"""The measurement function: six degrees of freedom in, nine counts out.

    h(x)_s = gain_s * ( sum_j B(sensor_s ; magnet_j at pose x) ) + offset_s

The sum over ``j`` runs over **all three magnets**, not just the one above the
sensor. That is not a refinement: with nominal geometry the two far magnets
contribute 8 to 14 counts at each sensor, against a measured noise floor of
about one count. Dropping them would put a pose-dependent error ten times the
noise straight into the innovation.

The state is the pose and nothing else. Every calibration constant lives in
:class:`~cadmouse.params.CalibParams` and is baked in before the filter runs,
so the firmware evaluates exactly this function with no calibration logic of
its own.

**Jacobian convention.** :func:`forward_and_jac` differentiates with respect to
a *local* perturbation: translation adds in the board frame, rotation
right-multiplies as ``R <- R exp(delta)``. This is the convention an iterated
EKF or an on-manifold Gauss-Newton step wants, and it avoids differentiating
Rodrigues' formula, which is where sign errors go to hide. Anything comparing
against finite differences must perturb the same way -- see
:func:`perturb_pose`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import (
    COUNTS_PER_MT,
    SENSOR_POS,
    right_jacobian_so3,
    rotation_from_rotvec,
    skew,
)
from .magnet import FieldTable, field_and_grad
from .params import CalibParams

#: Tesla -> counts. The field model works in SI; the sensor reports raw ADC
#: counts and the CSV stores them unconverted, so the comparison happens here.
TESLA_TO_COUNTS = 1e3 * COUNTS_PER_MT

#: Pose layout: three translations in mm, then a rotation vector in radians
#: about the knob's neutral centre.
POSE_DIM = 6
MEAS_DIM = 9


def pose_to_transform(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose, float)
    return pose[:3], rotation_from_rotvec(pose[3:6])


def perturb_pose(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Apply a local perturbation, matching the Jacobian convention.

    Translation is a plain board-frame addition; rotation composes on the right
    and is read back out as a rotation vector. Finite-difference checks must go
    through here or they will disagree with :func:`forward_and_jac` by exactly
    the amount Rodrigues' derivative differs from the identity.
    """
    pose = np.asarray(pose, float)
    delta = np.asarray(delta, float)
    rot = Rotation.from_rotvec(pose[3:6]) * Rotation.from_rotvec(delta[3:6])
    return np.concatenate([pose[:3] + delta[:3], rot.as_rotvec()])


def _geometry(pose: np.ndarray, params: CalibParams):
    """Magnet centres, axes and sensor offsets in the board frame."""
    translation, rotation = pose_to_transform(pose)
    body_axes = params.magnet_axes()

    centres = params.magnet_pos @ rotation.T + translation  # (3, 3) [magnet, xyz]
    axes = body_axes @ rotation.T  # (3, 3) [magnet, xyz]
    # delta[s, j] = sensor_s - magnet_centre_j
    delta = SENSOR_POS[:, None, :] - centres[None, :, :]
    return rotation, centres, axes, delta


def forward(pose: np.ndarray, params: CalibParams, table: FieldTable) -> np.ndarray:
    """Predicted counts, shape (9,), channel order MAG1/2/3 x,y,z."""
    _, _, axes, delta = _geometry(pose, params)

    b, _, _ = field_and_grad(
        table,
        delta,
        axes[None, :, :],
        params.magnet_moment[None, :],
        want_grad=False,
    )
    field_by_sensor = b.sum(axis=1)  # (3 sensors, xyz), tesla

    counts = params.sensor_gain * field_by_sensor * TESLA_TO_COUNTS
    return (counts + params.sensor_offset).ravel()


def forward_and_jac(
    pose: np.ndarray, params: CalibParams, table: FieldTable
) -> tuple[np.ndarray, np.ndarray]:
    """Predicted counts and ``d(counts)/d(local pose perturbation)``, (9, 6).

    Both magnet positions and magnet axes move with the pose, and both
    contribute. Under the local convention the derivatives are clean:

        d(centre_j)/d(rotation) = -R [p_j]_x
        d(axis_j)/d(rotation)   = -R [n_j]_x

    and ``delta = sensor - centre`` flips the sign of the first.
    """
    rotation, _, axes, delta = _geometry(pose, params)
    body_axes = params.magnet_axes()

    b, grad_delta, grad_axis = field_and_grad(
        table, delta, axes[None, :, :], params.magnet_moment[None, :]
    )
    field_by_sensor = b.sum(axis=1)

    # Translation: delta depends on it as -I, identically for every magnet.
    d_translation = -grad_delta.sum(axis=1)  # (3 sensors, xyz out, xyz in)

    # Rotation: accumulate per magnet, since each carries its own lever arm.
    d_rotation = np.zeros((3, 3, 3))
    for j in range(3):
        lever = rotation @ skew(params.magnet_pos[j])  # d(delta)/d(rot)
        spin = -rotation @ skew(body_axes[j])  # d(axis)/d(rot)
        d_rotation += grad_delta[:, j] @ lever + grad_axis[:, j] @ spin

    jac = np.concatenate([d_translation, d_rotation], axis=2)  # (3, 3, 6)
    jac = jac * (params.sensor_gain * TESLA_TO_COUNTS)[:, :, None]

    counts = params.sensor_gain * field_by_sensor * TESLA_TO_COUNTS
    return (counts + params.sensor_offset).ravel(), jac.reshape(MEAS_DIM, POSE_DIM)


# --------------------------------------------------------------------------
# Batched evaluation
#
# The calibration touches thousands of frames per iteration, so it needs the
# whole set evaluated at once. Crucially it also needs derivatives with respect
# to the *calibration parameters*, and those turn out to be free: the same
# ``dB/d(delta)`` and ``dB/d(axis)`` that give the pose Jacobian give every
# parameter derivative too, once the field is evaluated at unit moment and the
# moment applied afterwards. No finite differences anywhere.
# --------------------------------------------------------------------------


@dataclass
class BatchEvaluation:
    """Unit-moment fields and gradients for a batch of poses.

    Everything here is *per unit moment*, so the caller scales by the actual
    moments. That is what makes the moment derivative fall out as simply the
    unit field, with no division by a parameter that a broken fit might have
    driven to zero.

    Shapes, with ``n`` frames, ``s`` sensors, ``j`` magnets:
    ``field`` (n, s, j, 3); ``grad_delta`` and ``grad_axis`` (n, s, j, 3, 3)
    indexed ``[..., output, input]``.
    """

    rotation: np.ndarray  # (n, 3, 3)
    axes: np.ndarray  # (n, j, 3) world-frame magnetisation directions
    delta: np.ndarray  # (n, s, j, 3) sensor minus magnet centre
    field: np.ndarray
    grad_delta: np.ndarray | None
    grad_axis: np.ndarray | None


def evaluate_batch(
    poses: np.ndarray, params: CalibParams, table: FieldTable, want_grad: bool = True
) -> BatchEvaluation:
    poses = np.atleast_2d(np.asarray(poses, float))
    translation = poses[:, :3]
    rotation = Rotation.from_rotvec(poses[:, 3:6]).as_matrix()

    body_axes = params.magnet_axes()
    centres = np.einsum("nab,jb->nja", rotation, params.magnet_pos) + translation[:, None, :]
    axes = np.einsum("nab,jb->nja", rotation, body_axes)
    delta = SENSOR_POS[None, :, None, :] - centres[:, None, :, :]

    field, grad_delta, grad_axis = field_and_grad(
        table, delta, axes[:, None, :, :], np.ones(1), want_grad=want_grad
    )
    return BatchEvaluation(rotation, axes, delta, field, grad_delta, grad_axis)


def forward_batch(
    poses: np.ndarray, params: CalibParams, table: FieldTable
) -> np.ndarray:
    """Predicted counts for many poses at once, shape (n, 9)."""
    ev = evaluate_batch(poses, params, table, want_grad=False)
    tesla = (ev.field * params.magnet_moment[None, None, :, None]).sum(axis=2)
    counts = params.sensor_gain[None] * tesla * TESLA_TO_COUNTS
    return (counts + params.sensor_offset[None]).reshape(-1, MEAS_DIM)


def pose_jacobian_batch(
    ev: BatchEvaluation, params: CalibParams, poses: np.ndarray
) -> np.ndarray:
    """``d(counts)/d(pose vector)``, shape (n, 9, 6).

    Differentiates with respect to the rotation *vector*, not a local
    perturbation, so that a general-purpose optimiser can vary it directly.
    The conversion is the right Jacobian of SO(3); see
    :func:`~cadmouse.geometry.right_jacobian_so3`.
    """
    poses = np.atleast_2d(np.asarray(poses, float))
    moment = params.magnet_moment[None, None, :, None, None]
    grad_delta = ev.grad_delta * moment
    grad_axis = ev.grad_axis * moment

    d_translation = -grad_delta.sum(axis=2)

    body_axes = params.magnet_axes()
    d_rotation = np.zeros_like(d_translation)
    for j in range(3):
        lever = np.einsum("nab,bc->nac", ev.rotation, skew(params.magnet_pos[j]))
        spin = -np.einsum("nab,bc->nac", ev.rotation, skew(body_axes[j]))
        d_rotation += np.einsum("nsoi,nik->nsok", grad_delta[:, :, j], lever)
        d_rotation += np.einsum("nsoi,nik->nsok", grad_axis[:, :, j], spin)

    right = np.stack([right_jacobian_so3(p[3:6]) for p in poses])
    d_rotation = np.einsum("nsoi,nik->nsok", d_rotation, right)

    jac = np.concatenate([d_translation, d_rotation], axis=3)
    jac = jac * (params.sensor_gain * TESLA_TO_COUNTS)[None, :, :, None]
    return jac.reshape(-1, MEAS_DIM, POSE_DIM)


def forward_and_jac_vector(
    pose: np.ndarray, params: CalibParams, table: FieldTable
) -> tuple[np.ndarray, np.ndarray]:
    """Counts and ``d(counts)/d(pose vector)`` for a single pose, (9,) and (9, 6).

    The vector convention, matching what a filter carrying a plain
    six-element state expects. :func:`forward_and_jac` uses the local
    convention instead; see this module's docstring.
    """
    pose = np.asarray(pose, float).reshape(1, POSE_DIM)
    ev = evaluate_batch(pose, params, table)
    tesla = (ev.field * params.magnet_moment[None, None, :, None]).sum(axis=2)
    counts = params.sensor_gain[None] * tesla * TESLA_TO_COUNTS + params.sensor_offset[None]
    return counts.reshape(MEAS_DIM), pose_jacobian_batch(ev, params, pose)[0]


def solve_pose(
    measurement: np.ndarray,
    params: CalibParams,
    table: FieldTable,
    initial: np.ndarray | None = None,
    sigma: np.ndarray | None = None,
    iterations: int = 30,
    tol: float = 1e-9,
) -> tuple[np.ndarray, bool]:
    """Levenberg-Marquardt pose from one nine-channel frame.

    Nine measurements for six unknowns, and locally the problem is beautifully
    conditioned -- the Jacobian's singular values span a factor of 2.5. That
    makes plain Gauss-Newton tempting, and it is a trap. Cold-started from pose
    zero over the envelope the knob actually reaches (2.5 mm, 11 degrees,
    measured by filtering the recorded free motion) undamped Gauss-Newton lands
    in the wrong basin for about a third of poses, ending thousands of counts
    away, and more iterations do not help: 134 of 200 converged at twelve
    iterations and 136 at fifty.

    Damping fixes it because the failure is a step-length problem, not an
    iteration-count one. A 16.5 mm lever arm turns an over-long rotation step
    into a large positional error, and the field the model then predicts has no
    resemblance to the measurement.

    Warm-started from a neighbouring frame -- which is what the calibration and
    the filter both do -- either method works; this only matters cold.

    Returns the pose and whether it converged.
    """
    pose = np.zeros(POSE_DIM) if initial is None else np.asarray(initial, float).copy()
    weight = np.ones(MEAS_DIM) if sigma is None else 1.0 / np.asarray(sigma, float)

    def cost_at(x):
        residual = (measurement - forward(x, params, table)) * weight
        return float(residual @ residual)

    cost = cost_at(pose)
    damping = 1e-3

    for _ in range(iterations):
        predicted, jac = forward_and_jac(pose, params, table)
        residual = (measurement - predicted) * weight
        jac = jac * weight[:, None]
        normal = jac.T @ jac
        gradient = jac.T @ residual
        scale = np.diag(np.maximum(np.diag(normal), 1e-12))

        accepted = False
        for _ in range(12):
            try:
                step = np.linalg.solve(normal + damping * scale, gradient)
            except np.linalg.LinAlgError:
                damping *= 10.0
                continue
            candidate = perturb_pose(pose, step)
            candidate_cost = cost_at(candidate)
            if candidate_cost < cost:
                pose, cost = candidate, candidate_cost
                damping = max(damping / 3.0, 1e-12)
                accepted = True
                break
            damping *= 10.0
            if damping > 1e12:
                break

        if not accepted:
            # Damping has grown until the step is negligible; either converged
            # or stuck, and the caller can tell from the residual either way.
            return pose, cost < 4.0 * MEAS_DIM
        if np.linalg.norm(step) < tol:
            return pose, True

    return pose, cost < 4.0 * MEAS_DIM
