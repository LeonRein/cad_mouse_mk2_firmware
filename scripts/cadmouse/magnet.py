"""Field of one axially magnetised cylinder, three ways.

The three ways exist for three different jobs:

``field_axisym_exact``
    Surface-charge integration. Slow, and the reference everything else is
    measured against. Matches the closed-form on-axis solution to the printed
    digits.

``FieldTable`` + ``sample``
    What the filter actually evaluates. A cylinder is axisymmetric, so its
    field is a function of just two variables -- distance along the axis and
    distance from it -- and a 2-D table with bicubic interpolation reproduces
    the exact field to about one count, which is the sensor's noise floor.
    ``sample_scalar`` is the same interpolator written the way it will be
    written in Rust; ``sample`` is the vectorised twin the host tooling uses.
    ``tests/test_magnet.py`` asserts they agree, which is the only reason the
    golden vectors can be trusted.

``field_dipole``
    A point dipole, for reference and as a fallback. Better than its reputation
    here: a cylinder whose length is twice its radius has an almost vanishing
    leading multipole correction, so at the 6 mm operating gap this is only
    about 2 % low rather than the 30 % a general rule of thumb would suggest.
    Still 20-odd counts, though, so it is a fallback and not the plan.

Everything is expressed **per unit magnetic moment** (1 A*m^2). The field of a
uniformly magnetised body is linear in its magnetisation, so carrying the
moment as a separate multiplier keeps the table independent of magnet grade,
lets the calibration's warm-up phase treat the moments as a linear problem, and
makes the reversed third magnet nothing more special than a negative number.

Lengths are millimetres at the interface and metres inside the integrals;
fields are tesla.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import MAGNET_DIAMETER, MAGNET_HEIGHT, MU0

# --------------------------------------------------------------------------
# Exact reference: surface-charge model
#
# A uniformly magnetised cylinder is equivalent to a pair of uniformly charged
# discs, magnetic charge density +-M on the end faces. The field is then a
# Coulomb integral over the two faces, which needs no elliptic integrals and
# converges fast enough to serve as ground truth.
# --------------------------------------------------------------------------


def _disc_quadrature(radius_mm: float, n_radial: int, n_angular: int):
    """Product rule over a disc: Gauss-Legendre radially, midpoint in angle.

    The angular direction is periodic, so equally spaced midpoints converge
    geometrically and there is nothing to gain from anything cleverer.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_radial)
    rho = radius_mm * 0.5 * (nodes + 1.0)
    w_rho = radius_mm * 0.5 * weights
    theta = 2.0 * np.pi * (np.arange(n_angular) + 0.5) / n_angular
    d_theta = 2.0 * np.pi / n_angular

    # Area element rho*d(rho)*d(theta), converted mm^2 -> m^2.
    area = (w_rho * rho)[:, None] * d_theta * np.ones(n_angular)[None, :] * 1e-6
    xy = np.stack(
        [
            np.outer(rho, np.cos(theta)),
            np.outer(rho, np.sin(theta)),
        ],
        axis=-1,
    )
    return xy.reshape(-1, 2), area.reshape(-1)


#: Face quadrature order. 8x16 was ample while the nearest tabulated point sat
#: 3 mm from the magnet, and stopped being so when the magnet shrank to one
#: disc: a shorter magnet puts its centre closer to the sensor, which drags the
#: whole table towards the body, and the surface-charge integrand gets sharp as
#: the observer approaches a face. At the current top of the z range 8x16 is
#: worth 4.6 counts and 16x32 is exact to the float32 the table stores. The
#: whole table builds in well under a second either way, so this is not a
#: trade worth making finely.
_QUAD_XY, _QUAD_AREA = _disc_quadrature(MAGNET_DIAMETER / 2.0, n_radial=16, n_angular=32)


def field_axisym_exact(
    rho_mm: np.ndarray,
    z_mm: np.ndarray,
    diameter_mm: float = MAGNET_DIAMETER,
    height_mm: float = MAGNET_HEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """``(B_rho, B_z)`` per unit moment, in the magnet frame.

    The magnet sits at the origin with its axis along ``+z``. ``rho_mm`` is the
    perpendicular distance from that axis, ``z_mm`` the signed distance along
    it. Both may be arrays of any matching shape.

    Do not evaluate inside the magnet body: the charge sheets are singular on
    the rim, and the quadrature will return a large finite number rather than
    complaining. Nothing in this application gets closer than about 4 mm, and
    :func:`sample` clamps its queries to the tabulated range regardless.
    """
    rho_mm, z_mm = np.broadcast_arrays(np.asarray(rho_mm, float), np.asarray(z_mm, float))
    shape = rho_mm.shape

    if (diameter_mm, height_mm) == (MAGNET_DIAMETER, MAGNET_HEIGHT):
        quad_xy, quad_area = _QUAD_XY, _QUAD_AREA
    else:
        quad_xy, quad_area = _disc_quadrature(diameter_mm / 2.0, 8, 16)

    volume_m3 = np.pi * (diameter_mm / 2e3) ** 2 * (height_mm / 1e3)
    # Unit moment means M = 1/V, and sigma_m = M on the faces. The Coulomb
    # prefactor mu0/(4 pi) then cancels against B = mu0 H.
    sigma = 1.0 / volume_m3
    prefactor = MU0 / (4.0 * np.pi) * sigma

    obs = np.stack([rho_mm.ravel(), np.zeros(rho_mm.size), z_mm.ravel()], axis=-1)

    b = np.zeros_like(obs)
    for face_z, sign in ((+height_mm / 2.0, +1.0), (-height_mm / 2.0, -1.0)):
        src = np.concatenate(
            [quad_xy, np.full((quad_xy.shape[0], 1), face_z)], axis=1
        )
        # (n_obs, n_quad, 3), mm -> m
        delta = (obs[:, None, :] - src[None, :, :]) * 1e-3
        dist = np.linalg.norm(delta, axis=-1)
        w = sign * prefactor * quad_area
        b += ((w[None, :] / dist**3)[:, :, None] * delta).sum(axis=1)

    return b[:, 0].reshape(shape), b[:, 2].reshape(shape)


def field_dipole(delta_mm: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Point-dipole field per unit moment, ``delta`` = observation - centre."""
    delta = np.asarray(delta_mm, float) * 1e-3
    axis = np.asarray(axis, float)
    r = np.linalg.norm(delta, axis=-1, keepdims=True)
    axial = np.sum(axis * delta, axis=-1, keepdims=True)
    return MU0 / (4.0 * np.pi) * (3.0 * axial * delta / r**5 - axis / r**3)


# --------------------------------------------------------------------------
# Interpolation table
#
# Grid extent is driven by what the mechanism can reach. Every sensor sits
# below every magnet, so the axial coordinate is always negative; the radial
# coordinate runs from near zero (a magnet over its own sensor) out to the ring
# chord of about 29 mm (a magnet over one of the *other* two sensors, which
# contributes 8-14 counts of cross-talk and cannot be dropped). Tilting the
# knob swings the axial coordinate by up to rho*sin(tilt), so the range is
# padded well past the nominal values.
# --------------------------------------------------------------------------

#: Default grid. 0.25 mm spacing lands the interpolation error well under the
#: sensor's one-count noise floor; there is no point going finer.
#:
#: The radial axis deliberately starts *below zero*. A magnet sitting directly
#: over its own sensor -- which is the rest pose, to within the half-millimetre
#: the ring radii differ by -- queries rho near zero, and a table that began at
#: zero would have to interpolate from a lopsided boundary stencil exactly
#: where the fit spends most of its time. Extending into negative rho costs
#: four columns and makes rho = 0 an ordinary interior point.
#:
#: Negative rho is not a fiction: :func:`field_axisym_exact` places the
#: observer on the far side of the axis there, and by symmetry that is the odd
#: continuation of B_rho and the even continuation of B_z, which is precisely
#: what a stencil straddling the axis needs to see.
#: Both extents are tied to :data:`~cadmouse.geometry.MAGNET_HEIGHT`, which is
#: easy to miss. The magnet's *bottom face* is what the mechanism fixes, so a
#: shorter magnet puts its centre nearer the sensor plane -- going from a
#: stacked pair to one disc moved it from 9 mm away to 7.5 mm, and moved the
#: whole reachable ``z`` window 1.5 mm towards the top of the table.
#:
#: A query past an edge is *clamped*, not rejected, so getting this wrong does
#: not raise: it silently returns the field from the edge of the table. That is
#: cheap in rho, where only the two far magnets reach the boundary and they are
#: worth ~10 counts, and expensive in z, where the clamped magnet is the near
#: one sitting over its own sensor at ~500 counts. Over the envelope the tests
#: probe, ``z`` reaches -2.75 and rho reaches 35.5; both ends carry a few grid
#: cells of margin past that.
DEFAULT_RHO_RANGE = (-1.0, 37.0)
DEFAULT_Z_RANGE = (-22.0, -2.0)
DEFAULT_SPACING = 0.25


@dataclass(frozen=True)
class FieldTable:
    """``(B_rho, B_z)`` per unit moment on a uniform ``(rho, z)`` grid.

    Stored as float32 because that is what the RP2350 will hold: the Cortex-M33
    has a single-precision FPU and double precision is soft-float. Keeping the
    host table at the same width means the golden vectors are not quietly
    generated from a more accurate table than the firmware owns.
    """

    rho0: float
    d_rho: float
    n_rho: int
    z0: float
    d_z: float
    n_z: int
    b_rho: np.ndarray  # (n_rho, n_z) float32, tesla per A*m^2
    b_z: np.ndarray

    @property
    def rho_max(self) -> float:
        return self.rho0 + self.d_rho * (self.n_rho - 1)

    @property
    def z_max(self) -> float:
        return self.z0 + self.d_z * (self.n_z - 1)

    def nbytes(self) -> int:
        return self.b_rho.nbytes + self.b_z.nbytes


def build_table(
    rho_range: tuple[float, float] = DEFAULT_RHO_RANGE,
    z_range: tuple[float, float] = DEFAULT_Z_RANGE,
    spacing: float = DEFAULT_SPACING,
) -> FieldTable:
    """Tabulate :func:`field_axisym_exact` over a uniform grid."""
    n_rho = int(round((rho_range[1] - rho_range[0]) / spacing)) + 1
    n_z = int(round((z_range[1] - z_range[0]) / spacing)) + 1
    rho = rho_range[0] + spacing * np.arange(n_rho)
    z = z_range[0] + spacing * np.arange(n_z)

    b_rho, b_z = field_axisym_exact(rho[:, None], z[None, :])
    return FieldTable(
        rho0=rho_range[0],
        d_rho=spacing,
        n_rho=n_rho,
        z0=z_range[0],
        d_z=spacing,
        n_z=n_z,
        b_rho=np.ascontiguousarray(b_rho, dtype=np.float32),
        b_z=np.ascontiguousarray(b_z, dtype=np.float32),
    )


# --------------------------------------------------------------------------
# Bicubic interpolation
#
# Keys' cubic convolution with a = -1/2: separable, C1, needs only a 4x4
# neighbourhood, and its derivative is available in closed form -- which is
# what makes an analytic measurement Jacobian nearly free, and hence what makes
# an iterated EKF cheaper than a UKF here.
#
# This is deliberately *not* a spline. A spline would fit the samples better,
# but it needs a global solve and is a different function from anything that
# will be written in Rust. Prototyping against one function and shipping
# another is a debugging trap, so the host uses the interpolator the firmware
# will use.
# --------------------------------------------------------------------------


def _cubic_weights(t: float) -> tuple[float, float, float, float]:
    t2 = t * t
    t3 = t2 * t
    return (
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    )


def _cubic_weight_derivs(t: float) -> tuple[float, float, float, float]:
    t2 = t * t
    return (
        -1.5 * t2 + 2.0 * t - 0.5,
        4.5 * t2 - 5.0 * t,
        -4.5 * t2 + 4.0 * t + 0.5,
        1.5 * t2 - 1.0 * t,
    )


def _stencil_index(
    coord: float, origin: float, step: float, count: int
) -> tuple[int, float, float]:
    """4-point stencil base index and interpolation fraction.

    Two distinct clamps, and conflating them is a trap worth naming. The
    *stencil* is clamped so it always lies inside the array, which in the first
    and last cell leaves the fraction outside ``[0, 1]`` -- a shifted stencil,
    evaluating the same local cubic slightly beyond its middle interval, which
    stays accurate. The *query* is clamped only when it falls off the grid
    entirely, so that a sigma point wandering out during a filter transient
    yields a merely wrong field rather than a diverging one.

    Clamping the query to the stencil's range instead, which is the obvious
    thing to write, silently snaps every point in the first cell onto the
    second grid node. On the radial axis that is the rest pose, and the error
    reaches tens of counts against a one-count noise floor.

    The third return value is the derivative's validity: zero where the query
    was clamped, one elsewhere. Beyond the grid the interpolant is constant, so
    its derivative is zero, and reporting the cubic's slope there instead would
    hand an optimiser or a filter a phantom gradient pushing it further out.
    """
    raw = (coord - origin) / step
    u = min(max(raw, 0.0), count - 1.0)
    base = int(np.floor(u))
    i0 = min(max(base - 1, 0), count - 4)
    return i0, u - (i0 + 1), 0.0 if raw != u else 1.0


def sample_scalar(table: FieldTable, rho: float, z: float):
    """Reference interpolator: one point, plain scalars, port-shaped.

    Returns ``(b_rho, b_z, db_rho_d_rho, db_rho_dz, db_z_d_rho, db_z_dz)``.
    Both fields share the stencil and the weights, so evaluating them together
    costs barely more than evaluating one.
    """
    i0, tu, live_u = _stencil_index(rho, table.rho0, table.d_rho, table.n_rho)
    j0, tv, live_v = _stencil_index(z, table.z0, table.d_z, table.n_z)

    wu = _cubic_weights(tu)
    wv = _cubic_weights(tv)
    du = _cubic_weight_derivs(tu)
    dv = _cubic_weight_derivs(tv)

    b_rho = b_z = 0.0
    d_rho_du = d_rho_dv = 0.0
    d_z_du = d_z_dv = 0.0
    for i in range(4):
        for j in range(4):
            f_rho = float(table.b_rho[i0 + i, j0 + j])
            f_z = float(table.b_z[i0 + i, j0 + j])
            b_rho += wu[i] * wv[j] * f_rho
            b_z += wu[i] * wv[j] * f_z
            d_rho_du += du[i] * wv[j] * f_rho
            d_rho_dv += wu[i] * dv[j] * f_rho
            d_z_du += du[i] * wv[j] * f_z
            d_z_dv += wu[i] * dv[j] * f_z

    return (
        b_rho,
        b_z,
        live_u * d_rho_du / table.d_rho,
        live_v * d_rho_dv / table.d_z,
        live_u * d_z_du / table.d_rho,
        live_v * d_z_dv / table.d_z,
    )


def sample(table: FieldTable, rho: np.ndarray, z: np.ndarray):
    """Vectorised twin of :func:`sample_scalar`, same tuple of six arrays."""
    rho = np.asarray(rho, float)
    z = np.asarray(z, float)

    # See _stencil_index: clamp the query only when it leaves the grid, then
    # clamp the stencil separately and let the fraction run outside [0, 1] in
    # the boundary cells.
    raw_u = (rho - table.rho0) / table.d_rho
    raw_v = (z - table.z0) / table.d_z
    u = np.clip(raw_u, 0.0, table.n_rho - 1.0)
    v = np.clip(raw_v, 0.0, table.n_z - 1.0)
    live_u = (raw_u == u).astype(float)
    live_v = (raw_v == v).astype(float)
    iu = np.clip(np.floor(u).astype(np.intp) - 1, 0, table.n_rho - 4)
    iv = np.clip(np.floor(v).astype(np.intp) - 1, 0, table.n_z - 4)
    tu = u - (iu + 1)
    tv = v - (iv + 1)

    def weights(t):
        t2, t3 = t * t, t * t * t
        return np.stack(
            [
                -0.5 * t3 + t2 - 0.5 * t,
                1.5 * t3 - 2.5 * t2 + 1.0,
                -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                0.5 * t3 - 0.5 * t2,
            ]
        )

    def weight_derivs(t):
        t2 = t * t
        return np.stack(
            [
                -1.5 * t2 + 2.0 * t - 0.5,
                4.5 * t2 - 5.0 * t,
                -4.5 * t2 + 4.0 * t + 0.5,
                1.5 * t2 - 1.0 * t,
            ]
        )

    wu, wv = weights(tu), weights(tv)
    du, dv = weight_derivs(tu), weight_derivs(tv)

    # Gather the 4x4 neighbourhood: (4, 4, ...) with the query shape trailing.
    idx_u = iu[None, ...] + np.arange(4).reshape(4, *([1] * rho.ndim))
    idx_v = iv[None, ...] + np.arange(4).reshape(4, *([1] * z.ndim))
    gather_u = idx_u[:, None, ...]
    gather_v = idx_v[None, :, ...]
    f_rho = table.b_rho[gather_u, gather_v].astype(float)
    f_z = table.b_z[gather_u, gather_v].astype(float)

    wu4 = wu[:, None, ...]
    wv4 = wv[None, :, ...]
    du4 = du[:, None, ...]
    dv4 = dv[None, :, ...]

    def contract(f, a, b):
        return (a * b * f).sum(axis=(0, 1))

    return (
        contract(f_rho, wu4, wv4),
        contract(f_z, wu4, wv4),
        live_u * contract(f_rho, du4, wv4) / table.d_rho,
        live_v * contract(f_rho, wu4, dv4) / table.d_z,
        live_u * contract(f_z, du4, wv4) / table.d_rho,
        live_v * contract(f_z, wu4, dv4) / table.d_z,
    )


# --------------------------------------------------------------------------
# Cartesian field and its gradients
# --------------------------------------------------------------------------

#: Below this radial distance the ``e_rho`` direction is ill-defined. On axis
#: the radial field is zero by symmetry, so returning the axial term alone is
#: exact rather than merely safe.
_RHO_FLOOR = 1e-9


def field(table: FieldTable, delta: np.ndarray, axis: np.ndarray, moment) -> np.ndarray:
    """Field in the board frame from one magnet.

    ``delta`` is ``sensor_position - magnet_centre`` in millimetres, ``axis``
    the unit magnetisation direction, ``moment`` the signed moment in A*m^2.
    Leading dimensions broadcast.
    """
    b, _, _ = field_and_grad(table, delta, axis, moment, want_grad=False)
    return b


def field_and_grad(
    table: FieldTable,
    delta: np.ndarray,
    axis: np.ndarray,
    moment,
    want_grad: bool = True,
):
    """Field plus ``dB/d(delta)`` and ``dB/d(axis)``.

    Returns ``(B, dB_ddelta, dB_daxis)`` with the Jacobians shaped
    ``(..., 3, 3)`` and indexed ``[output, input]``, or ``(B, None, None)``
    when ``want_grad`` is false. ``dB/d(delta)`` carries units of tesla per
    millimetre, matching the millimetre input.

    The chain rule here is worth spelling out because it is the whole reason
    the table stores derivatives. Writing ``zc = delta . axis`` for the axial
    coordinate and ``e`` for the radial unit vector,

        B = m * ( b_rho(rho, zc) * e + b_z(rho, zc) * axis )

    and both ``rho`` and ``e`` depend on ``delta`` and ``axis`` in turn. The
    terms below are that expansion, with ``e . axis == 0`` used to drop the
    cross terms it kills.
    """
    delta = np.asarray(delta, float)
    axis = np.asarray(axis, float)
    moment = np.asarray(moment, float)[..., None]

    zc = np.sum(delta * axis, axis=-1, keepdims=True)
    radial = delta - zc * axis
    rho = np.linalg.norm(radial, axis=-1, keepdims=True)
    safe_rho = np.maximum(rho, _RHO_FLOOR)
    e = radial / safe_rho

    b_rho, b_z, dbr_dr, dbr_dz, dbz_dr, dbz_dz = sample(
        table, rho[..., 0], zc[..., 0]
    )
    b_rho = b_rho[..., None]
    b_z = b_z[..., None]

    b = moment * (b_rho * e + b_z * axis)
    if not want_grad:
        return b, None, None

    tail = rho.shape[:-1] + (1,)
    dbr_dr = dbr_dr.reshape(tail)
    dbr_dz = dbr_dz.reshape(tail)
    dbz_dr = dbz_dr.reshape(tail)
    dbz_dz = dbz_dz.reshape(tail)

    eye = np.eye(3)
    outer = lambda p, q: p[..., :, None] * q[..., None, :]  # noqa: E731

    # d/d(delta): drho/d(delta) = e^T, dzc/d(delta) = axis^T,
    # de/d(delta) = (1/rho) (I - e e^T - axis axis^T).
    grad_delta = (
        outer(e, dbr_dr * e + dbr_dz * axis)
        + b_rho[..., None] * (eye - outer(e, e) - outer(axis, axis)) / safe_rho[..., None]
        + outer(axis, dbz_dr * e + dbz_dz * axis)
    )

    # d/d(axis): drho/d(axis) = -zc e^T, dzc/d(axis) = delta^T,
    # de/d(axis) = (1/rho) ( -zc (I - e e^T) - axis delta^T ).
    grad_axis = (
        outer(e, -zc * dbr_dr * e + dbr_dz * delta)
        + b_rho[..., None]
        * (-zc[..., None] * (eye - outer(e, e)) - outer(axis, delta))
        / safe_rho[..., None]
        + outer(axis, -zc * dbz_dr * e + dbz_dz * delta)
        + b_z[..., None] * eye
    )

    scale = moment[..., None]
    return b, scale * grad_delta, scale * grad_axis
