"""Magnet field models.

**Units.** Positions are millimetres and moments are dimensionless. The models
return field in "model units", defined by taking mu0/4pi = 1:

    B = (3 rhat (rhat . m) - m) / |r|^3

The conversion from model units to sensor counts is absorbed entirely into the
per-sensor gain matrix ``A_i``. This is why the calibration pins ||m_1|| = 1:
scaling every moment by *s* and every gain by 1/*s* is unobservable, so one of
them has to be fixed. Nothing downstream needs SI teslas.

**Why two models.** At the rest pose a magnet sits 6 mm from its sensor and is
6 mm in diameter -- exactly one diameter away, where the point-dipole
approximation carries a ~10-20 % error. That error is a systematic *shape*
error, so it biases the recovered pose rather than merely adding noise.
`SubdipoleCylinder` is the fallback when residuals show structure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The BOM disc: 6 mm diameter x 3 mm thick neodymium
# (`cad-mouse-mk2/bom/CAD Mouse MK2 _bom.csv`).
MAGNET_DIAMETER_MM = 6.0
MAGNET_THICKNESS_MM = 3.0

#: Remanence of the disc, in tesla. The BOM does not state a grade, so N35 is
#: assumed -- the cheapest common grade, and what an unspecified hobby magnet
#: usually is. Grades N35..N52 span roughly 1.17-1.45 T, so treat this as good
#: to about +/-25 %.
MAGNET_REMANENCE_T = 1.19

VACUUM_PERMEABILITY = 4e-7 * np.pi


def physical_moment_am2(
    remanence_t: float = MAGNET_REMANENCE_T,
    diameter_mm: float = MAGNET_DIAMETER_MM,
    thickness_mm: float = MAGNET_THICKNESS_MM,
) -> float:
    """Magnetic moment of the disc in A m^2, from its size and grade.

    ``m = Br * V / mu0`` for a uniformly magnetised body.
    """
    volume_m3 = np.pi * (diameter_mm / 2e3) ** 2 * (thickness_mm / 1e3)
    return remanence_t * volume_m3 / VACUUM_PERMEABILITY


def model_units_to_mt(moment_am2: float) -> float:
    """Millitesla per unit of model field, for a magnet of the given moment.

    The model works in millimetres with mu0/4pi = 1 and a dimensionless moment
    (see the module docstring). Restoring SI:

        B[T] = (mu0/4pi) * m[A m^2] * shape / r[m]^3
        B_model =                     1 * shape / r[mm]^3

    so ``B[T] = 1e-7 * m * 1e9 * B_model``, i.e. ``B[mT] = 1e5 * m * B_model``.
    """
    return 1e5 * moment_am2


def _dipole_field(r: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Point-dipole field at offset ``r`` from a dipole of moment ``m``.

    ``r`` and ``m`` broadcast against each other on a trailing axis of size 3.
    """
    r2 = np.sum(r * r, axis=-1, keepdims=True)
    r1 = np.sqrt(r2)
    r_dot_m = np.sum(r * m, axis=-1, keepdims=True)
    return (3.0 * r * r_dot_m / r2 - m) / (r1 * r2)


class MagnetModel:
    """Interface: field at `points` from magnets at `positions` with `moments`.

    All implementations are pure functions of their arguments and vectorised
    over a leading batch axis, because the calibration evaluates thousands of
    poses per solver iteration.
    """

    def field(
        self,
        points: np.ndarray,
        positions: np.ndarray,
        moments: np.ndarray,
        axes: np.ndarray | None = None,
    ) -> np.ndarray:
        """Sum the field of every magnet at every point.

        Args:
            points: (S, 3) sensor positions, fixed in the board frame.
            positions: (M, 3) or (N, M, 3) magnet positions.
            moments: same shape as ``positions``.
            axes: same shape -- each magnet's *body* symmetry axis, which orients
                its finite-size shape. Defaults to the moment direction.

        Returns:
            (S, 3) or (N, S, 3) field in model units.

        Every implementation is **linear in** ``moments`` for fixed ``positions``
        and ``axes``. `calibrate` depends on this: it is what lets the bootstrap
        recover the magnet moments by an exact linear solve instead of hoping a
        nonlinear optimiser finds them.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class PointDipole(MagnetModel):
    """Each magnet as a single point dipole. Fast, and wrong by ~10-20 % here."""

    def field(
        self,
        points: np.ndarray,
        positions: np.ndarray,
        moments: np.ndarray,
        axes: np.ndarray | None = None,
    ) -> np.ndarray:
        del axes  # a point has no shape to orient
        points = np.asarray(points, dtype=float)
        positions = np.asarray(positions, dtype=float)
        moments = np.asarray(moments, dtype=float)
        # points (S, 1, 3) - positions (..., 1, M, 3) -> (..., S, M, 3)
        r = points[..., :, None, :] - positions[..., None, :, :]
        m = np.broadcast_to(moments[..., None, :, :], r.shape)
        return _dipole_field(r, m).sum(axis=-2)


@dataclass(frozen=True)
class SubdipoleCylinder(MagnetModel):
    """A uniformly magnetised disc as a weighted set of sub-dipoles.

    Converges to the true finite-cylinder field as the node count rises.

    **Quadrature choice matters more than node count.** The radial and axial
    integrals use Gauss-Legendre nodes rather than a uniform grid: the radial
    one in ``u = (r/R)^2``, which is the variable a uniformly magnetised disc is
    uniform in, and the axial one directly in *z*. Measured on-axis error at the
    6 mm operating gap:

        uniform grid,  4x8x3 = 96 nodes    -1.08 %
        Gauss-Legendre 2x6x2 = 24 nodes    -0.17 %

    Four times cheaper and six times more accurate. The angular direction stays
    on equally spaced nodes, which is already spectrally accurate for a periodic
    integrand.

    Because Gauss-Legendre weights are not equal, sub-dipoles carry *weighted*
    moments -- see `weights`.
    """

    n_r: int = 3
    n_theta: int = 8
    n_z: int = 2
    diameter_mm: float = MAGNET_DIAMETER_MM
    thickness_mm: float = MAGNET_THICKNESS_MM

    def _nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Sub-dipole offsets (K, 3) and their moment fractions (K,).

        Offsets are in the magnet's own frame, whose local +z is the symmetry
        axis; they get rotated by whatever takes local +z onto the body axis.
        The weights sum to 1, so the sub-moments always reconstitute the whole.
        """
        radius = self.diameter_mm / 2.0

        # Radial: Gauss-Legendre in u = (r/R)^2 over [0, 1].
        u_nodes, u_weights = np.polynomial.legendre.leggauss(self.n_r)
        u = (u_nodes + 1.0) / 2.0
        u_weights = u_weights / 2.0
        r = radius * np.sqrt(u)

        # Axial: Gauss-Legendre over the thickness.
        z_nodes, z_weights = np.polynomial.legendre.leggauss(self.n_z)
        z = z_nodes * self.thickness_mm / 2.0
        z_weights = z_weights / 2.0

        # Angular: equally spaced, already spectrally accurate here.
        theta = 2.0 * np.pi * (np.arange(self.n_theta) + 0.5) / self.n_theta

        r_grid, th_grid, z_grid = np.meshgrid(r, theta, z, indexing="ij")
        offsets = np.stack(
            [
                (r_grid * np.cos(th_grid)).ravel(),
                (r_grid * np.sin(th_grid)).ravel(),
                z_grid.ravel(),
            ],
            axis=-1,
        )
        weights = (
            u_weights[:, None, None]
            * np.full((1, self.n_theta, 1), 1.0 / self.n_theta)
            * z_weights[None, None, :]
        ).ravel()
        return offsets, weights

    def field(
        self,
        points: np.ndarray,
        positions: np.ndarray,
        moments: np.ndarray,
        axes: np.ndarray | None = None,
    ) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        positions = np.asarray(positions, dtype=float)
        moments = np.asarray(moments, dtype=float)
        # The *body* axis orients the shape, not the moment. Physically right --
        # a disc glued in with a slight tilt between its geometric axis and its
        # magnetisation still has a disc-shaped body -- and it is what keeps the
        # field linear in `moments`, which `calibrate`'s linear bootstrap needs.
        axes = moments if axes is None else np.asarray(axes, dtype=float)

        offsets, weights = self._nodes()  # (K, 3), (K,) in the local frame

        rot = _align_z_to(axes)  # (..., M, 3, 3)
        # (..., M, K, 3)
        world_offsets = np.einsum("...mij,kj->...mki", rot, offsets)

        sub_pos = positions[..., :, None, :] + world_offsets
        # Quadrature weights, not an even split: Gauss-Legendre nodes are not
        # equally weighted. They sum to 1, so the sub-moments total the whole.
        sub_mom = moments[..., :, None, :] * weights[:, None]
        sub_mom = np.broadcast_to(sub_mom, sub_pos.shape)

        # Collapse (M, K) into one magnet axis and reuse the dipole sum.
        flat_shape = sub_pos.shape[:-3] + (-1, 3)
        return PointDipole().field(
            points, sub_pos.reshape(flat_shape), sub_mom.reshape(flat_shape)
        )


def _align_z_to(vectors: np.ndarray) -> np.ndarray:
    """Rotation matrices taking local +z onto each (unnormalised) vector.

    The rotation about the target axis is arbitrary and irrelevant: the
    sub-dipole cloud is axisymmetric by construction. Degenerate cases (a
    moment pointing along -z, or a zero moment) fall back to a fixed axis
    rather than producing NaNs.
    """
    v = np.asarray(vectors, dtype=float)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    unit = np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)
    # Zero moments get an identity rotation; their field is zero anyway.
    unit = np.where(norm > 0, unit, np.array([0.0, 0.0, 1.0]))

    z = np.zeros_like(unit)
    z[..., 2] = 1.0
    axis = np.cross(z, unit)
    sin = np.linalg.norm(axis, axis=-1, keepdims=True)
    cos = unit[..., 2:3]

    # Antiparallel (sin == 0, cos == -1): a pi rotation about any perpendicular
    # axis works; pick x.
    fallback = np.zeros_like(axis)
    fallback[..., 0] = 1.0
    axis = np.where(sin > 1e-12, axis, fallback)
    sin = np.where(sin > 1e-12, sin, 0.0)

    k = np.divide(axis, np.linalg.norm(axis, axis=-1, keepdims=True))
    # Rodrigues: R = I + sin*K + (1-cos)*K^2
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    kmat = np.stack(
        [zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis=-1
    ).reshape(k.shape[:-1] + (3, 3))

    eye = np.broadcast_to(np.eye(3), kmat.shape)
    s = sin[..., None]
    c = cos[..., None]
    return eye + s * kmat + (1.0 - c) * (kmat @ kmat)


def cylinder_on_axis_bz(
    z_mm: np.ndarray,
    moment: float = 1.0,
    diameter_mm: float = MAGNET_DIAMETER_MM,
    thickness_mm: float = MAGNET_THICKNESS_MM,
) -> np.ndarray:
    """Exact on-axis field of an axially magnetised cylinder, in model units.

    Reference solution used to validate `SubdipoleCylinder` and to quantify how
    wrong `PointDipole` is at this geometry. ``z_mm`` is measured from the
    magnet's centre along its magnetisation axis.

    Derived from the standard result B_z = (mu0 M / 2)[...] with
    M = moment / (pi R^2 h) and mu0 = 4 pi (the model-unit convention), which
    reduces to the 2m/z^3 dipole limit for z >> R, h.
    """
    z = np.asarray(z_mm, dtype=float)
    radius = diameter_mm / 2.0
    half = thickness_mm / 2.0
    near, far = z - half, z + half
    return (2.0 * moment / (radius**2 * thickness_mm)) * (
        far / np.sqrt(far**2 + radius**2) - near / np.sqrt(near**2 + radius**2)
    )
