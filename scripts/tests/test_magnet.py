"""Gates for the magnet model.

The chain of trust runs: closed-form on-axis solution -> charge-sheet
quadrature -> interpolation table -> the scalar interpolator that gets ported.
Each link is checked against the one before it, because a table generated from
a subtly wrong integral would still interpolate beautifully.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cadmouse import magnet
from cadmouse.geometry import COUNTS_PER_MT, MAGNET_DIAMETER, MAGNET_HEIGHT, MOMENT_N35

TESLA_TO_COUNTS = 1e3 * COUNTS_PER_MT


def analytic_on_axis(gap_mm: float) -> float:
    """Closed-form B_z per unit moment on the axis, ``gap`` from the near face.

    For a uniformly magnetised cylinder of radius R and length L,
    ``B = (Br/2) [ (z+L)/sqrt((z+L)^2+R^2) - z/sqrt(z^2+R^2) ]``. Dividing by
    the moment ``(Br/mu0) V`` cancels the remanence, leaving pure geometry.
    """
    radius = MAGNET_DIAMETER / 2.0
    length = MAGNET_HEIGHT
    shape = (gap_mm + length) / math.sqrt((gap_mm + length) ** 2 + radius**2) - gap_mm / math.sqrt(
        gap_mm**2 + radius**2
    )
    br = 1.0  # cancels; carried explicitly for readability
    volume = math.pi * (radius / 1e3) ** 2 * (length / 1e3)
    moment = br / (4e-7 * math.pi) * volume
    return (br / 2.0) * shape / moment


@pytest.mark.parametrize("gap", [3.0, 6.0, 9.0, 15.0])
def test_quadrature_matches_closed_form(gap):
    """The charge-sheet integral is the reference, so it must be exact."""
    z = gap + MAGNET_HEIGHT / 2.0
    _, b_z = magnet.field_axisym_exact(np.array(0.0), np.array(z))
    assert b_z == pytest.approx(analytic_on_axis(gap), rel=2e-4)


def test_radial_field_vanishes_on_axis():
    b_rho, _ = magnet.field_axisym_exact(np.zeros(5), np.linspace(-15.0, -5.0, 5))
    assert np.all(np.abs(b_rho) < 1e-12)


def test_dipole_is_close_but_not_close_enough():
    """A point dipole is far better here than the usual rule of thumb implies.

    Length equal to twice the radius nearly cancels the leading multipole
    correction, so the error at the operating gap is a couple of percent rather
    than tens. That is still ~15 counts against a 1-count noise floor, which is
    precisely why the table exists.
    """
    z = 6.0 + MAGNET_HEIGHT / 2.0
    _, exact = magnet.field_axisym_exact(np.array(0.0), np.array(z))
    approx = magnet.field_dipole(np.array([0.0, 0.0, z]), np.array([0.0, 0.0, 1.0]))[2]
    error = abs(approx / exact - 1.0)
    assert error < 0.05, "dipole should be within a few percent on axis"
    counts = abs(approx - exact) * MOMENT_N35 * TESLA_TO_COUNTS
    assert counts > 5.0, "if the dipole were this good the table would be pointless"


# ---------------------------------------------------------------- table


def test_table_covers_the_reachable_geometry(table):
    """Every sensor-magnet pair the mechanism can produce must be in range."""
    assert table.rho0 <= -2.0 * table.d_rho, "rho = 0 must be an interior point"
    assert table.rho_max >= 32.0, "the far magnets sit ~29 mm away and still matter"
    assert table.z0 <= -18.0
    assert table.z_max >= -5.0
    assert table.b_rho.dtype == np.float32, "must match the RP2350's single-precision FPU"


def test_table_size_is_reasonable(table):
    assert table.nbytes() < 512 * 1024, f"table is {table.nbytes() / 1024:.0f} kB"


def test_table_reproduces_the_exact_field(table):
    """Interpolation error over the region the mechanism can actually reach.

    Expressed in counts, because that is the only unit in which "good enough"
    means anything. The axis is included deliberately: rho = 0 is the rest
    pose, not an edge case, and a boundary-handling mistake there is worth tens
    of counts against a one-count noise floor.
    """
    rng = np.random.default_rng(7)
    rho = rng.uniform(0.0, 30.0, size=20000)
    z = rng.uniform(-13.0, -6.0, size=20000)

    exact_rho, exact_z = magnet.field_axisym_exact(rho, z)
    got_rho, got_z, *_ = magnet.sample(table, rho, z)

    err = np.hypot(got_rho - exact_rho, got_z - exact_z) * MOMENT_N35 * TESLA_TO_COUNTS
    assert np.median(err) < 0.01
    assert np.percentile(err, 95) < 0.1
    assert err.max() < 1.0, f"worst interpolation error {err.max():.2f} counts"


def test_table_is_accurate_on_the_axis(table):
    """A magnet directly over its own sensor -- i.e. the rest pose.

    Called out separately because it is the one place a plausible-looking
    boundary clamp does real damage, and a uniformly sampled test can miss it.
    """
    z = np.linspace(-13.0, -6.0, 200)
    rho = np.zeros_like(z)
    exact_rho, exact_z = magnet.field_axisym_exact(rho, z)
    got_rho, got_z, *_ = magnet.sample(table, rho, z)

    assert np.abs(got_rho).max() * MOMENT_N35 * TESLA_TO_COUNTS < 0.05, "B_rho must vanish on axis"
    err = np.abs(got_z - exact_z) * MOMENT_N35 * TESLA_TO_COUNTS
    assert err.max() < 0.2, f"on-axis error {err.max():.3f} counts"


def test_table_matches_exact_field_end_to_end(table, nominal, envelope_poses):
    """Gate for stage 1: the assembled nine-channel prediction, in counts.

    This is the number that matters. Everything upstream can be individually
    fine while the composition -- frame decomposition, cross-talk summation,
    the reversed third magnet -- quietly is not.
    """
    from cadmouse import forward
    from cadmouse.model import TESLA_TO_COUNTS as MODEL_SCALE
    from cadmouse.model import _geometry

    worst = 0.0
    for pose in envelope_poses[:60]:
        _, _, axes, delta = _geometry(pose, nominal)

        axial = np.einsum("sjk,jk->sj", delta, axes)
        radial_vec = delta - axial[..., None] * axes[None, :, :]
        radial = np.linalg.norm(radial_vec, axis=-1)
        b_rho, b_z = magnet.field_axisym_exact(radial, axial)

        with np.errstate(invalid="ignore", divide="ignore"):
            unit_radial = np.where(
                radial[..., None] > 1e-9, radial_vec / np.maximum(radial, 1e-30)[..., None], 0.0
            )
        exact = (
            nominal.magnet_moment[None, :, None]
            * (b_rho[..., None] * unit_radial + b_z[..., None] * axes[None, :, :])
        ).sum(axis=1)
        exact_counts = (exact * MODEL_SCALE).ravel()

        worst = max(worst, float(np.abs(forward(pose, nominal, table) - exact_counts).max()))

    assert worst < 1.0, f"worst end-to-end table error {worst:.3f} counts"


def test_scalar_and_vectorised_interpolators_agree(table):
    """The golden vectors are only worth anything if these are one function.

    Deliberately includes the boundary cells and points off the grid entirely:
    those are where the two implementations' index arithmetic can diverge, and
    where the interesting bugs have actually been.
    """
    rng = np.random.default_rng(11)
    rho = np.concatenate(
        [rng.uniform(-1.0, 34.0, size=300), [-1.0, -0.9, 0.0, 0.1, 33.9, 34.0, -8.0, 90.0]]
    )
    z = np.concatenate(
        [rng.uniform(-22.0, -3.0, size=300), [-22.0, -21.9, -3.1, -3.0, -50.0, 10.0, -9.0, -9.0]]
    )

    batched = magnet.sample(table, rho, z)
    for k in range(rho.size):
        one = magnet.sample_scalar(table, float(rho[k]), float(z[k]))
        for got, expected in zip(one, (arr[k] for arr in batched)):
            assert got == pytest.approx(expected, rel=1e-9, abs=1e-18)


def test_interpolator_clamps_outside_the_grid(table):
    """Out of range must be merely wrong, never infinite.

    A sigma point can wander outside the tabulated region during a filter
    transient; a cubic extrapolated far enough will produce numbers that poison
    the covariance permanently.
    """
    for rho, z in [(-5.0, -9.0), (60.0, -9.0), (5.0, 40.0), (5.0, -60.0)]:
        values = magnet.sample(table, np.array(rho), np.array(z))
        assert all(np.isfinite(v).all() for v in values)


def test_clamped_derivatives_are_zero(table):
    """Beyond the grid the interpolant is flat, so its slope must read flat.

    Reporting the cubic's slope out there instead would hand an optimiser or a
    filter a phantom gradient pushing it further out -- a wrong *value* is
    recoverable, a wrong gradient is what makes a divergence permanent.
    """
    # Off the radial end: the rho derivatives go, the z derivatives stay.
    _, _, dbr_dr, dbr_dz, dbz_dr, dbz_dz = magnet.sample(
        table, np.array(60.0), np.array(-9.0)
    )
    assert dbr_dr == 0.0 and dbz_dr == 0.0
    assert dbr_dz != 0.0 or dbz_dz != 0.0

    # Off the axial end, and the roles swap.
    _, _, dbr_dr, dbr_dz, dbz_dr, dbz_dz = magnet.sample(
        table, np.array(5.0), np.array(-60.0)
    )
    assert dbr_dz == 0.0 and dbz_dz == 0.0

    # In range, nothing is suppressed.
    _, _, dbr_dr, _, _, dbz_dz = magnet.sample(table, np.array(5.0), np.array(-9.0))
    assert dbr_dr != 0.0 and dbz_dz != 0.0


def test_table_derivatives_match_finite_differences(table):
    """The analytic Jacobian is built on these, so they carry real weight."""
    rng = np.random.default_rng(3)
    rho = rng.uniform(2.0, 28.0, size=200)
    z = rng.uniform(-17.0, -6.0, size=200)
    step = 1e-4

    _, _, dbr_dr, dbr_dz, dbz_dr, dbz_dz = magnet.sample(table, rho, z)

    def value(r, zz):
        a, b, *_ = magnet.sample(table, r, zz)
        return a, b

    plus_r, plus_r_z = value(rho + step, z)
    minus_r, minus_r_z = value(rho - step, z)
    plus_z, plus_z_z = value(rho, z + step)
    minus_z, minus_z_z = value(rho, z - step)

    scale = np.abs(dbr_dr).max()
    assert np.allclose((plus_r - minus_r) / (2 * step), dbr_dr, atol=1e-6 * scale)
    assert np.allclose((plus_z - minus_z) / (2 * step), dbr_dz, atol=1e-6 * scale)
    scale = np.abs(dbz_dr).max()
    assert np.allclose((plus_r_z - minus_r_z) / (2 * step), dbz_dr, atol=1e-6 * scale)
    assert np.allclose((plus_z_z - minus_z_z) / (2 * step), dbz_dz, atol=1e-6 * scale)


# ---------------------------------------------------------------- cartesian


def test_field_gradients_match_finite_differences(table):
    rng = np.random.default_rng(5)
    step = 1e-5
    for _ in range(20):
        delta = np.array([rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-16, -6)])
        if np.hypot(delta[0], delta[1]) < 1.0:
            delta[0] += 2.0
        axis = np.array([0.03, -0.02, 1.0])
        axis /= np.linalg.norm(axis)
        moment = 0.08

        _, grad_delta, grad_axis = magnet.field_and_grad(table, delta, axis, moment)

        for k in range(3):
            bump = np.zeros(3)
            bump[k] = step
            hi = magnet.field(table, delta + bump, axis, moment)
            lo = magnet.field(table, delta - bump, axis, moment)
            assert np.allclose((hi - lo) / (2 * step), grad_delta[:, k], rtol=1e-4, atol=1e-9)

            hi = magnet.field(table, delta, axis + bump, moment)
            lo = magnet.field(table, delta, axis - bump, moment)
            assert np.allclose((hi - lo) / (2 * step), grad_axis[:, k], rtol=1e-4, atol=1e-9)


def test_field_is_linear_in_moment(table):
    """The calibration's warm-up phase depends on this, so pin it down."""
    delta = np.array([0.5, 0.0, -9.0])
    axis = np.array([0.0, 0.0, 1.0])
    one = magnet.field(table, delta, axis, 1.0)
    scaled = magnet.field(table, delta, axis, -2.5)
    assert np.allclose(scaled, -2.5 * one)
