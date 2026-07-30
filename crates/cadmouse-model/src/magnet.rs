//! Field of one axially magnetised cylinder, from a 2-D interpolation table.
//!
//! A direct port of `scripts/cadmouse/magnet.py`. It has to stay a direct port:
//! the host tooling validates against this arithmetic, so any divergence turns
//! the golden vectors into a comparison of two different functions rather than
//! a check of one.
//!
//! Everything is `f32`. The Cortex-M33 has a single-precision FPU and `f64` is
//! soft-float, so double precision here would cost roughly two orders of
//! magnitude for accuracy the sensor cannot resolve.
//!
//! Lengths are millimetres, fields are tesla per unit magnetic moment; the
//! caller applies the moment, which is what keeps the reversed third magnet an
//! ordinary negative number.

use crate::generated as consts;

/// `(B_rho, B_z)` on a uniform `(rho, z)` grid, per unit moment.
pub struct FieldTable {
    b_rho: &'static [f32],
    b_z: &'static [f32],
}

#[repr(C, align(4))]
struct Aligned<const N: usize>([u8; N]);

/// The table lives in flash and is read through the XIP cache. Alignment is
/// forced because `include_bytes!` yields a `[u8]` with no guarantee, and a
/// misaligned `f32` read on this core faults rather than being merely slow.
static TABLE_FLASH: Aligned<{ consts::TABLE_BYTES }> =
    Aligned(*include_bytes!("../gen/field_table.bin"));

impl FieldTable {
    /// Borrow the table directly out of flash.
    pub fn from_flash() -> Self {
        Self::from_bytes(&TABLE_FLASH.0)
    }

    /// Build a table over a caller-provided buffer, for measuring the table in
    /// RAM against the table in flash. Both paths run the same interpolation.
    ///
    /// # Panics
    /// If `buffer` is not exactly [`consts::TABLE_BYTES`] long.
    pub fn copy_into(buffer: &'static mut [u8]) -> Self {
        buffer.copy_from_slice(&TABLE_FLASH.0);
        Self::from_bytes(buffer)
    }

    fn from_bytes(bytes: &'static [u8]) -> Self {
        let (head, floats, tail) = unsafe { bytes.align_to::<f32>() };
        debug_assert!(head.is_empty() && tail.is_empty(), "table is misaligned");
        let n = consts::N_RHO * consts::N_Z;
        FieldTable {
            b_rho: &floats[..n],
            b_z: &floats[n..2 * n],
        }
    }
}

/// Value and both partial derivatives of one tabulated field component.
#[derive(Clone, Copy, Default)]
pub struct Sample {
    pub b_rho: f32,
    pub b_z: f32,
    pub d_rho_d_rho: f32,
    pub d_rho_d_z: f32,
    pub d_z_d_rho: f32,
    pub d_z_d_z: f32,
}

#[inline(always)]
fn cubic_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2,
    ]
}

#[inline(always)]
fn cubic_weight_derivs(t: f32) -> [f32; 4] {
    let t2 = t * t;
    [
        -1.5 * t2 + 2.0 * t - 0.5,
        4.5 * t2 - 5.0 * t,
        -4.5 * t2 + 4.0 * t + 0.5,
        1.5 * t2 - t,
    ]
}

/// Stencil base index, interpolation fraction, and whether the derivative is
/// live.
///
/// Two separate clamps, and conflating them is a trap that costs tens of counts
/// on the radial axis -- which is the rest pose. The *stencil* is clamped so it
/// stays inside the array, leaving the fraction outside `[0, 1]` in the first
/// and last cell; the *query* is clamped only when it leaves the grid entirely.
/// Past the grid the interpolant is flat, so its derivative reads zero rather
/// than handing the filter a phantom gradient pushing it further out.
#[inline(always)]
fn stencil(coord: f32, origin: f32, step: f32, count: usize) -> (usize, f32, f32) {
    let raw = (coord - origin) / step;
    let hi = (count - 1) as f32;
    let u = if raw < 0.0 {
        0.0
    } else if raw > hi {
        hi
    } else {
        raw
    };
    let base = u as i32; // u >= 0, so truncation is floor
    let i0 = base.saturating_sub(1).min(count as i32 - 4).max(0) as usize;
    let live = if raw == u { 1.0 } else { 0.0 };
    (i0, u - (i0 + 1) as f32, live)
}

impl FieldTable {
    /// Bicubic (Keys' cubic convolution, a = -1/2) with derivatives.
    ///
    /// Both components share the stencil and the weights, so evaluating them
    /// together costs barely more than one -- and the derivatives fall out of
    /// the same 4x4 gather, which is what makes an analytic measurement
    /// Jacobian nearly free and an iterated EKF cheaper than a UKF here.
    ///
    /// Relocated to SRAM, and deliberately not inlined despite being called
    /// nine times per Jacobian -- that is exactly why. See the note on
    /// `model::field_and_grad`.
    #[unsafe(link_section = ".data")]
    #[inline(never)]
    pub fn sample(&self, rho: f32, z: f32) -> Sample {
        let (i0, tu, live_u) = stencil(rho, consts::RHO0, consts::D_RHO, consts::N_RHO);
        let (j0, tv, live_v) = stencil(z, consts::Z0, consts::D_Z, consts::N_Z);

        let wu = cubic_weights(tu);
        let wv = cubic_weights(tv);
        let du = cubic_weight_derivs(tu);
        let dv = cubic_weight_derivs(tv);

        let mut out = Sample::default();
        let mut d_rho_du = 0.0f32;
        let mut d_rho_dv = 0.0f32;
        let mut d_z_du = 0.0f32;
        let mut d_z_dv = 0.0f32;

        for i in 0..4 {
            let row = (i0 + i) * consts::N_Z;
            for j in 0..4 {
                let idx = row + j0 + j;
                let f_rho = self.b_rho[idx];
                let f_z = self.b_z[idx];
                let w = wu[i] * wv[j];
                out.b_rho += w * f_rho;
                out.b_z += w * f_z;
                d_rho_du += du[i] * wv[j] * f_rho;
                d_rho_dv += wu[i] * dv[j] * f_rho;
                d_z_du += du[i] * wv[j] * f_z;
                d_z_dv += wu[i] * dv[j] * f_z;
            }
        }

        out.d_rho_d_rho = live_u * d_rho_du / consts::D_RHO;
        out.d_rho_d_z = live_v * d_rho_dv / consts::D_Z;
        out.d_z_d_rho = live_u * d_z_du / consts::D_RHO;
        out.d_z_d_z = live_v * d_z_dv / consts::D_Z;
        out
    }
}

/// Square root. **Currently a software implementation**, and a known cost.
///
/// The core has a `VSQRT.F32` instruction and this does not use it. `core` has
/// no `f32::sqrt` (still `std`-only as of 1.97), and `libm::sqrtf` compiles to
/// a call rather than the instruction -- confirmed in the disassembly, with
/// neither the default features nor `+fp-armv8d16sp` changing that.
///
/// Reaching the instruction needs inline asm with the `sreg` register class,
/// which Rust gates behind `target-feature=+vfp2`. **Do not enable that
/// feature.** It claims *double*-precision hardware, which this FPU (FPv5-SP)
/// does not have; `libm::sinf` then emits `.f64` instructions and the board
/// takes an undefined-instruction UsageFault on the first rotation. That was
/// tried, and it faulted on the first flash.
///
/// Worth revisiting only if the benchmark says it matters: there are nine
/// square roots per measurement, so this is a few hundred cycles. Getting them
/// back means either writing f32 `sin`/`cos` to eliminate every `libm` f64 path
/// before enabling `+vfp2`, or an `asm!` formulation that avoids `sreg`.
#[inline(always)]
pub fn sqrtf(x: f32) -> f32 {
    libm::sqrtf(x)
}
