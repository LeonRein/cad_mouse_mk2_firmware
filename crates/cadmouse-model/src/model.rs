//! The measurement function: six degrees of freedom in, nine counts out.
//!
//! A direct port of `scripts/cadmouse/model.py`, and the reason the benchmark
//! exists: this is roughly ninety per cent of the filter's cost, whichever
//! filter ships. See that file for why the sum runs over all three magnets
//! rather than just the one above each sensor (the two far magnets are worth
//! 8-14 counts against a one-count noise floor).
//!
//! [`forward_and_jac`] differentiates with respect to a *local* perturbation:
//! translation adds in the board frame, rotation right-multiplies as
//! `R <- R exp(delta)`. That is what an iterated EKF wants, and it avoids
//! differentiating Rodrigues' formula.

use crate::generated as consts;
use crate::magnet::{sqrtf, FieldTable};

pub const N_SENSORS: usize = 3;
pub const N_MAGNETS: usize = 3;
pub const MEAS_DIM: usize = 9;
pub const POSE_DIM: usize = 6;

/// Pose: three translations in millimetres, then a rotation vector in radians
/// about the knob's neutral centre.
pub type Pose = [f32; POSE_DIM];

#[inline(always)]
fn dot(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline(always)]
fn matvec(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [f32; 3] {
    [dot(&m[0], v), dot(&m[1], v), dot(&m[2], v)]
}

/// Rodrigues' formula, with the small-angle limit handled.
///
/// The knob never exceeds a few degrees, so the series would do -- but a
/// rotation that quietly stops being orthonormal is a miserable thing to debug
/// on a target with no debugger attached.
///
/// Relocated to SRAM: see [the note on `field_and_grad`](field_and_grad).
#[unsafe(link_section = ".data")]
#[inline(never)]
pub fn rotation_from_rotvec(rv: &[f32; 3]) -> [[f32; 3]; 3] {
    let theta2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
    if theta2 < 1e-16 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let theta = sqrtf(theta2);
    let (s, c) = (sinf(theta), cosf(theta));
    let k = [rv[0] / theta, rv[1] / theta, rv[2] / theta];
    let v = 1.0 - c;
    [
        [
            c + k[0] * k[0] * v,
            k[0] * k[1] * v - k[2] * s,
            k[0] * k[2] * v + k[1] * s,
        ],
        [
            k[1] * k[0] * v + k[2] * s,
            c + k[1] * k[1] * v,
            k[1] * k[2] * v - k[0] * s,
        ],
        [
            k[2] * k[0] * v - k[1] * s,
            k[2] * k[1] * v + k[0] * s,
            c + k[2] * k[2] * v,
        ],
    ]
}

#[inline]
fn sinf(x: f32) -> f32 {
    libm::sinf(x)
}

#[inline]
fn cosf(x: f32) -> f32 {
    libm::cosf(x)
}

/// Field in the board frame from one magnet, per unit moment, plus the
/// gradients with respect to the offset vector and the magnetisation axis.
///
/// The chain rule is spelled out in the Python; writing `zc` for the axial
/// coordinate and `e` for the radial unit vector, `B = b_rho e + b_z axis`,
/// and both `rho` and `e` depend on `delta` and `axis` in turn. `e . axis == 0`
/// kills the cross terms.
///
/// # Why `.data` and `inline(never)`
///
/// This crate's hot path executes from external QSPI flash through a small XIP
/// cache, and the working set does not fit. `.data` is copied to SRAM by the
/// startup code, so placing a function there takes it out of that contest;
/// doing it for this one, [`FieldTable::sample`], [`rotation_from_rotvec`] and
/// [`right_jacobian_so3`] cut a filter step by 35 % on target.
///
/// `inline(never)` is the load-bearing half. Forcing these inline instead was
/// measured **14 % slower**: this is called nine times per Jacobian, so
/// inlining duplicates it nine times and grows the very working set the
/// relocation exists to shrink. One shared copy in SRAM beats nine in flash.
/// Do not "optimise" the attribute away -- see `scripts/README.md`.
#[unsafe(link_section = ".data")]
#[inline(never)]
fn field_and_grad(
    table: &FieldTable,
    delta: &[f32; 3],
    axis: &[f32; 3],
    want_grad: bool,
) -> ([f32; 3], [[f32; 3]; 3], [[f32; 3]; 3]) {
    let zc = dot(delta, axis);
    let radial = [
        delta[0] - zc * axis[0],
        delta[1] - zc * axis[1],
        delta[2] - zc * axis[2],
    ];
    let rho2 = radial[0] * radial[0] + radial[1] * radial[1] + radial[2] * radial[2];
    let rho = sqrtf(rho2);
    // On axis the radial field is zero by symmetry, so the direction being
    // undefined there is harmless as long as it is finite.
    let inv_rho = if rho > 1e-9 { 1.0 / rho } else { 0.0 };
    let e = [radial[0] * inv_rho, radial[1] * inv_rho, radial[2] * inv_rho];

    let s = table.sample(rho, zc);
    let b = [
        s.b_rho * e[0] + s.b_z * axis[0],
        s.b_rho * e[1] + s.b_z * axis[1],
        s.b_rho * e[2] + s.b_z * axis[2],
    ];

    let mut grad_delta = [[0.0f32; 3]; 3];
    let mut grad_axis = [[0.0f32; 3]; 3];
    if want_grad {
        let ga = [
            s.d_rho_d_rho * e[0] + s.d_rho_d_z * axis[0],
            s.d_rho_d_rho * e[1] + s.d_rho_d_z * axis[1],
            s.d_rho_d_rho * e[2] + s.d_rho_d_z * axis[2],
        ];
        let gz = [
            s.d_z_d_rho * e[0] + s.d_z_d_z * axis[0],
            s.d_z_d_rho * e[1] + s.d_z_d_z * axis[1],
            s.d_z_d_rho * e[2] + s.d_z_d_z * axis[2],
        ];
        let spread = s.b_rho * inv_rho;
        for o in 0..3 {
            for i in 0..3 {
                let ident = if o == i { 1.0 } else { 0.0 };
                grad_delta[o][i] = e[o] * ga[i]
                    + spread * (ident - e[o] * e[i] - axis[o] * axis[i])
                    + axis[o] * gz[i];
                grad_axis[o][i] = e[o] * (-zc * s.d_rho_d_rho * e[i] + s.d_rho_d_z * delta[i])
                    + spread * (-zc * (ident - e[o] * e[i]) - axis[o] * delta[i])
                    + axis[o] * (-zc * s.d_z_d_rho * e[i] + s.d_z_d_z * delta[i])
                    + s.b_z * ident;
            }
        }
    }
    (b, grad_delta, grad_axis)
}

/// Predicted counts, channel order MAG1/2/3 x,y,z.
pub fn forward(pose: &Pose, table: &FieldTable) -> [f32; MEAS_DIM] {
    let rot = rotation_from_rotvec(&[pose[3], pose[4], pose[5]]);

    let mut centres = [[0.0f32; 3]; N_MAGNETS];
    let mut axes = [[0.0f32; 3]; N_MAGNETS];
    for j in 0..N_MAGNETS {
        let p = matvec(&rot, &consts::MAGNET_POS[j]);
        centres[j] = [p[0] + pose[0], p[1] + pose[1], p[2] + pose[2]];
        axes[j] = matvec(&rot, &consts::MAGNET_AXIS[j]);
    }

    let mut out = [0.0f32; MEAS_DIM];
    for s in 0..N_SENSORS {
        let mut tesla = [0.0f32; 3];
        for j in 0..N_MAGNETS {
            let delta = [
                consts::SENSOR_POS[s][0] - centres[j][0],
                consts::SENSOR_POS[s][1] - centres[j][1],
                consts::SENSOR_POS[s][2] - centres[j][2],
            ];
            let (b, _, _) = field_and_grad(table, &delta, &axes[j], false);
            let m = consts::MAGNET_MOMENT[j];
            for a in 0..3 {
                tesla[a] += m * b[a];
            }
        }
        for a in 0..3 {
            out[3 * s + a] = tesla[a] * consts::TESLA_TO_COUNTS + consts::SENSOR_OFFSET[s][a];
        }
    }
    out
}

/// Predicted counts and `d(counts)/d(local pose perturbation)`.
#[inline(always)]
pub fn forward_and_jac(
    pose: &Pose,
    table: &FieldTable,
) -> ([f32; MEAS_DIM], [[f32; POSE_DIM]; MEAS_DIM]) {
    let rot = rotation_from_rotvec(&[pose[3], pose[4], pose[5]]);

    let mut centres = [[0.0f32; 3]; N_MAGNETS];
    let mut axes = [[0.0f32; 3]; N_MAGNETS];
    // d(centre)/d(rotation) = -R [p]_x and d(axis)/d(rotation) = -R [n]_x.
    let mut lever = [[[0.0f32; 3]; 3]; N_MAGNETS];
    let mut spin = [[[0.0f32; 3]; 3]; N_MAGNETS];
    for j in 0..N_MAGNETS {
        let p = matvec(&rot, &consts::MAGNET_POS[j]);
        centres[j] = [p[0] + pose[0], p[1] + pose[1], p[2] + pose[2]];
        axes[j] = matvec(&rot, &consts::MAGNET_AXIS[j]);
        lever[j] = mul_skew(&rot, &consts::MAGNET_POS[j]);
        spin[j] = neg(&mul_skew(&rot, &consts::MAGNET_AXIS[j]));
    }

    let mut counts = [0.0f32; MEAS_DIM];
    let mut jac = [[0.0f32; POSE_DIM]; MEAS_DIM];

    for s in 0..N_SENSORS {
        let mut tesla = [0.0f32; 3];
        let mut d_trans = [[0.0f32; 3]; 3];
        let mut d_rot = [[0.0f32; 3]; 3];

        for j in 0..N_MAGNETS {
            let delta = [
                consts::SENSOR_POS[s][0] - centres[j][0],
                consts::SENSOR_POS[s][1] - centres[j][1],
                consts::SENSOR_POS[s][2] - centres[j][2],
            ];
            let (b, gd, ga) = field_and_grad(table, &delta, &axes[j], true);
            let m = consts::MAGNET_MOMENT[j];
            for a in 0..3 {
                tesla[a] += m * b[a];
                for i in 0..3 {
                    // delta depends on translation as -I, identically per magnet.
                    d_trans[a][i] -= m * gd[a][i];
                    let mut acc = 0.0;
                    for k in 0..3 {
                        acc += gd[a][k] * lever[j][k][i] + ga[a][k] * spin[j][k][i];
                    }
                    d_rot[a][i] += m * acc;
                }
            }
        }

        for a in 0..3 {
            let scale = consts::TESLA_TO_COUNTS;
            counts[3 * s + a] = scale * tesla[a] + consts::SENSOR_OFFSET[s][a];
            for i in 0..3 {
                jac[3 * s + a][i] = scale * d_trans[a][i];
                jac[3 * s + a][3 + i] = scale * d_rot[a][i];
            }
        }
    }
    (counts, jac)
}

/// Right Jacobian of SO(3), relating the two rotation conventions.
///
/// [`forward_and_jac`] differentiates a *local* perturbation, which avoids
/// differentiating Rodrigues' formula. A filter carrying a plain rotation
/// vector in its state varies the vector itself. The two are related by
/// `exp((theta + d)^) ~= exp(theta^) exp((Jr(theta) d)^)`, so a Jacobian in the
/// local convention becomes one in the vector convention by right-multiplying
/// the rotation block by this matrix.
///
/// At the few degrees this mechanism reaches `Jr` differs from the identity by
/// under a percent -- but it is the difference between a filter that converges
/// quadratically and one that limps, and it costs almost nothing.
///
/// Relocated to SRAM: see [the note on `field_and_grad`](field_and_grad).
#[unsafe(link_section = ".data")]
#[inline(never)]
pub fn right_jacobian_so3(rv: &[f32; 3]) -> [[f32; 3]; 3] {
    let theta2 = rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2];
    let kx = skew(rv);
    let kx2 = matmul3(&kx, &kx);

    // Below this the closed form loses the cancellation in (1 - cos)/theta^2
    // to f32 rounding; the series is exact enough well past where it is used.
    let (a, b) = if theta2 < 1e-8 {
        (0.5, 1.0 / 6.0)
    } else {
        let theta = sqrtf(theta2);
        (
            (1.0 - cosf(theta)) / theta2,
            (theta - sinf(theta)) / (theta2 * theta),
        )
    };

    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let ident = if r == c { 1.0 } else { 0.0 };
            out[r][c] = ident - a * kx[r][c] + b * kx2[r][c];
        }
    }
    out
}

/// Predicted counts and `d(counts)/d(pose vector)`.
///
/// The convention a filter carrying a six-element state wants;
/// [`forward_and_jac`] uses the local one.
#[inline(always)]
pub fn forward_and_jac_vector(
    pose: &Pose,
    table: &FieldTable,
) -> ([f32; MEAS_DIM], [[f32; POSE_DIM]; MEAS_DIM]) {
    let (counts, local) = forward_and_jac(pose, table);
    let right = right_jacobian_so3(&[pose[3], pose[4], pose[5]]);

    let mut jac = local;
    for row in jac.iter_mut() {
        let rot = [row[3], row[4], row[5]];
        for c in 0..3 {
            row[3 + c] = rot[0] * right[0][c] + rot[1] * right[1][c] + rot[2] * right[2][c];
        }
    }
    (counts, jac)
}

/// The knob as an [`iekf::MeasurementModel`]: six-element pose in, nine counts
/// out.
///
/// Borrows the table rather than owning it so the filter does not care whether
/// it is reading flash or the RAM copy.
pub struct PoseModel<'a> {
    pub table: &'a FieldTable,
}

impl<'a> PoseModel<'a> {
    pub fn new(table: &'a FieldTable) -> Self {
        Self { table }
    }
}

impl iekf::MeasurementModel<POSE_DIM, MEAS_DIM> for PoseModel<'_> {
    #[inline]
    fn predict_and_jacobian(
        &self,
        state: &[f32; POSE_DIM],
    ) -> ([f32; MEAS_DIM], [[f32; POSE_DIM]; MEAS_DIM]) {
        forward_and_jac_vector(state, self.table)
    }
}

#[inline]
fn skew(v: &[f32; 3]) -> [[f32; 3]; 3] {
    [
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ]
}

#[inline]
fn matmul3(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

#[inline]
fn mul_skew(m: &[[f32; 3]; 3], v: &[f32; 3]) -> [[f32; 3]; 3] {
    // m * [v]_x, where [v]_x u == cross(v, u).
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        out[r][0] = m[r][1] * v[2] - m[r][2] * v[1];
        out[r][1] = m[r][2] * v[0] - m[r][0] * v[2];
        out[r][2] = m[r][0] * v[1] - m[r][1] * v[0];
    }
    out
}

#[inline]
fn neg(m: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = -m[r][c];
        }
    }
    out
}
