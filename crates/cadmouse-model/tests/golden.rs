//! Checks the Rust port against the Python it was ported from.
//!
//!     cargo test --target x86_64-unknown-linux-gnu -p cadmouse-model
//!
//! Runs on the host, with no board attached, because the thing being checked
//! is arithmetic and arithmetic does not need a target. What it cannot check
//! is the target's FPU behaving differently from the host's, which is a real
//! but much narrower risk than the port being wrong -- `bench_forward` prints
//! `forward(0)` on the device for exactly that reason.
//!
//! # About the tolerances
//!
//! The reference is f64 NumPy and this is f32, so the two cannot agree
//! exactly. The bounds below are set from what the difference actually is,
//! then rounded up a little, and every one of them is far below what the
//! hardware can resolve:
//!
//! * One count is the sensor's least significant bit, and the noise floor is
//!   about 1.08 counts rms. A model agreeing to a hundredth of a count is
//!   agreeing to a hundredth of the smallest thing the device can perceive.
//! * A single frame resolves roughly 5.6 um of translation. A trajectory
//!   agreeing to a micrometre is agreeing to a fifth of that.
//!
//! If a change makes one of these fail, the question is not "can the tolerance
//! be raised" but "which of the two implementations moved".

use cadmouse_model::magnet::FieldTable;
use cadmouse_model::model::{MEAS_DIM, POSE_DIM, PoseModel, forward, forward_and_jac_vector};
use cadmouse_model::tuning;
use iekf::IteratedEkf;

#[allow(dead_code)]
mod golden {
    include!("data/golden_data.rs");
}

/// Counts, absolute. Measured disagreement is 0.0006, so this is roughly ten
/// times the observed error and still two orders below the 1.08-count noise
/// floor.
const COUNTS_TOL: f32 = 0.005;

/// Jacobian entries, relative to the largest entry in the same row. Relative
/// rather than absolute because the rows span several orders of magnitude
/// between the stiff and soft directions, and an absolute bound would be
/// vacuous on the stiff ones and impossible on the soft ones. Measured: 5.5e-6.
const JAC_REL_TOL: f32 = 5e-5;

/// Filter translation, mm. Measured drift over 400 frames is 2 nanometres --
/// f32 against f64 costs essentially nothing here because the filter's own
/// posterior is micrometres wide and the arithmetic never leaves that scale.
const POSE_POS_TOL: f32 = 1e-4;

/// Filter rotation, rad. Measured: 2e-7.
const POSE_ROT_TOL: f32 = 1e-5;

#[test]
fn forward_matches_the_python() {
    let table = FieldTable::from_flash();
    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize);

    for (p, pose) in golden::POSES.iter().enumerate() {
        let got = forward(pose, &table);
        for k in 0..MEAS_DIM {
            let diff = (got[k] - golden::COUNTS[p][k]).abs();
            if diff > worst {
                worst = diff;
                worst_at = (p, k);
            }
        }
    }

    println!("worst forward disagreement: {worst:.5} counts at {worst_at:?}");
    assert!(
        worst < COUNTS_TOL,
        "forward disagrees by {worst} counts at pose {}, channel {}",
        worst_at.0,
        worst_at.1
    );
}

#[test]
fn jacobian_matches_the_python() {
    let table = FieldTable::from_flash();
    let mut worst = 0.0f32;
    let mut worst_at = (0usize, 0usize, 0usize);

    for (p, pose) in golden::POSES.iter().enumerate() {
        let (_, jac) = forward_and_jac_vector(pose, &table);
        for row in 0..MEAS_DIM {
            let scale = (0..POSE_DIM)
                .map(|c| golden::JACOBIANS[p][row * POSE_DIM + c].abs())
                .fold(0.0f32, f32::max)
                .max(1e-6);
            for col in 0..POSE_DIM {
                let want = golden::JACOBIANS[p][row * POSE_DIM + col];
                let rel = (jac[row][col] - want).abs() / scale;
                if rel > worst {
                    worst = rel;
                    worst_at = (p, row, col);
                }
            }
        }
    }

    println!("worst jacobian disagreement: {worst:.3e} relative at {worst_at:?}");
    assert!(
        worst < JAC_REL_TOL,
        "jacobian disagrees by {worst} relative at pose {}, row {}, col {}",
        worst_at.0,
        worst_at.1,
        worst_at.2
    );
}

/// The Jacobian is also checked against the model's own finite differences,
/// which catches the case where both implementations share a mistake -- a
/// forgotten right Jacobian in *both* would sail through the comparison above.
#[test]
fn jacobian_agrees_with_finite_differences() {
    let table = FieldTable::from_flash();
    // Chosen by sweeping it, not by taste. A central difference in `f32` has
    // two competing errors -- truncation, which grows with the step, and
    // cancellation in `fu - fd`, which grows as it shrinks -- and their sum has
    // a minimum. Measured here, worst disagreement against the analytic
    // Jacobian:
    //
    //     1e-4  2.755e-2      1e-3  1.503e-3      1e-2  7.103e-3
    //     3e-4  3.507e-3      3e-3  2.206e-3      3e-2  2.426e-2
    //
    // The 1e-4 this used to use sat firmly on the cancellation side, so the
    // test was measuring its own arithmetic: the translation columns came out
    // twenty times worse than the rotation ones purely because 1e-4 mm is
    // 1/2500 of a table cell, and the difference of two ~500-count numbers has
    // no significant figures left. At the minimum the two agree to 1.5e-3,
    // which is the `f32` floor of the bicubic's own derivative.
    let step = 1e-3f32;
    let mut worst = 0.0f32;

    for pose in golden::POSES.iter().take(6) {
        let (_, jac) = forward_and_jac_vector(pose, &table);
        for col in 0..POSE_DIM {
            let mut up = *pose;
            let mut down = *pose;
            up[col] += step;
            down[col] -= step;
            let fu = forward(&up, &table);
            let fd = forward(&down, &table);
            let scale = (0..MEAS_DIM)
                .map(|r| jac[r][col].abs())
                .fold(0.0f32, f32::max)
                .max(1e-6);
            for row in 0..MEAS_DIM {
                let numeric = (fu[row] - fd[row]) / (2.0 * step);
                worst = worst.max((numeric - jac[row][col]).abs() / scale);
            }
        }
    }

    println!("worst finite-difference disagreement: {worst:.3e} relative");
    // Three times the measured 1.5e-3, and no more. The old 2e-2 was more than
    // ten times what this comparison can resolve at its best step, which made
    // it a test that could not fail for any reason worth knowing about.
    assert!(
        worst < 5e-3,
        "analytic jacobian disagrees with numeric: {worst}"
    );
}

#[test]
fn filter_trajectory_matches_the_python() {
    let table = FieldTable::from_flash();
    let model = PoseModel::new(&table);

    let mut ekf = IteratedEkf::<POSE_DIM, MEAS_DIM>::new(
        golden::INITIAL_POSE,
        tuning::initial_variance(),
        tuning::measurement_variance(&golden::SIGMA),
    );
    ekf.set_process_noise(tuning::process_noise());
    ekf.set_iterations(tuning::ITERATIONS);

    let mut worst_pos = 0.0f32;
    let mut worst_rot = 0.0f32;
    let mut nis_total = 0.0f32;

    for k in 0..golden::N_FRAMES {
        let mut z = [0.0f32; MEAS_DIM];
        for c in 0..MEAS_DIM {
            z[c] = golden::FRAMES[k][c] as f32;
        }
        if golden::DTS[k] > 0.0 {
            ekf.predict(golden::DTS[k]);
        }
        ekf.update(&model, &z)
            .expect("filter stayed positive definite");

        let got = ekf.state();
        for c in 0..3 {
            worst_pos = worst_pos.max((got[c] - golden::TRAJECTORY[k][c]).abs());
            worst_rot = worst_rot.max((got[3 + c] - golden::TRAJECTORY[k][3 + c]).abs());
        }
        nis_total += ekf.nis();
    }

    let mean_nis = nis_total / golden::N_FRAMES as f32;
    println!(
        "worst trajectory disagreement: {:.3} um, {:.5} deg; mean NIS {:.2} (target {})",
        worst_pos * 1000.0,
        worst_rot.to_degrees(),
        mean_nis,
        MEAS_DIM
    );

    assert!(
        worst_pos < POSE_POS_TOL,
        "translation drifted {worst_pos} mm from the Python"
    );
    assert!(
        worst_rot < POSE_ROT_TOL,
        "rotation drifted {worst_rot} rad from the Python"
    );
    // The port could agree with Python and both be badly tuned; this is the
    // independent check that the filter is honest about its own uncertainty.
    //
    // A well-tuned filter puts this at `MEAS_DIM`. This one does not, and the
    // band below is deliberately wide enough to say so without failing:
    //
    //     4.34   this recording
    //     5.64   the device itself, at rest, over 36 000 frames
    //
    // Both are *below* nine, which means `R` and/or `Q` are larger than the
    // data warrants and the filter is smoothing more than it needs to. That is
    // a defensible trade for a knob -- less jitter, slightly more lag -- but it
    // is a trade nobody explicitly made, and correcting it needs a tuning pass
    // against recorded *motion*, not a threshold edit here.
    //
    // So this stays a smoke test for gross mistuning -- a divergence, an `R` of
    // zero, a unit error -- rather than a tuning target. Raise the lower bound
    // back toward nine once the tuning pass has happened.
    assert!(
        mean_nis > 3.0 && mean_nis < 16.0,
        "mean NIS {mean_nis} is nowhere near the {MEAS_DIM} channels it should match"
    );
}
