//! Filter tuning, carried over from `scripts/cadmouse/filter.py`.
//!
//! These are design constants rather than per-device ones -- they describe how
//! the knob moves and how much the estimate is allowed to be pushed around,
//! not what this particular unit's magnets are doing. Per-device numbers live
//! in [`crate::generated`] (fitted on the host) or come from the on-device rest
//! calibration (measured at boot).

use crate::model::{MEAS_DIM, POSE_DIM};

/// Random-walk process noise on translation, mm^2/s.
///
/// Tuned on the held-out `free` segment against the normalised innovation
/// squared, which is the only thing available: with no ground truth a filter
/// can be checked for honesty about its own uncertainty but not for accuracy.
///
/// Getting this wrong is not a mild mistuning. Three orders of magnitude too
/// large puts the estimate millimetres from the mean -- outside anything the
/// mechanism can reach, and outside the field table.
pub const Q_POS: f32 = 0.02;

/// Random-walk process noise on rotation, rad^2/s.
pub const Q_ROT: f32 = 3.0e-4;

/// Initial translation variance, mm^2. Modest on purpose: the filter starts
/// from the measured rest pose, which is good to a few micrometres.
pub const INITIAL_POS_VAR: f32 = 0.05 * 0.05;

/// Initial rotation variance, rad^2 -- 0.2 degrees.
pub const INITIAL_ROT_VAR: f32 = 3.490_658_5e-3 * 3.490_658_5e-3;

/// Relinearisation passes per update.
///
/// Two, not because more was measured to help -- one to five are
/// indistinguishable on the held-out segment, forty times below the noise
/// floor -- but because each costs a measured ~20 400 cycles on target against
/// a 75 000-cycle budget, so the second is affordable insurance for transients
/// that 20 s of recorded motion may not contain.
pub const ITERATIONS: u8 = 2;

/// Fallback per-channel measurement noise, counts.
///
/// Used only until the rest calibration measures the real thing. The recorded
/// session put the sensor noise at 1.08 counts rms with the knob pinned.
pub const FALLBACK_SIGMA_COUNTS: f32 = 1.08;

/// Process noise vector in the filter's state order.
pub const fn process_noise() -> [f32; POSE_DIM] {
    [Q_POS, Q_POS, Q_POS, Q_ROT, Q_ROT, Q_ROT]
}

/// Initial covariance diagonal in the filter's state order.
pub const fn initial_variance() -> [f32; POSE_DIM] {
    [
        INITIAL_POS_VAR,
        INITIAL_POS_VAR,
        INITIAL_POS_VAR,
        INITIAL_ROT_VAR,
        INITIAL_ROT_VAR,
        INITIAL_ROT_VAR,
    ]
}

/// Measurement variance from per-channel standard deviations.
pub fn measurement_variance(sigma: &[f32; MEAS_DIM]) -> [f32; MEAS_DIM] {
    let mut out = [0.0; MEAS_DIM];
    for i in 0..MEAS_DIM {
        out[i] = sigma[i] * sigma[i];
    }
    out
}
