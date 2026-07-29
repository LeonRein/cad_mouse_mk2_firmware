//! Pose in millimetres and radians, out as HID axis units.
//!
//! The last stage before USB, and the one place where a physical estimate
//! becomes an arbitrary number that some application will interpret however it
//! likes.
//!
//! # Why the original firmware's gains are not here
//!
//! `Config.h` in the C++ carried `GAIN_T = {28, 28, 24}` and
//! `GAIN_R = {18, 18, 20}`. Those multiplied **raw magnetic deltas in counts**,
//! which is what that firmware called a pose. Nothing here is in counts, so
//! those numbers would be meaningless — reusing them is the obvious mistake and
//! it would show up as an axis that barely moves rather than as anything that
//! looks like a bug.
//!
//! # Full scale is per group, not per axis
//!
//! The envelope the mechanism was measured to reach is not the same on every
//! axis: filtering the recorded free motion gives peaks of 1.24, 2.47 and
//! 0.84 mm, and 7.6, 8.5 and 10.2 degrees. It is tempting to normalise each
//! axis to its own peak so that every axis reaches full scale.
//!
//! That would be wrong. Those peaks describe *how the knob was moved during one
//! recording*, not what the mechanism allows, and per-axis normalisation warps
//! direction: a push at 45 degrees between two axes with different scales comes
//! out at some other angle, and the device feels like it pulls to one side.
//! One scale for translation and one for rotation keeps directions honest and
//! costs only that the stiffer axes do not reach the rails.

use crate::model::POSE_DIM;

/// Largest value any axis reports, matching the HID report descriptor's
/// logical maximum. Also the original firmware's `AXIS_LIMIT`.
pub const AXIS_LIMIT: f32 = 350.0;

/// Translation that maps to full scale, millimetres.
///
/// Far larger than the mechanism's own travel, and that is the point: it sets
/// the *sensitivity*, not the range. Started at the measured 2.5 mm envelope,
/// which put full scale within a gentle push and made the device unusably
/// fast, and was divided by fifty from there by hand.
pub const TRANSLATION_FULL_SCALE_MM: f32 = 125.0;

/// Rotation that maps to full scale, radians (100 degrees).
///
/// Same story as the translation, but only a tenth as far off — the rotations
/// needed a factor of ten where the translations needed fifty.
pub const ROTATION_FULL_SCALE_RAD: f32 = 1.745;

/// Per-axis sign, applied last.
///
/// The estimator works in the board frame — `+x` right, `+y` toward the rear,
/// `+z` up, rotations by the right-hand rule — and an application consuming
/// six HID axes has its own idea of which way is which. These start at `+1`
/// because the board frame and the usual multi-axis convention agree on paper;
/// **that is a hypothesis, not a measurement**, and the only way to settle it
/// is to push the knob and watch the model on screen. Flip the entry that goes
/// the wrong way.
pub const AXIS_SIGN: [f32; POSE_DIM] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// Convert an estimated pose to the six axis values the HID report carries.
///
/// The deadzone is deliberately *not* applied here: it is applied on the
/// estimator core, against the measured jitter of the filtered pose, so by the
/// time a pose reaches this function a value of exactly zero already means
/// "at rest".
pub fn to_axes(pose: &[f32; POSE_DIM]) -> [i16; POSE_DIM] {
    let mut out = [0i16; POSE_DIM];
    for i in 0..POSE_DIM {
        let full_scale = if i < 3 {
            TRANSLATION_FULL_SCALE_MM
        } else {
            ROTATION_FULL_SCALE_RAD
        };
        let scaled = AXIS_SIGN[i] * pose[i] / full_scale * AXIS_LIMIT;
        out[i] = clamp_to_limit(scaled);
    }
    out
}

#[inline]
fn clamp_to_limit(value: f32) -> i16 {
    if value >= AXIS_LIMIT {
        AXIS_LIMIT as i16
    } else if value <= -AXIS_LIMIT {
        -(AXIS_LIMIT as i16)
    } else if value.is_nan() {
        // A NaN pose means the filter has diverged. Reporting zero is the only
        // safe answer: `as i16` on a NaN is 0 anyway, but saying so out loud
        // beats relying on that.
        0
    } else {
        value as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rest_reports_exactly_zero() {
        assert_eq!(to_axes(&[0.0; POSE_DIM]), [0; POSE_DIM]);
    }

    #[test]
    fn full_scale_reaches_the_limit() {
        let mut pose = [0.0f32; POSE_DIM];
        pose[0] = TRANSLATION_FULL_SCALE_MM;
        pose[3] = ROTATION_FULL_SCALE_RAD;
        let axes = to_axes(&pose);
        assert_eq!(axes[0], 350);
        assert_eq!(axes[3], 350);
    }

    #[test]
    fn beyond_full_scale_clamps_rather_than_wrapping() {
        let mut pose = [0.0f32; POSE_DIM];
        pose[0] = 100.0 * TRANSLATION_FULL_SCALE_MM;
        pose[1] = -100.0 * TRANSLATION_FULL_SCALE_MM;
        let axes = to_axes(&pose);
        assert_eq!(axes[0], 350);
        assert_eq!(axes[1], -350);
    }

    /// The property that per-axis normalisation would have broken.
    #[test]
    fn diagonal_motion_keeps_its_direction() {
        // A tenth of full scale is exactly 35 axis units, so the ratios below
        // are not confused by the truncation in the final `as i16`.
        let unit = TRANSLATION_FULL_SCALE_MM / 10.0;
        let axes = to_axes(&[unit, unit, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(axes[0], axes[1], "equal push should give equal axes");
        assert_eq!(axes[0], 35);

        // And a 2:1 push stays 2:1.
        let axes = to_axes(&[2.0 * unit, unit, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(axes[0], 2 * axes[1]);
    }

    /// What the chosen sensitivity actually leaves to work with.
    ///
    /// Pinned as a test rather than left as a comment because it is the cost
    /// of the sensitivity choice: full scale sits far outside the mechanism,
    /// so the knob's real travel only reaches a small part of the axis range,
    /// and the reports are correspondingly coarse.
    #[test]
    fn the_mechanisms_own_travel_reaches_only_part_of_the_range() {
        // Measured envelope: 2.5 mm and about 10 degrees.
        let axes = to_axes(&[2.5, 0.0, 0.0, 0.1745, 0.0, 0.0]);
        assert_eq!(axes[0], 7, "2.5 mm of travel");
        assert_eq!(axes[3], 35, "10 degrees of travel");
    }

    #[test]
    fn a_diverged_pose_reports_zero_rather_than_something_arbitrary() {
        let pose = [f32::NAN; POSE_DIM];
        assert_eq!(to_axes(&pose), [0; POSE_DIM]);
    }
}
