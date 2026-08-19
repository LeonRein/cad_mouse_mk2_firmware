//! Pose in millimetres and radians, out as HID axis units.
//!
//! The last stage before USB, and the one place where a physical estimate
//! becomes an arbitrary number that some application will interpret however it
//! likes.
//!
//! # Gains from a counts-based design do not carry over
//!
//! Worth stating because it is the obvious mistake: a design that treats the
//! raw magnetic delta *as* the pose needs per-axis gains to turn counts into
//! axis units. Nothing here is in counts — the estimator hands over
//! millimetres and radians — so any such gain is meaningless here, and reusing
//! one shows up as an axis that barely moves rather than as anything that
//! looks like a bug.
//!
//! # Full scale is measured, but the sensitivity is chosen
//!
//! Full scale used to be two hand-tuned constants, 125 mm and 100 degrees,
//! reached by dividing a measured envelope by fifty and by ten respectively
//! until the device felt right. That worked, but it fused two independent
//! things into one number: how far this particular knob gets pushed, which is
//! measurable, and how much deflection a person wants per millimetre, which is
//! taste. They are separate here — [`USAGE_ENVELOPE`](crate::generated) comes
//! from the calibration, [`USAGE_FRACTION_OF_RANGE`] is the choice — so
//! recalibrating never silently changes the feel, and changing the feel is one
//! obvious number.
//!
//! The knob has **no endstop**: the restoring force simply grows until
//! something gives. So there is no mechanical limit to normalise against, and
//! the only well-defined envelope is behavioural — how far the operator
//! actually moves it. That is what the `usage` segment records.
//!
//! # Full scale is per group, not per axis
//!
//! The envelope is not the same on every axis, and it is tempting to normalise
//! each one to its own so that every axis reaches full scale.
//!
//! That would be wrong twice over. Those numbers describe *how the knob was
//! moved during one recording*, not what the mechanism allows; and per-axis
//! normalisation warps direction, because a push at 45 degrees between two axes
//! scaled differently comes out at some other angle and the device feels like
//! it pulls to one side. One scale for translation and one for rotation keeps
//! directions honest and costs only that the stiffer axes do not reach the
//! rails.
//!
//! The same argument rules out normalising the two *directions* of one axis
//! separately, and there the temptation is stronger because the recorded
//! asymmetry is large — `tz` comes out six times bigger downward. But
//! `record.py` says "press the knob DOWN and let it rise", so that asymmetry is
//! the instruction, not the spring. Scaling the two directions differently
//! would make equal pushes left and right report unequal numbers.

use crate::generated as consts;
use crate::model::POSE_DIM;

/// Largest value any axis reports, matching the HID report descriptor's
/// logical maximum.
pub const AXIS_LIMIT: f32 = 350.0;

/// What fraction of the axis range ordinary use should reach.
///
/// This is the whole sensitivity choice, and separating it from the measured
/// envelope is the point of the split below. The two used to be fused into one
/// hand-tuned constant — full scale at 125 mm, arrived at by taking a measured
/// 2.5 mm envelope and dividing by fifty by feel — which made the number
/// impossible to reason about and tied to one particular device.
///
/// At 0.9, the deflection the operator reaches 99 % of the time in normal work
/// reports 90 % of full scale. Pushing harder still resolves, up to the clamp;
/// the knob has no endstop, so there is always more travel available and
/// leaving headroom above "normal" matters more than reaching exactly 350.
pub const USAGE_FRACTION_OF_RANGE: f32 = 0.9;

/// Fallback envelope for a calibration recorded before the `usage` segment
/// existed, chosen to reproduce the old hand-tuned constants exactly.
const FALLBACK_ENVELOPE: (f32, f32) = (
    125.0 * USAGE_FRACTION_OF_RANGE,
    1.745 * USAGE_FRACTION_OF_RANGE,
);

const ENVELOPE: (f32, f32) = match consts::USAGE_ENVELOPE {
    Some(e) => e,
    None => FALLBACK_ENVELOPE,
};

/// Translation that maps to full scale, millimetres.
///
/// Derived, not chosen: the measured radius of ordinary use divided by the
/// fraction of the range it should occupy.
pub const TRANSLATION_FULL_SCALE_MM: f32 = ENVELOPE.0 / USAGE_FRACTION_OF_RANGE;

/// Rotation that maps to full scale, radians.
pub const ROTATION_FULL_SCALE_RAD: f32 = ENVELOPE.1 / USAGE_FRACTION_OF_RANGE;

/// Per-axis sign, applied last.
///
/// The estimator works in the board frame — `+x` right, `+y` toward the rear,
/// `+z` up, rotations by the right-hand rule — and an application consuming
/// six HID axes has its own idea of which way is which.
///
/// All `+1` is **confirmed on hardware**: every axis moves the model the right
/// way in FreeCAD and in Onshape, with no inversion configured anywhere. One
/// qualifier, because it is easy to read as a stronger claim than it is — that
/// was measured through `spacenavd` with `swap-yz = true` set in `spnavrc`, so
/// what is verified is *this firmware together with that setting*, not the raw
/// HID frame standing alone. A host reading the device directly (3DxWare,
/// WebHID, hidraw) sees an unswapped frame and may need different signs.
///
/// If an axis ever goes the wrong way, flip its entry here.
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
        let unit = TRANSLATION_FULL_SCALE_MM / 10.0;
        let axes = to_axes(&[unit, unit, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(axes[0], axes[1], "equal push should give equal axes");
        assert!(
            (axes[0] - 35).abs() <= 1,
            "a tenth of full scale: {}",
            axes[0]
        );

        // And a 2:1 push stays 2:1, to within the final truncation.
        let axes = to_axes(&[2.0 * unit, unit, 0.0, 0.0, 0.0, 0.0]);
        assert!(
            (axes[0] - 2 * axes[1]).abs() <= 1,
            "2:1 push gave {}:{}",
            axes[0],
            axes[1]
        );
    }

    /// Ordinary use lands where the sensitivity says it should.
    ///
    /// This is the property the whole split exists to guarantee: whatever the
    /// calibration measured as the edge of normal movement must report
    /// [`USAGE_FRACTION_OF_RANGE`] of full scale. It holds for a measured
    /// envelope and for the fallback alike, so it does not need updating when
    /// a device is recalibrated.
    #[test]
    fn ordinary_use_reaches_the_intended_fraction_of_the_range() {
        // Within one unit, because the last step is a truncating `as i16` and
        // the full scale is no longer a round number.
        let expected = (USAGE_FRACTION_OF_RANGE * AXIS_LIMIT) as i16;
        let t = TRANSLATION_FULL_SCALE_MM * USAGE_FRACTION_OF_RANGE;
        let r = ROTATION_FULL_SCALE_RAD * USAGE_FRACTION_OF_RANGE;
        let axes = to_axes(&[t, 0.0, 0.0, r, 0.0, 0.0]);
        assert!(
            (axes[0] - expected).abs() <= 1,
            "translation at the edge of ordinary use: {} vs {expected}",
            axes[0]
        );
        assert!(
            (axes[3] - expected).abs() <= 1,
            "rotation at the edge of ordinary use: {} vs {expected}",
            axes[3]
        );
    }

    /// There is headroom above ordinary use, because the knob has no endstop.
    ///
    /// Pushing harder than normal must keep resolving rather than sitting on
    /// the clamp, or the device would feel like it hits a wall that the
    /// mechanism does not actually have.
    #[test]
    fn pushing_harder_than_usual_still_resolves() {
        let hard = 1.1 * TRANSLATION_FULL_SCALE_MM * USAGE_FRACTION_OF_RANGE;
        let axes = to_axes(&[hard, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let usual = (USAGE_FRACTION_OF_RANGE * AXIS_LIMIT) as i16;
        assert!(axes[0] > usual, "harder push must report more");
        assert!(axes[0] <= AXIS_LIMIT as i16, "and still be clamped");
    }

    /// A calibration with no `usage` segment must not change the feel.
    #[test]
    fn the_fallback_reproduces_the_old_hand_tuned_constants() {
        if crate::generated::USAGE_ENVELOPE.is_none() {
            assert!((TRANSLATION_FULL_SCALE_MM - 125.0).abs() < 1e-3);
            assert!((ROTATION_FULL_SCALE_RAD - 1.745).abs() < 1e-4);
        }
    }

    #[test]
    fn a_diverged_pose_reports_zero_rather_than_something_arbitrary() {
        let pose = [f32::NAN; POSE_DIM];
        assert_eq!(to_axes(&pose), [0; POSE_DIM]);
    }
}
