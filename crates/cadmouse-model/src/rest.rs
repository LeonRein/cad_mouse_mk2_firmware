//! The calibration that runs on the device, with the knob untouched.
//!
//! There are two calibrations in this project and they answer different
//! questions. The one on the PC (`scripts/calibrate.py`) fits the twenty-seven
//! parameters of the *mechanism* -- where the magnets are, which way they
//! point, how strong they are -- from a session where the user deliberately
//! exercises every axis. That is slow, needs an optimiser, and only has to
//! happen once per device; its result is baked into
//! [`crate::generated`].
//!
//! This one answers something much smaller and much more perishable: *where is
//! rest, right now, on this power-up, at this temperature*. Nothing about the
//! mechanism is touched. Three numbers come out:
//!
//! * **The rest pose**, subtracted from every later estimate so the device
//!   reports zero when the hand is off it. This is also what makes thermal
//!   drift a non-problem: a re-zero corrects the bias whatever caused it,
//!   which is why no thermal model was ever written.
//! * **The per-channel measurement noise**, which becomes the filter's `R`.
//!   Measuring it beats assuming it, and a filter whose `R` is wrong is
//!   dishonest about its own uncertainty in a way that is very hard to notice.
//! * **The deadzone**, per axis, from the jitter the *filtered pose* actually
//!   shows at rest. Note this is a deadzone on the pose, not on raw counts --
//!   by the time the estimator has run, counts are no longer the quantity
//!   anyone cares about.
//!
//! The user is not asked to do anything, so the routine has to check for
//! itself that the knob really was at rest; see [`MAX_REST_MOTION_MM`].

use crate::model::{MEAS_DIM, POSE_DIM};

/// Frames discarded before measurement starts, letting the filter settle onto
/// the rest pose from wherever it happened to be. At the rates this runs at
/// this is a fraction of a second, and the filter converges in far less.
const SETTLE_FRAMES: u16 = 256;

/// Frames averaged. At 800-2000 Hz this is roughly half a second to a second
/// -- long enough for the mean to be worth much more than one sample, short
/// enough that the user is not left waiting at a spinner.
const COLLECT_FRAMES: u16 = 1024;

/// Deadzone as a multiple of the measured pose jitter. Three sigma, so genuine
/// stillness reads as zero essentially always while anything deliberate gets
/// through.
const DEADZONE_SIGMAS: f32 = 3.0;

/// How far the pose may wander during collection before the result is thrown
/// away, in millimetres of peak-to-peak travel.
///
/// Sized to sit in the wide gap between the two things it must separate: the
/// filtered pose at genuine rest moves by a few micrometres, and a hand
/// resting on the knob moves it by hundreds. Anything in this range would be a
/// bad zero, and a bad zero is worse than no zero -- it is silently wrong for
/// as long as the device stays powered.
pub const MAX_REST_MOTION_MM: f32 = 0.05;

/// The same, in radians -- about 0.3 degrees.
pub const MAX_REST_MOTION_RAD: f32 = 0.005;

/// What the device knows about its own rest state.
#[derive(Clone, Copy)]
pub struct Calibration {
    /// Pose reported when the knob is at rest, subtracted from every estimate.
    pub rest_pose: [f32; POSE_DIM],
    /// Per-channel measurement noise, counts.
    pub sigma: [f32; MEAS_DIM],
    /// Per-axis deadzone, mm and rad.
    pub deadzone: [f32; POSE_DIM],
}

impl Calibration {
    /// What to use before any calibration has run: no zeroing, no deadzone,
    /// and the noise figure measured on the recorded session.
    pub fn fallback() -> Self {
        Self {
            rest_pose: [0.0; POSE_DIM],
            sigma: [crate::tuning::FALLBACK_SIGMA_COUNTS; MEAS_DIM],
            deadzone: [0.0; POSE_DIM],
        }
    }
}

/// Why a calibration attempt was thrown away.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Abort {
    /// The pose moved more than [`MAX_REST_MOTION_MM`] during collection.
    KnobMoved,
}

/// What to do after feeding one frame in.
pub enum Step {
    /// Still working.
    Continue,
    Finished(Calibration),
    Aborted(Abort),
}

/// Collects the statistics one frame at a time.
///
/// Deliberately a fed state machine rather than a loop of its own: it runs
/// inside the estimator's frame loop, so the filter keeps updating and the
/// device keeps streaming while it works.
pub struct RestCalibration {
    settling: u16,
    collected: u16,
    /// First collected sample, subtracted from everything after it.
    ///
    /// Not cosmetic. At rest the pose barely moves, so accumulating raw values
    /// and forming `E[x^2] - E[x]^2` subtracts two nearly equal `f32`s and
    /// leaves noise -- the variance comes out as garbage, sometimes negative.
    /// Working in deltas from the first sample keeps every accumulated
    /// quantity at the scale of the thing being measured.
    origin: [f32; POSE_DIM],
    pose_sum: [f32; POSE_DIM],
    pose_sumsq: [f32; POSE_DIM],
    pose_min: [f32; POSE_DIM],
    pose_max: [f32; POSE_DIM],
    /// Exact in `i32`/`i64`, since the counts are integers to begin with.
    count_sum: [i32; MEAS_DIM],
    count_sumsq: [i64; MEAS_DIM],
}

impl RestCalibration {
    pub fn new() -> Self {
        Self {
            settling: SETTLE_FRAMES,
            collected: 0,
            origin: [0.0; POSE_DIM],
            pose_sum: [0.0; POSE_DIM],
            pose_sumsq: [0.0; POSE_DIM],
            pose_min: [f32::MAX; POSE_DIM],
            pose_max: [f32::MIN; POSE_DIM],
            count_sum: [0; MEAS_DIM],
            count_sumsq: [0; MEAS_DIM],
        }
    }

    /// Fraction complete, 0-255, for the host and the LED.
    pub fn progress(&self) -> u8 {
        let done = (SETTLE_FRAMES - self.settling) as u32 + self.collected as u32;
        let total = SETTLE_FRAMES as u32 + COLLECT_FRAMES as u32;
        (done * 255 / total) as u8
    }

    /// Feed one frame: the raw counts, and the pose the filter made of them.
    pub fn feed(&mut self, counts: &[i16; MEAS_DIM], pose: &[f32; POSE_DIM]) -> Step {
        if self.settling > 0 {
            self.settling -= 1;
            return Step::Continue;
        }

        if self.collected == 0 {
            self.origin = *pose;
        }

        for i in 0..POSE_DIM {
            let d = pose[i] - self.origin[i];
            self.pose_sum[i] += d;
            self.pose_sumsq[i] += d * d;
            if d < self.pose_min[i] {
                self.pose_min[i] = d;
            }
            if d > self.pose_max[i] {
                self.pose_max[i] = d;
            }
        }
        for i in 0..MEAS_DIM {
            let c = counts[i] as i32;
            self.count_sum[i] += c;
            self.count_sumsq[i] += (c as i64) * (c as i64);
        }

        self.collected += 1;
        if self.collected < COLLECT_FRAMES {
            return Step::Continue;
        }

        // Did the knob hold still for all of it?
        for i in 0..POSE_DIM {
            let travel = self.pose_max[i] - self.pose_min[i];
            let limit = if i < 3 {
                MAX_REST_MOTION_MM
            } else {
                MAX_REST_MOTION_RAD
            };
            if travel > limit {
                return Step::Aborted(Abort::KnobMoved);
            }
        }

        Step::Finished(self.finish())
    }

    fn finish(&self) -> Calibration {
        let n = self.collected as f32;

        let mut rest_pose = [0.0; POSE_DIM];
        let mut deadzone = [0.0; POSE_DIM];
        for i in 0..POSE_DIM {
            let mean = self.pose_sum[i] / n;
            rest_pose[i] = self.origin[i] + mean;
            let variance = (self.pose_sumsq[i] / n - mean * mean).max(0.0);
            deadzone[i] = DEADZONE_SIGMAS * libm::sqrtf(variance);
        }

        let mut sigma = [0.0; MEAS_DIM];
        for i in 0..MEAS_DIM {
            let sum = self.count_sum[i] as f64;
            let sumsq = self.count_sumsq[i] as f64;
            let n64 = self.collected as f64;
            let variance = ((sumsq - sum * sum / n64) / (n64 - 1.0)).max(0.0);
            // Never hand the filter a zero: a channel that happens not to
            // move for a second would make R singular and every update fail.
            sigma[i] = (libm::sqrt(variance) as f32).max(0.1);
        }

        Calibration {
            rest_pose,
            sigma,
            deadzone,
        }
    }
}

/// Per-axis deadzone with hysteresis.
///
/// Hysteresis rather than a hard threshold. A hard threshold chatters: a pose
/// sitting exactly at the boundary flips between zero and not-zero every
/// frame, which downstream reads as a very fast, very small oscillation.
/// Leaving the zero requires exceeding the threshold by half again, so the
/// boundary can only be crossed decisively.
pub struct Deadzone {
    thresholds: [f32; POSE_DIM],
    zeroed: [bool; POSE_DIM],
}

/// Multiple of the threshold at which an axis is allowed to leave zero.
const HYSTERESIS: f32 = 1.5;

impl Deadzone {
    pub fn new(thresholds: [f32; POSE_DIM]) -> Self {
        Self {
            thresholds,
            zeroed: [true; POSE_DIM],
        }
    }

    pub fn set_thresholds(&mut self, thresholds: [f32; POSE_DIM]) {
        self.thresholds = thresholds;
    }

    /// Returns the gated pose and whether every axis is sitting at zero.
    pub fn apply(&mut self, pose: &[f32; POSE_DIM]) -> ([f32; POSE_DIM], bool) {
        let mut out = [0.0; POSE_DIM];
        let mut all_zero = true;
        for i in 0..POSE_DIM {
            let magnitude = libm::fabsf(pose[i]);
            let threshold = self.thresholds[i];
            self.zeroed[i] = if self.zeroed[i] {
                magnitude < threshold * HYSTERESIS
            } else {
                magnitude < threshold
            };
            if self.zeroed[i] {
                out[i] = 0.0;
            } else {
                out[i] = pose[i];
                all_zero = false;
            }
        }
        (out, all_zero)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed a still knob and check the three outputs are what was fed in.
    #[test]
    fn a_still_knob_produces_its_own_rest_pose() {
        let mut cal = RestCalibration::new();
        let pose = [0.31, -0.12, 0.05, 0.001, -0.002, 0.0];
        let counts = [100i16, -200, 3000, 10, -10, 2500, 0, 5, -1500];

        let mut result = None;
        for k in 0..(SETTLE_FRAMES + COLLECT_FRAMES) {
            // A deterministic 2-count wobble on the counts and a micrometre
            // of pose jitter, which is roughly what rest actually looks like.
            let wobble = if k % 2 == 0 { 1e-6 } else { -1e-6 };
            let mut jittered = pose;
            for v in jittered.iter_mut() {
                *v += wobble;
            }
            match cal.feed(&counts, &jittered) {
                Step::Finished(c) => {
                    result = Some(c);
                    break;
                }
                Step::Aborted(_) => panic!("a still knob must not abort"),
                Step::Continue => {}
            }
        }

        let c = result.expect("calibration should finish");
        for i in 0..POSE_DIM {
            assert!(
                (c.rest_pose[i] - pose[i]).abs() < 1e-5,
                "axis {i}: {} vs {}",
                c.rest_pose[i],
                pose[i]
            );
            assert!(c.deadzone[i] >= 0.0);
        }
        // Constant counts: the noise floor clamp is what should show up.
        assert!(c.sigma.iter().all(|&s| s == 0.1), "{:?}", c.sigma);
    }

    #[test]
    fn a_moving_knob_is_rejected_rather_than_averaged() {
        let mut cal = RestCalibration::new();
        let counts = [0i16; MEAS_DIM];
        let mut aborted = false;
        for k in 0..(SETTLE_FRAMES + COLLECT_FRAMES) {
            let mut pose = [0.0f32; POSE_DIM];
            // A slow drift of a millimetre, twenty times the limit.
            pose[0] = k as f32 * 1e-3;
            if let Step::Aborted(Abort::KnobMoved) = cal.feed(&counts, &pose) {
                aborted = true;
                break;
            }
        }
        assert!(aborted, "drifting pose should have aborted the calibration");
    }

    #[test]
    fn deadzone_holds_zero_until_the_threshold_is_clearly_passed() {
        let mut dz = Deadzone::new([0.1; POSE_DIM]);

        let (out, all_zero) = dz.apply(&[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(out[0], 0.0);
        assert!(all_zero);

        // Between the threshold and the hysteresis limit: still zero, because
        // it started zeroed.
        let (out, _) = dz.apply(&[0.12, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(out[0], 0.0);

        // Past the hysteresis limit: releases.
        let (out, all_zero) = dz.apply(&[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(out[0], 0.2);
        assert!(!all_zero);

        // And now it stays released until it comes back inside the threshold.
        let (out, _) = dz.apply(&[0.12, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(out[0], 0.12);
        let (out, _) = dz.apply(&[0.05, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(out[0], 0.0);
    }
}
