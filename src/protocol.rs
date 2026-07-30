//! The debug wire format, device to host.
//!
//! Its own module because it is a contract with code outside this repository's
//! Rust: `scripts/record.py` and `scripts/view.py` decode exactly this layout,
//! and a change here that is not mirrored there produces a host-side parser
//! that reads plausible nonsense rather than an error.
//!
//! One frame per sample. Everything little-endian.
//!
//! | offset | size | field                                                |
//! |--------|------|------------------------------------------------------|
//! | 0      | 2    | [`FRAME_MAGIC`]                                      |
//! | 2      | 2    | `seq`, wrapping frame counter                        |
//! | 4      | 4    | `t_us`, device uptime in microseconds                |
//! | 8      | 18   | nine `i16` raw counts, MAG1/2/3 x,y,z                |
//! | 26     | 24   | six `f32` pose: x, y, z (mm), rx, ry, rz (rad)       |
//! | 50     | 4    | `f32` normalised innovation squared                  |
//! | 54     | 1    | [`Status`] bits                                      |
//! | 55     | 1    | calibration progress, 0-255                          |
//!
//! Binary rather than CSV: a worst-case CSV line would have exceeded the
//! 64-byte packet and been silently truncated, and formatting fifteen numbers
//! per sample at 2 kHz is real work for no gain.
//!
//! The pose is what the estimator reports *after* zeroing and the deadzone --
//! that is, what the HID axes will eventually be built from. The raw counts
//! travel alongside it so a recorded session can still be re-filtered on the
//! host with different tuning, which is the whole point of keeping the debug
//! stream once HID exists.

/// Sync word starting every frame.
///
/// The host needs a way to resynchronise: USB CDC is a byte stream with no
/// record boundaries, so a reader that attaches mid-frame has to find the
/// start of the next one.
///
/// **Bumped from `0xA55A` when the pose fields were added.** A host built for
/// the old 26-byte frame will now find no frames at all, which is a far better
/// failure than one that silently reads pose bytes as the next sample's counts.
pub const FRAME_MAGIC: u16 = 0xA55B;

/// Wire size of one frame.
///
/// Deliberately under the 64-byte USB full-speed bulk packet limit so a frame
/// is always exactly one `write_packet` -- no fragmentation, no
/// zero-length-packet handling on either side.
pub const FRAME_LEN: usize = 56;

/// Status bits in byte 54.
pub mod status {
    /// The filter has been initialised and its last update succeeded.
    pub const FILTER_VALID: u8 = 1 << 0;
    /// A rest calibration has completed, so the pose is zeroed and the
    /// deadzone and measurement noise are the measured ones.
    pub const CALIBRATED: u8 = 1 << 1;
    /// A rest calibration is in progress; `progress` is meaningful.
    pub const CALIBRATING: u8 = 1 << 2;
    /// The last calibration attempt was abandoned because the knob moved.
    pub const CALIBRATION_ABORTED: u8 = 1 << 3;
    /// The last update was rejected -- the innovation covariance stopped being
    /// positive definite, which means the filter had already diverged.
    pub const DIVERGED: u8 = 1 << 4;
    /// Every axis is inside its deadzone, i.e. the device is reporting rest.
    pub const IN_DEADZONE: u8 = 1 << 5;
    /// The left button is down right now.
    ///
    /// Raw press state rather than any gesture: `record.py` uses it to let the
    /// operator step through a capture without reaching for the keyboard, and
    /// the hand is already on the knob. The five-second both-buttons hold that
    /// starts a rest calibration is unaffected -- it is timed on the device and
    /// a tap is nowhere near it.
    pub const BUTTON_LEFT: u8 = 1 << 6;
    /// The right button is down right now.
    pub const BUTTON_RIGHT: u8 = 1 << 7;
}

/// One sample as it goes on the wire.
#[derive(Clone, Copy, Default)]
pub struct Frame {
    pub seq: u16,
    pub t_us: u32,
    pub counts: [i16; 9],
    pub pose: [f32; 6],
    pub nis: f32,
    pub status: u8,
    pub progress: u8,
}

impl Frame {
    /// Serialise into a fixed-size buffer.
    pub fn encode(&self, buf: &mut [u8; FRAME_LEN]) {
        buf[0..2].copy_from_slice(&FRAME_MAGIC.to_le_bytes());
        buf[2..4].copy_from_slice(&self.seq.to_le_bytes());
        buf[4..8].copy_from_slice(&self.t_us.to_le_bytes());
        for (i, &v) in self.counts.iter().enumerate() {
            let at = 8 + i * 2;
            buf[at..at + 2].copy_from_slice(&v.to_le_bytes());
        }
        for (i, &v) in self.pose.iter().enumerate() {
            let at = POSE_OFFSET + i * 4;
            buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        buf[NIS_OFFSET..NIS_OFFSET + 4].copy_from_slice(&self.nis.to_le_bytes());
        buf[STATUS_OFFSET] = self.status;
        buf[STATUS_OFFSET + 1] = self.progress;
    }
}

// The layout is a contract with the Python, so the sizes that make it up are
// asserted at compile time rather than trusted to the table above staying in
// step with `encode`. `scripts/tests/test_protocol.py` checks the other end of
// the same contract against a hand-built frame.
const _: () = assert!(FRAME_LEN == 56, "frame length changed; update the host");
const _: () = assert!(FRAME_LEN <= 64, "a frame must fit one full-speed packet");
const _: () = assert!(POSE_OFFSET == 26 && NIS_OFFSET == 50);

const POSE_OFFSET: usize = 8 + 9 * 2;
const NIS_OFFSET: usize = POSE_OFFSET + 6 * 4;
const STATUS_OFFSET: usize = NIS_OFFSET + 4;
