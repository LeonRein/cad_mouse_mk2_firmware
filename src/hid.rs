//! The USB HID interface: a six-axis controller plus two buttons.
//!
//! # The descriptor
//!
//! Its exact shape is dictated by what host software recognises as a 3-D
//! input device, not by taste: a Generic Desktop *multi-axis controller*, with
//! all six axes in one report rather than split across two. Treat it as an
//! interface constant. Older 3Dconnexion devices sent translation in report 1
//! and rotation in report 2, and host software decides which to expect; a
//! device that gets this wrong fails in ways that look like a sign or scaling
//! problem rather than a descriptor problem.
//!
//! # Using it on Linux
//!
//! The kernel's generic HID driver turns this descriptor into an evdev device
//! with `ABS_X`..`ABS_RZ`, which is what `spacenavd` consumes. It will not
//! pick the device up on sight, though: `spacenavd` matches a built-in table
//! of 3Dconnexion and Logitech USB IDs, and this firmware deliberately does
//! not claim one of those. Add this device's ID to `/etc/spnavrc`:
//!
//! ```text
//! device-id = 1209:0001
//! ```
//!
//! and it is treated exactly like a retail device from then on. See
//! `scripts/README.md` for the whole setup.
//!
//! # Rate
//!
//! Two separate numbers, and confusing them is easy. `poll_ms = 1` is the
//! endpoint interval — how often the host *asks*. [`REPORT_INTERVAL_MS`] is how
//! often this task has anything new to give it, and that is the one that
//! decides how the device feels. See its docs for the choice.
//!
//! Reports go out only when a value has actually changed, which keeps an idle
//! device silent on the bus. On Linux that is indistinguishable from a retail
//! device's continuous stream: the input core drops an `EV_ABS` event whose
//! value did not change, so a held deflection produces nothing either way. The
//! one place the difference bites is the return to rest — see
//! [`ZERO_REPEATS`].

use cadmouse_model::shaping;
use defmt::info;
use embassy_rp::peripherals::USB;
use embassy_rp::usb::Driver;
use embassy_time::{Duration, Ticker};
use embassy_usb::class::hid::HidWriter;

use crate::buttons;
use crate::estimator;
use crate::protocol::status;

/// Report ID carrying the six axes.
pub const REPORT_ID_AXES: u8 = 1;

/// Report ID carrying the buttons.
pub const REPORT_ID_BUTTONS: u8 = 3;

/// Endpoint packet size. The largest report is 1 + 6 x 2 = 13 bytes.
pub const MAX_PACKET_SIZE: usize = 16;

/// Multi-axis controller, six 16-bit axes at +-350, plus two buttons.
#[rustfmt::skip]
pub const REPORT_DESCRIPTOR: &[u8] = &[
    0x05, 0x01,        // USAGE_PAGE (Generic Desktop)
    0x09, 0x08,        // USAGE (Multi-axis Controller)
    0xA1, 0x01,        // COLLECTION (Application)
    0xA1, 0x00,        //   COLLECTION (Physical)
    0x85, 0x01,        //     REPORT_ID (1)
    0x16, 0xA2, 0xFE,  //     LOGICAL_MINIMUM (-350)
    0x26, 0x5E, 0x01,  //     LOGICAL_MAXIMUM (350)
    0x09, 0x30,        //     USAGE (X)
    0x09, 0x31,        //     USAGE (Y)
    0x09, 0x32,        //     USAGE (Z)
    0x09, 0x33,        //     USAGE (Rx)
    0x09, 0x34,        //     USAGE (Ry)
    0x09, 0x35,        //     USAGE (Rz)
    0x75, 0x10,        //     REPORT_SIZE (16)
    0x95, 0x06,        //     REPORT_COUNT (6)
    0x81, 0x02,        //     INPUT (Data,Var,Abs)
    0xC0,              //   END_COLLECTION
    0xA1, 0x00,        //   COLLECTION (Physical)
    0x85, 0x03,        //     REPORT_ID (3)
    0x05, 0x09,        //     USAGE_PAGE (Button)
    0x19, 0x01,        //     USAGE_MINIMUM (Button 1)
    0x29, 0x02,        //     USAGE_MAXIMUM (Button 2)
    0x15, 0x00,        //     LOGICAL_MINIMUM (0)
    0x25, 0x01,        //     LOGICAL_MAXIMUM (1)
    0x75, 0x01,        //     REPORT_SIZE (1)
    0x95, 0x02,        //     REPORT_COUNT (2)
    0x81, 0x02,        //     INPUT (Data,Var,Abs)
    0x95, 0x0E,        //     REPORT_COUNT (14) padding
    0x81, 0x01,        //     INPUT (Const,Array,Abs)
    0xC0,              //   END_COLLECTION
    0xC0,              // END_COLLECTION
];

/// Milliseconds between axis reports. **The one number to change.**
///
/// Two settings make sense, and which is right depends on the host rather than
/// on anything measurable here, so it is one named constant rather than
/// something derived:
///
/// * **8** — what a retail 3Dconnexion device does: roughly 125 packets per
///   second while the knob is displaced. Host software that advances the view
///   once per *event* rather than per unit of elapsed time is tuned against
///   that cadence, and more of it works that way than you would guess:
///   FreeCAD's `pollSpacenav` runs off a `QSocketNotifier`, so at any rate its
///   event loop can service it steps the camera once per event.
/// * **1** — a report at every host poll. Eight times the traffic, and less
///   quantisation on the report's own latency. Per-event consumers then run
///   proportionally fast, corrected by scaling the host's sensitivity and
///   changing nothing else.
///
/// **The ratio between the two is not the tick ratio.** Reports go out only on
/// change, and at 1 ms the estimate frequently has not moved since the last
/// tick, so ticks are skipped. Measured on this device through `spacenavd`:
///
/// | tick | nominal | measured | duty |
/// |------|---------|----------|------|
/// | 1 ms | 1000 Hz |   830 Hz | 83 % |
/// | 8 ms |  125 Hz |   122 Hz | 98 % |
///
/// Eight times more change accumulates per tick at 8 ms, so almost no tick is
/// wasted. The real factor between the two settings is therefore about **6.8**,
/// not 8 — and 1 ms spends a fifth of its bandwidth on reports that say nothing
/// new. Measure with `spacenav-ws measure-rate` rather than assuming either.
///
/// Eight by default: a device that needs the host reconfigured before it feels
/// right is a device that feels broken on the first machine it meets.
pub const REPORT_INTERVAL_MS: u64 = 1;

/// How often the report is rebuilt and compared against the last one sent.
const TICK: Duration = Duration::from_millis(REPORT_INTERVAL_MS);

/// How many times the all-zero axis report is sent when the knob returns to
/// rest.
///
/// A retail device sends three, and the reason is worth keeping: every other
/// report is a correction that the next one supersedes, but the report saying
/// "stopped" has nothing behind it. Lose it and the host goes on applying the
/// last non-zero value forever — a view that creeps on its own, which is the
/// exact failure the `CALIBRATED` gate below already exists to avoid.
const ZERO_REPEATS: u8 = 3;

/// Sends axis and button reports, on change.
#[embassy_executor::task]
pub async fn task(mut writer: HidWriter<'static, Driver<'static, USB>, MAX_PACKET_SIZE>) -> ! {
    info!("HID task started");

    let mut ticker = Ticker::every(TICK);
    let mut last_axes = [0i16; 6];
    let mut last_buttons = 0u16;
    // Zero reports still owed, counting down. Starts at nothing: an idle
    // device that has never moved has no rest to announce.
    let mut zero_repeats = 0u8;

    loop {
        ticker.next().await;

        // Report zeros until the device has measured its own rest. Before
        // that the pose is relative to a nominal zero and has no deadzone, so
        // it would arrive as a small permanent drift — a 3-D view creeping on
        // its own is a miserable first impression, and worse, it looks like a
        // hardware fault rather than a device that is not ready yet.
        let estimate = estimator::latest();
        let axes = match estimate {
            Some(e) if e.status & status::CALIBRATED != 0 => shaping::to_axes(&e.pose),
            _ => [0i16; 6],
        };

        let buttons = (buttons::left_pressed() as u16) | ((buttons::right_pressed() as u16) << 1);

        let changed = axes != last_axes;
        if changed || zero_repeats > 0 {
            let mut report = [0u8; 13];
            report[0] = REPORT_ID_AXES;
            for (i, &v) in axes.iter().enumerate() {
                report[1 + i * 2..3 + i * 2].copy_from_slice(&v.to_le_bytes());
            }
            // Only on success, so a refused write is simply retried next tick
            // rather than counting as one of the three.
            if writer.write(&report).await.is_ok() {
                last_axes = axes;
                zero_repeats = if !changed {
                    zero_repeats - 1
                } else if axes == [0i16; 6] {
                    // This send was the first of the three.
                    ZERO_REPEATS - 1
                } else {
                    // Moving again; whatever was owed is moot.
                    0
                };
            }
        }

        if buttons != last_buttons {
            let mut report = [0u8; 3];
            report[0] = REPORT_ID_BUTTONS;
            report[1..3].copy_from_slice(&buttons.to_le_bytes());
            if writer.write(&report).await.is_ok() {
                last_buttons = buttons;
            }
        }
    }
}
