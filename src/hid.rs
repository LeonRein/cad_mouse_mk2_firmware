//! The USB HID interface: a six-axis controller plus two buttons.
//!
//! # The descriptor
//!
//! Ported byte for byte from the original C++ firmware's
//! `HIDController.cpp`, and deliberately not rewritten. Its exact shape is
//! what host software recognises as a 3-D input device: a Generic Desktop
//! *multi-axis controller*, with all six axes in one report rather than split
//! across two. Older 3Dconnexion devices sent translation in report 1 and
//! rotation in report 2, and host software decides which to expect; a device
//! that gets this wrong fails in ways that look like a sign or scaling
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
//! `poll_ms = 1`, so the host asks for a report every millisecond at full
//! speed — 1 kHz, independent of both the sensor readout and the filter.
//! Reports go out only when a value has actually changed, which is what the
//! original did and which keeps an idle device silent on the bus.

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

/// How often the report is rebuilt and compared against the last one sent.
///
/// Matched to the host's 1 ms polling. Faster would only produce reports the
/// host has no slot to collect.
const TICK: Duration = Duration::from_millis(1);

/// Sends axis and button reports, on change.
#[embassy_executor::task]
pub async fn task(mut writer: HidWriter<'static, Driver<'static, USB>, MAX_PACKET_SIZE>) -> ! {
    info!("HID task started");

    let mut ticker = Ticker::every(TICK);
    let mut last_axes = [0i16; 6];
    let mut last_buttons = 0u16;

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

        if axes != last_axes {
            let mut report = [0u8; 13];
            report[0] = REPORT_ID_AXES;
            for (i, &v) in axes.iter().enumerate() {
                report[1 + i * 2..3 + i * 2].copy_from_slice(&v.to_le_bytes());
            }
            if writer.write(&report).await.is_ok() {
                last_axes = axes;
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
