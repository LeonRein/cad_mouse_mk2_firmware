//! The two side buttons, and the gesture that starts a rest calibration.
//!
//! Both buttons are active low with the internal pull-up, matching the
//! original firmware's `INPUT_PULLUP`. D0/GPIO26 is the right button and
//! D2/GPIO28 the left.
//!
//! Two things come out of here:
//!
//! * **Press state**, in [`left_pressed`] / [`right_pressed`], for whatever
//!   wants it. Nothing does yet -- the HID report that will carry these is not
//!   implemented -- so they exist to be read, not to be acted on.
//! * **The calibration gesture**: hold both for five seconds. This module owns
//!   the gesture end to end, including its LED feedback, because a hold that
//!   gives no sign of being noticed until it fires is indistinguishable from a
//!   dead button. The ring fills one pixel per 625 ms so the user can see the
//!   commitment building, and lets go of the previous pattern only for as long
//!   as the hold lasts.

use core::sync::atomic::{AtomicBool, Ordering};

use embassy_rp::gpio::Input;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::signal::Signal;
use embassy_time::{Duration, Instant, Timer};

use crate::led::{self, LED_COUNT, Pattern};

/// How long both buttons must be held to request a calibration.
pub const CALIBRATION_HOLD: Duration = Duration::from_secs(5);

/// Poll interval. Fast enough to feel immediate, slow enough to be free.
const POLL: Duration = Duration::from_millis(10);

/// Consecutive agreeing samples before a change is believed -- 30 ms of
/// contact bounce, which is generous for a tactile switch.
const DEBOUNCE_SAMPLES: u8 = 3;

static LEFT: AtomicBool = AtomicBool::new(false);
static RIGHT: AtomicBool = AtomicBool::new(false);

/// Raised when the hold completes. The estimator on core 1 waits on this.
///
/// A `Signal` carries no queue, which is what we want: holding the buttons
/// twice while a calibration is already running should not book a second one.
static CALIBRATION_REQUEST: Signal<CriticalSectionRawMutex, ()> = Signal::new();

// Read by the HID report once there is one. Kept because the debouncing that
// produces them is already running for the gesture, and because a button whose
// state is not published anywhere is a button nobody remembers to wire up.
#[allow(dead_code)]
pub fn left_pressed() -> bool {
    LEFT.load(Ordering::Relaxed)
}

#[allow(dead_code)]
pub fn right_pressed() -> bool {
    RIGHT.load(Ordering::Relaxed)
}

/// Take a pending calibration request, if there is one.
///
/// Polled rather than awaited because the estimator is already blocked on the
/// next sample and must not be blocked on two things at once -- a request that
/// arrives is handled at the next frame, half a millisecond later.
pub fn take_calibration_request() -> bool {
    CALIBRATION_REQUEST.try_take().is_some()
}

/// Debounces both buttons and watches for the hold gesture.
#[embassy_executor::task]
pub async fn task(left: Input<'static>, right: Input<'static>) -> ! {
    let mut left = Debounced::new(left);
    let mut right = Debounced::new(right);

    // When the hold starts, and what the ring was showing before it did.
    let mut hold_started: Option<(Instant, Pattern)> = None;

    loop {
        let l = left.sample();
        let r = right.sample();
        LEFT.store(l, Ordering::Relaxed);
        RIGHT.store(r, Ordering::Relaxed);

        match (l && r, hold_started) {
            // Hold begins: remember what to put back if it is abandoned.
            (true, None) => hold_started = Some((Instant::now(), led::current())),

            // Hold continues: show how far along it is, and fire once.
            (true, Some((start, previous))) => {
                let elapsed = start.elapsed();
                if elapsed >= CALIBRATION_HOLD {
                    defmt::info!("both buttons held {} s: requesting calibration", 5);
                    CALIBRATION_REQUEST.signal(());
                    // Hand the ring to the estimator, which owns the display
                    // for the duration of the calibration, and disarm until
                    // the buttons are released.
                    hold_started = None;
                    // Deliberately *not* restoring `previous`: the calibration
                    // is about to take the ring over.
                    let _ = previous;
                    // Wait for release so one long press cannot retrigger.
                    while left.sample() || right.sample() {
                        Timer::after(POLL).await;
                    }
                    LEFT.store(false, Ordering::Relaxed);
                    RIGHT.store(false, Ordering::Relaxed);
                } else {
                    let lit = (elapsed.as_millis() * LED_COUNT as u64
                        / CALIBRATION_HOLD.as_millis()) as u8;
                    led::set(Pattern::Progress {
                        color: led::GREEN,
                        lit: lit.min(LED_COUNT as u8),
                    });
                }
            }

            // Hold abandoned: put back whatever was on the ring before.
            (false, Some((_, previous))) => {
                led::set(previous);
                hold_started = None;
            }

            (false, None) => {}
        }

        Timer::after(POLL).await;
    }
}

/// A pin plus the small amount of state it takes to stop believing bounce.
struct Debounced {
    pin: Input<'static>,
    stable: bool,
    candidate: bool,
    agreed: u8,
}

impl Debounced {
    fn new(pin: Input<'static>) -> Self {
        Self {
            pin,
            stable: false,
            candidate: false,
            agreed: 0,
        }
    }

    /// Read the pin and return the debounced level. Active low.
    fn sample(&mut self) -> bool {
        let raw = self.pin.is_low();
        if raw == self.candidate {
            self.agreed = self.agreed.saturating_add(1);
            if self.agreed >= DEBOUNCE_SAMPLES {
                self.stable = self.candidate;
            }
        } else {
            self.candidate = raw;
            self.agreed = 1;
        }
        self.stable
    }
}
