//! The two side buttons, and the gesture that starts a rest calibration.
//!
//! Both buttons are active low with the internal pull-up. D0/GPIO26 is the
//! right button and D2/GPIO28 the left.
//!
//! Two things come out of here:
//!
//! * **Press state**, in [`left_pressed`] / [`right_pressed`], for whatever
//!   wants it. Nothing does yet -- the HID report that will carry these is not
//!   implemented -- so they exist to be read, not to be acted on.
//! * **The hold gesture**, which has two stages. This module owns it end to
//!   end, including its LED feedback, because a hold that gives no sign of
//!   being noticed until it fires is indistinguishable from a dead button.
//!
//! # The gesture
//!
//! Hold both buttons. The ring fills **green** over five seconds; let go
//! before it is full and nothing happens. At five seconds the ring turns
//! **solid blue** -- releasing now starts a rest calibration. Keep holding and
//! **yellow** fills over the blue for another five seconds. When the yellow
//! ring is full the whole ring turns **red**: the hold is done, let go, and
//! the device reboots into the USB bootloader.
//!
//! Two things about that shape are deliberate. The colour says what releasing
//! *now* would do, and the fill says how far away the next thing is, so at no
//! point does the user have to count seconds. And the calibration fires on
//! **release**, not on reaching five seconds -- otherwise there would be no
//! way to pass through the calibration point on the way to the bootloader
//! without also triggering a calibration.

use core::sync::atomic::{AtomicBool, Ordering};

use embassy_rp::gpio::Input;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::signal::Signal;
use embassy_time::{Duration, Instant, Timer};

use crate::led::{self, LED_COUNT, Pattern};

/// How long both buttons must be held before releasing them asks for a
/// calibration.
pub const CALIBRATION_HOLD: Duration = Duration::from_secs(5);

/// How long both buttons must be held, in total, to reboot into the USB
/// bootloader.
///
/// Twice the calibration hold, and reached only by holding *through* the point
/// where the ring has already gone blue and is filling yellow. Nobody arrives
/// here without having been told twice what is about to happen.
pub const BOOTLOADER_HOLD: Duration = Duration::from_secs(10);

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

            // Hold continues: show how far along it is, and which of the two
            // things releasing would do.
            (true, Some((start, _))) => {
                let elapsed = start.elapsed();
                if elapsed >= BOOTLOADER_HOLD {
                    enter_bootloader().await;
                } else if elapsed >= CALIBRATION_HOLD {
                    // Past the calibration point: yellow filling over blue. At
                    // `lit == 0` this renders as the solid blue that marks the
                    // moment the calibration became available, so the two
                    // stages need no special case between them.
                    led::set(Pattern::Progress {
                        color: led::YELLOW,
                        background: led::BLUE,
                        lit: filled(
                            elapsed - CALIBRATION_HOLD,
                            BOOTLOADER_HOLD - CALIBRATION_HOLD,
                        ),
                    });
                } else {
                    led::set(Pattern::Progress {
                        color: led::GREEN,
                        background: led::OFF,
                        lit: filled(elapsed, CALIBRATION_HOLD),
                    });
                }
            }

            // Released. Past the calibration point that is the request;
            // short of it the gesture was abandoned and the ring goes back to
            // whatever it was showing.
            (false, Some((start, previous))) => {
                if start.elapsed() >= CALIBRATION_HOLD {
                    defmt::info!("buttons released past the hold: requesting calibration");
                    CALIBRATION_REQUEST.signal(());
                    // Deliberately *not* restoring `previous`: the calibration
                    // is about to take the ring over.
                } else {
                    led::set(previous);
                }
                hold_started = None;
            }

            (false, None) => {}
        }

        Timer::after(POLL).await;
    }
}

/// Pixels to light for `elapsed` of the way through `span`.
///
/// Rounds *up*, which is not a rounding preference but the difference between
/// a usable indicator and a misleading one. Rounding down, the first pixel
/// lights 625 ms after the press -- long enough to read as a dead button --
/// and, worse, the ring reaches full only at the instant the stage ends, so a
/// full ring is a state the user passes through rather than one they see.
/// Rounding up acknowledges the press on the first frame and holds the ring
/// full for the last pixel-time of the stage, which is what makes "it is full,
/// I can let go" a thing anyone can act on.
fn filled(elapsed: Duration, span: Duration) -> u8 {
    let span_ms = span.as_millis().max(1);
    let lit = (elapsed.as_millis() * LED_COUNT as u64).div_ceil(span_ms);
    lit.min(LED_COUNT as u64) as u8
}

/// Reboot into the ROM's USB bootloader, from which a UF2 can be copied on.
///
/// This is what makes the device updatable once it is sealed and the debug
/// probe is gone -- see `scripts/mkuf2.sh`.
async fn enter_bootloader() -> ! {
    defmt::info!(
        "both buttons held {} s: rebooting into the USB bootloader",
        10
    );

    // Red, and red only here: it is the one colour in the gesture that means
    // "done, let go" rather than "keep holding". Without it the last thing the
    // user sees is a yellow ring that looks no different from a yellow ring
    // one pixel ago, and the only way to learn that the hold had finished
    // would be to notice the device drop off the bus.
    //
    // The pause is long enough to read as a deliberate state change rather
    // than a flicker. WS2812s hold the last value clocked into them, so the
    // red then stays lit across the reboot and through the whole time the
    // device sits in the bootloader -- which is also the only sign it gives
    // that it is waiting for a UF2 rather than dead.
    led::set(Pattern::Solid(led::RED));
    Timer::after(Duration::from_millis(400)).await;

    // The watchdog goes first, and this is not belt-and-braces. It is running
    // on a two-second period fed by the readout loop, and that loop is about
    // to stop existing. Whether the bootrom leaves it armed across a BOOTSEL
    // reboot is not something to discover from a device that drops off the bus
    // two seconds after the user reaches for it, so disable it rather than
    // assume. This is exactly what `embassy_rp`'s own private `Watchdog::
    // enable` does; it is reached through the PAC only because that method is
    // not public and the `Watchdog` lives in `main`.
    embassy_rp::pac::WATCHDOG
        .ctrl()
        .modify(|w| w.set_enable(false));

    // No activity LED -- the ring is not a GPIO the bootrom knows how to blink
    // -- and neither interface disabled: mass storage for dragging a UF2 on,
    // PICOBOOT so `picotool` still works.
    embassy_rp::rom_data::reset_to_usb_boot(0, 0);

    // `reset_to_usb_boot` passes `REBOOT2_FLAG_NO_RETURN_ON_SUCCESS`, so it
    // does not come back -- but its signature does not say so, and this
    // function promises `!`.
    loop {
        cortex_m::asm::wfi();
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
