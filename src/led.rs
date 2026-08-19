//! The LED ring: one task owns the hardware, everyone else publishes intent.
//!
//! Several unrelated parts of the firmware have something to say with the ring
//! -- boot progress, the button hold, the calibration running on the other
//! core, a sensor fault. If each of them drove the PIO directly they would
//! need the peripheral handle passed around, they would fight over it, and the
//! answer to "why is it blue right now" would be spread across five files.
//!
//! So the pattern is a *value*. Anything, on either core, calls [`set`]; the
//! task here renders whatever the current value is. Writes are last-one-wins,
//! which is fine because the states are mutually exclusive by construction --
//! the device is calibrating, or faulted, or running, never two at once.
//!
//! The physical ring is eight WS2812s on D3/GPIO5, driven through PIO, with a
//! level-shifter enable on D1/GPIO27.
//!
//! Note D3 is GPIO5 on the XIAO RP2350, *not* GPIO29 as on the XIAO RP2040 the
//! original C firmware's pin table was written for. The two boards share the
//! same D-to-GPIO mapping everywhere else; D3 was remapped.

use core::cell::Cell;

use embassy_rp::gpio::Output;
use embassy_rp::peripherals::PIO0;
use embassy_rp::pio_programs::ws2812::{Grb, PioWs2812};
use embassy_sync::blocking_mutex::Mutex as BlockingMutex;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::signal::Signal;
use embassy_time::{Duration, Instant, Timer};
use smart_leds::RGB8;

/// Pixels on the ring (original firmware `Config::LED_COUNT`).
pub const LED_COUNT: usize = 8;

/// Animation step. Fast enough that the spinner looks smooth, slow enough that
/// the task is invisible in the CPU budget.
const FRAME_INTERVAL: Duration = Duration::from_millis(30);

/// Brightness cap. These are bright enough to be uncomfortable at full scale
/// under a hand, and every colour below is written at this level.
const LEVEL: u8 = 24;

pub const GREEN: RGB8 = RGB8::new(0, LEVEL, 0);
pub const BLUE: RGB8 = RGB8::new(0, 0, LEVEL);
pub const RED: RGB8 = RGB8::new(LEVEL, 0, 0);
pub const WHITE: RGB8 = RGB8::new(LEVEL / 2, LEVEL / 2, LEVEL / 2);
pub const YELLOW: RGB8 = RGB8::new(LEVEL, LEVEL, 0);
/// A dark pixel, for use as a [`Pattern::Progress`] background.
pub const OFF: RGB8 = RGB8::new(0, 0, 0);

/// What the ring should be doing.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // `Off` is for the sleep state, which waits for HID.
pub enum Pattern {
    Off,
    /// Every pixel the same colour.
    Solid(RGB8),
    /// All pixels on and off together, `period_ms` for a full cycle.
    Blink { color: RGB8, period_ms: u32 },
    /// One lit pixel walking around the ring. The calibration indicator, as in
    /// the original firmware's `CalibratingState`.
    Spinner(RGB8),
    /// The first `lit` pixels in `color`, the rest left showing `background`
    /// -- a progress bar bent into a circle.
    ///
    /// The background is what lets one gesture have two stages: the button
    /// hold fills green on black up to the calibration point, then yellow on
    /// blue up to the bootloader one, so the ring says both "how far along am
    /// I" and "what will happen if I let go now" at the same time.
    Progress {
        color: RGB8,
        background: RGB8,
        lit: u8,
    },
}

/// The current pattern. A `Signal` rather than a channel: only the newest
/// value can matter, and a sender must never block behind the renderer.
static PATTERN: Signal<CriticalSectionRawMutex, Pattern> = Signal::new();

/// Shadow copy of what was last requested, so [`current`] can answer without
/// consuming the signal the renderer is waiting on.
static CURRENT: BlockingMutex<CriticalSectionRawMutex, Cell<Pattern>> =
    BlockingMutex::new(Cell::new(Pattern::Solid(WHITE)));

/// Ask the ring to show `pattern`. Safe from either core, from any task, and
/// never blocks.
pub fn set(pattern: Pattern) {
    CURRENT.lock(|c| c.set(pattern));
    PATTERN.signal(pattern);
}

/// What the ring was last asked to show.
///
/// For the caller that wants to interrupt the display briefly and put back
/// whatever was there -- the button hold does exactly this.
pub fn current() -> Pattern {
    CURRENT.lock(|c| c.get())
}

/// Renders whatever [`set`] last asked for.
///
/// `_level_shifter` is held for the task's lifetime so the enable pin stays
/// driven; dropping the `Output` would let it float and the ring would go dark
/// for reasons that look like a data problem.
#[embassy_executor::task]
pub async fn task(
    _level_shifter: Output<'static>,
    mut ws2812: PioWs2812<'static, PIO0, 0, LED_COUNT, Grb>,
) -> ! {
    let mut pattern = Pattern::Solid(WHITE);
    let started = Instant::now();

    loop {
        if let Some(next) = PATTERN.try_take() {
            pattern = next;
        }

        let elapsed_ms = started.elapsed().as_millis() as u32;
        let pixels = render(pattern, elapsed_ms);
        ws2812.write(&pixels).await;
        Timer::after(FRAME_INTERVAL).await;
    }
}

/// The whole of the drawing, separated from the hardware so it can be reasoned
/// about (and tested) without a board.
fn render(pattern: Pattern, elapsed_ms: u32) -> [RGB8; LED_COUNT] {
    let off = RGB8::default();
    match pattern {
        Pattern::Off => [off; LED_COUNT],
        Pattern::Solid(color) => [color; LED_COUNT],
        Pattern::Blink { color, period_ms } => {
            let on = period_ms > 0 && (elapsed_ms % period_ms) * 2 < period_ms;
            [if on { color } else { off }; LED_COUNT]
        }
        Pattern::Spinner(color) => {
            let mut pixels = [off; LED_COUNT];
            let index = (elapsed_ms / 60) as usize % LED_COUNT;
            pixels[index] = color;
            pixels
        }
        Pattern::Progress {
            color,
            background,
            lit,
        } => {
            let mut pixels = [background; LED_COUNT];
            // `take` rather than an index test: it saturates on its own, so a
            // caller that computes `lit` past the end of the ring gets a full
            // ring instead of a panic.
            for pixel in pixels.iter_mut().take(lit as usize) {
                *pixel = color;
            }
            pixels
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn progress_lights_exactly_the_requested_pixels() {
        let pixels = render(
            Pattern::Progress {
                color: GREEN,
                background: OFF,
                lit: 3,
            },
            0,
        );
        assert_eq!(pixels[2], GREEN);
        assert_eq!(pixels[3], OFF);
    }

    #[test]
    fn progress_leaves_the_background_where_it_is_not_filled() {
        let pixels = render(
            Pattern::Progress {
                color: YELLOW,
                background: BLUE,
                lit: 2,
            },
            0,
        );
        assert_eq!(pixels[1], YELLOW);
        assert_eq!(pixels[2], BLUE, "the unfilled part carries the background");
    }

    #[test]
    fn progress_saturates_rather_than_panicking_past_the_end() {
        let pixels = render(
            Pattern::Progress {
                color: YELLOW,
                background: BLUE,
                lit: 200,
            },
            0,
        );
        assert_eq!(pixels, [YELLOW; LED_COUNT]);
    }

    #[test]
    fn an_empty_fill_is_all_background() {
        let pixels = render(
            Pattern::Progress {
                color: YELLOW,
                background: BLUE,
                lit: 0,
            },
            0,
        );
        assert_eq!(
            pixels, [BLUE; LED_COUNT],
            "the moment the hold passes the calibration point"
        );
    }

    #[test]
    fn spinner_walks_all_the_way_round() {
        let first = render(Pattern::Spinner(BLUE), 0);
        let later = render(Pattern::Spinner(BLUE), 60 * LED_COUNT as u32);
        assert_eq!(first, later, "one full lap should return to the start");
    }

    #[test]
    fn blink_is_on_for_half_its_period() {
        let p = Pattern::Blink { color: RED, period_ms: 400 };
        assert_eq!(render(p, 0)[0], RED);
        assert_eq!(render(p, 300)[0], RGB8::default());
    }
}
