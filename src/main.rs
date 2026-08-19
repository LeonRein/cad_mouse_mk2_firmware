//! CAD Mouse MK2 firmware: bring-up, wiring, and the sensor readout loop.
//!
//! This file is deliberately thin. It knows the pin numbers and the order
//! things have to start in, and nothing else -- every behaviour lives in a
//! module that can be read on its own:
//!
//! | module            | owns                                              |
//! |-------------------|---------------------------------------------------|
//! | [`sensors`]       | the three TLI493D sensors on the shared I2C bus   |
//! | [`estimator`]     | the pose filter, on core 1, and the core boundary |
//! | [`led`]           | the ring; anything may ask it for a pattern       |
//! | [`buttons`]       | debouncing, and the hold-to-calibrate gesture     |
//! | [`protocol`]      | the debug frame the host decodes                  |
//!
//! The rest calibration itself is in `cadmouse-model`, with the rest of the
//! estimation code, so it can be tested on the host.
//!
//! # Board
//!
//! Seeed XIAO RP2350. The D-pin numbering below is the board's silkscreen; the
//! GPIO numbers are what the code uses.
//!
//! | function          | pin | GPIO |
//! |-------------------|-----|------|
//! | right button      | D0  | 26   |
//! | LED level shifter | D1  | 27   |
//! | left button       | D2  | 28   |
//! | LED data          | D3  | 5    |
//! | I2C1 SDA          | D4  | 6    |
//! | I2C1 SCL          | D5  | 7    |
//! | MAG3 power        | D8  | 2    |
//! | MAG2 power        | D9  | 4    |
//! | MAG1 power        | D10 | 3    |
//!
//! D3 is GPIO5 here and GPIO29 on the XIAO RP2040. Every other D-pin maps
//! identically between the two boards, which makes this the one entry in the
//! table worth double-checking.

#![no_std]
#![no_main]

use core::ptr::addr_of_mut;
use core::sync::atomic::{AtomicBool, Ordering};

use defmt::{error, info, unwrap, warn};
use defmt_rtt as _;
use embassy_executor::{Executor, Spawner};
use embassy_rp::bind_interrupts;
use embassy_rp::gpio::{Level, Output, Pull};
use embassy_rp::i2c::{Config as I2cConfig, I2c, InterruptHandler};
use embassy_rp::multicore::{Stack, spawn_core1};
use embassy_rp::peripherals::{PIO0, USB};
use embassy_rp::pio::{InterruptHandler as PioInterruptHandler, Pio};
use embassy_rp::pio_programs::ws2812::{Grb, PioWs2812, PioWs2812Program};
use embassy_rp::usb::{Driver, InterruptHandler as UsbInterruptHandler};
use embassy_rp::watchdog::Watchdog;
use embassy_sync::mutex::Mutex;
use embassy_time::{Duration, Instant, Timer, with_timeout};
use embassy_usb::UsbDevice;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::class::hid::{Config as HidConfig, HidWriter, State as HidState};
use static_cell::StaticCell;

mod buttons;
mod estimator;
mod hid;
mod led;
mod protocol;
mod sensors;

use led::{LED_COUNT, Pattern};
use protocol::{FRAME_LEN, Frame, status};

bind_interrupts!(struct UsbIrqs {
    USBCTRL_IRQ => UsbInterruptHandler<USB>;
});

bind_interrupts!(struct I2cIrqs {
    I2C1_IRQ => InterruptHandler<embassy_rp::peripherals::I2C1>;
});

bind_interrupts!(struct PioIrqs {
    PIO0_IRQ_0 => PioInterruptHandler<PIO0>;
});

type MyUsbDriver = Driver<'static, USB>;
type MyUsbDevice = UsbDevice<'static, MyUsbDriver>;

/// Stack for core 1.
///
/// The estimator's working set is a few hundred bytes of matrices -- the
/// 87 kB field table is a `static`, not a local -- so this is generous.
static mut CORE1_STACK: Stack<8192> = Stack::new();
static EXECUTOR1: StaticCell<Executor> = StaticCell::new();

/// Halt under a debugger, reset without one.
///
/// `panic-probe`, which this replaces, ends every panic in `udf` -- a
/// deliberate `HardFault` that a probe catches and turns into a backtrace.
/// That is the right behaviour on the bench and the wrong one in a sealed
/// enclosure, where it leaves a device that is indistinguishable from broken
/// hardware and recovers only by being unplugged.
///
/// Rather than choose, ask: `C_DEBUGEN` in `DHCSR` says whether anything is
/// attached. So the same binary breaks into the debugger during development
/// and reboots in the field. A panic that repeats will reboot in a loop, which
/// is visible in the LED bring-up sequence and is a far better symptom than
/// silence.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    static PANICKED: AtomicBool = AtomicBool::new(false);

    // Both guards are taken from `panic-probe`, which handled this properly
    // and which this replaces only to change the *ending*, not the care taken
    // getting there: nothing else may run while a panic is being reported, and
    // if formatting the message panics in turn, the flag stops it recursing
    // until the stack runs out.
    cortex_m::interrupt::disable();
    if !PANICKED.swap(true, Ordering::Relaxed) {
        error!("panic: {}", defmt::Display2Format(info));
    }

    if cortex_m::peripheral::DCB::is_debugger_attached() {
        cortex_m::asm::udf();
    }
    cortex_m::peripheral::SCB::sys_reset()
}

/// How long either core may stop making progress before the chip resets.
///
/// Generous next to anything legitimate: a single readout iteration can
/// legally block for the sensors' 10 ms read timeout on each of three sensors,
/// plus [`USB_WRITE_TIMEOUT`] -- about 40 ms all told. Two seconds is
/// therefore only ever reached by something genuinely wedged, and still
/// recovers fast enough that a user reads it as a hiccup rather than a
/// failure.
const WATCHDOG_PERIOD: Duration = Duration::from_secs(2);

/// How often core 1 is checked for progress.
///
/// The readout loop being alive says nothing about the estimator -- core 1 can
/// fault or wedge while core 0 happily keeps sampling, and the only visible
/// symptom would be a pose that never changes. Feeding the watchdog only while
/// *both* counters advance turns it into a liveness check on the pair.
const CORE1_LIVENESS_WINDOW: Duration = Duration::from_millis(200);

/// Consecutive failed reads before the ring is turned red.
///
/// The watchdog deliberately keeps being fed while reads are merely failing:
/// resetting would not fix a disconnected sensor and would produce a reboot
/// loop instead of a diagnosis. So failure needs its own signal, and at the
/// 1 ms back-off this is roughly half a second of solid failure -- long past
/// anything transient.
const ERRORS_BEFORE_FAULT: u32 = 500;

/// Sensor samples averaged into one measurement for the estimator.
///
/// Not a taste setting -- it follows from measured rates. Core 1 needs 644 us
/// per sample, half as long again as the 406-us filter step itself, because
/// the estimator task around the step lives in flash while only the step is
/// relocated. The readout meanwhile delivers a sample every 340-490 us. So
/// core 1 cannot take every sample, and the question is only what to do about
/// it.
///
/// Note the readout rate is not a constant: 2033 to 2957 Hz have been recorded
/// on the same board, varying with the sensors' own conversion timing rather
/// than with anything in this firmware. At `2` the estimator needs 644 us of
/// every second sample interval, which is comfortable at 2100 Hz and tight at
/// the 2957 Hz end. If the `stream:` line ever shows the estimator at less
/// than half the readout, that margin has run out and this wants to be `3`.
///
/// The steady-state posterior of a random walk goes as `sqrt(T * R)`, with `T`
/// the interval between filter updates and `R` the variance of what it is fed.
/// All three columns below are measured on target, `sigma` by the device's own
/// rest calibration:
///
/// | | readout | estimator | dropped | `T` | `sigma` | `sqrt(T*R)` |
/// |---|---|---|---|---|---|---|
/// | `1` | 2040 Hz | 1553 Hz | 24 % | 644 us | 0.944 | 23.9 |
/// | **`2`** | 2120 Hz | 1060 Hz | **0 %** | 943 us | **0.707** | **21.7** |
/// | `3` (predicted) | ~2150 Hz | ~717 Hz | 0 % | 1395 us | 0.617 | 23.0 |
///
/// `2` is the optimum, and it is a shallow one -- 9 % in variance, under 5 % in
/// standard deviation. The stronger argument is the `dropped` column: at `2`
/// the estimator consumes exactly every second sample, so `dt` is uniform,
/// which is what the random-walk process model assumes and what a sporadic
/// 24 % drop quietly violates.
///
/// The mean is rounded back to `i16` so nothing downstream changes and the
/// rest calibration keeps its exact integer statistics. That rounding adds
/// 1/12 count^2, which is why `sigma` lands at 0.707 rather than the ideal
/// 0.667 -- but the calibration then measures 0.707 directly, so `R` stays
/// honest without anybody scaling it by hand. Measuring what is actually fed
/// to the filter is worth more than the last few per cent.
///
/// The cost is half a window of group delay, about 240 us. The HID tick is
/// 1 ms and the axes are absolute rather than incremental, so it is not
/// observable.
///
/// Re-derive this if either rate moves: the `stream:` line prints all of
/// readout, estimator and `sigma`.
const SAMPLES_PER_ESTIMATE: u32 = 2;

/// Integer divide, rounding halves away from zero.
///
/// `i32` throughout: the counts are integers and the sum of a handful of them
/// cannot overflow, so there is no reason to route the mean through `f32` and
/// no rounding beyond the single one made here.
fn rounded_div(sum: i32, n: i32) -> i16 {
    let half = n / 2;
    let quotient = if sum >= 0 {
        (sum + half) / n
    } else {
        (sum - half) / n
    };
    quotient as i16
}

/// Longest a single debug frame may spend waiting for the host.
///
/// The readout loop must not be hostage to whatever is on the other end of the
/// USB cable. A host that opens the port and then stops reading would
/// otherwise stall the sensor loop, and with it the estimator -- so a frame
/// the host will not take is dropped, and the gap shows up in `seq`.
const USB_WRITE_TIMEOUT: Duration = Duration::from_millis(10);

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    // Read before anything else can clear it, and log it unconditionally: on a
    // device with no probe this is the only way to tell an ordinary power-up
    // from a watchdog reset or a panic reboot, and "it restarted itself" is
    // otherwise invisible. `Timer` means a reset that the watchdog caused; the
    // debug stream carries it to the host, so a field unit can be asked.
    let mut watchdog = Watchdog::new(p.WATCHDOG);
    // `ResetReason` has no `defmt::Format`, so it is matched rather than
    // formatted -- which is no loss, since the two cases want different words.
    match watchdog.reset_reason() {
        Some(embassy_rp::watchdog::ResetReason::TimedOut) => {
            warn!("boot: the watchdog reset the last run -- something wedged")
        }
        Some(embassy_rp::watchdog::ResetReason::Forced) => {
            warn!("boot: the last run was reset deliberately")
        }
        None => info!("boot: power-on or external reset"),
    }
    // The default, set explicitly because it is load-bearing during
    // development: without it, halting the core at a breakpoint lets the
    // watchdog expire and reset the target out from under the debugger.
    watchdog.pause_on_debug(true);

    // ── LED ring ──
    // Started first, so that everything after this point has a way to say what
    // it is doing. White while booting.
    let led_level_shifter = Output::new(p.PIN_27, Level::High);
    let Pio {
        mut common, sm0, ..
    } = Pio::new(p.PIO0, PioIrqs);
    let ws2812_program = PioWs2812Program::new(&mut common);
    let ws2812: PioWs2812<'_, PIO0, 0, LED_COUNT, Grb> =
        PioWs2812::new(&mut common, sm0, p.DMA_CH0, p.PIN_5, &ws2812_program);
    unwrap!(spawner.spawn(led::task(led_level_shifter, ws2812)));
    led::set(Pattern::Solid(led::WHITE));

    // ── Buttons ──
    let right = embassy_rp::gpio::Input::new(p.PIN_26, Pull::Up);
    let left = embassy_rp::gpio::Input::new(p.PIN_28, Pull::Up);
    unwrap!(spawner.spawn(buttons::task(left, right)));

    // ── USB ──
    let driver = Driver::new(p.USB, UsbIrqs);

    // pid.codes' community vendor ID with one of its test product IDs.
    //
    // Deliberately not 3Dconnexion's `256f:c631`. Claiming it makes the
    // vendor's own driver bind, which is the usual reason to do it -- but on
    // Linux nothing needs that: `spacenavd` will drive any device given its ID
    // in `/etc/spnavrc`, so there is no reason to answer to another company's
    // name.
    //
    // `1209:0001` is a *test* PID, fine for a personal build. A permanent one
    // is free from pid.codes for an open-source project, which is the only
    // condition attached to the VID.
    let config = {
        let mut config = embassy_usb::Config::new(0x1209, 0x0001);
        config.manufacturer = Some("lr-net");
        config.product = Some("CAD Mouse MK2");
        config.serial_number = Some("00000001");
        config.max_power = 100;
        config.max_packet_size_0 = 64;
        config.composite_with_iads = true;
        config.device_class = 0xEF;
        config.device_sub_class = 0x02;
        config.device_protocol = 0x01;
        config
    };

    let mut builder = {
        static CONFIG_DESCRIPTOR: StaticCell<[u8; 512]> = StaticCell::new();
        static BOS_DESCRIPTOR: StaticCell<[u8; 256]> = StaticCell::new();
        static CONTROL_BUF: StaticCell<[u8; 128]> = StaticCell::new();
        embassy_usb::Builder::new(
            driver,
            config,
            CONFIG_DESCRIPTOR.init([0; 512]),
            BOS_DESCRIPTOR.init([0; 256]),
            &mut [],
            CONTROL_BUF.init([0; 128]),
        )
    };

    // CDC ACM carrying the debug stream: raw counts and the pose built from
    // them. This is what `scripts/record.py` and `scripts/view.py` read, and
    // it stays even once HID exists -- a session recorded through it can be
    // re-filtered on the host with different tuning.
    let mut data_class = {
        static STATE: StaticCell<State> = StaticCell::new();
        let state = STATE.init(State::new());
        CdcAcmClass::new(&mut builder, state, 64)
    };

    // The HID interface: what actually makes this a mouse. Declared after the
    // CDC so the debug stream keeps its interface numbers; the composite is
    // harmless to `spacenavd`, which binds to the HID interface and ignores
    // the rest.
    let hid_writer = {
        static STATE: StaticCell<HidState> = StaticCell::new();
        let state = STATE.init(HidState::new());
        HidWriter::new(
            &mut builder,
            state,
            HidConfig {
                report_descriptor: hid::REPORT_DESCRIPTOR,
                request_handler: None,
                poll_ms: 1,
                max_packet_size: hid::MAX_PACKET_SIZE as u16,
            },
        )
    };

    unwrap!(spawner.spawn(usb_task(builder.build())));
    unwrap!(spawner.spawn(hid::task(hid_writer)));

    // ── Core 1: the estimator ──
    // Started before the sensors so that the 87 kB flash-to-RAM copy of the
    // field table overlaps with sensor bring-up rather than following it.
    spawn_core1(
        p.CORE1,
        // SAFETY: taken once, and core 1 is spawned exactly once.
        unsafe { &mut *addr_of_mut!(CORE1_STACK) },
        move || {
            let executor = EXECUTOR1.init(Executor::new());
            executor.run(|spawner| unwrap!(spawner.spawn(estimator::task())));
        },
    );

    // ── Sensors ──
    info!("Initializing sensors…");

    let i2c_cfg = {
        let mut c = I2cConfig::default();
        c.frequency = 1_000_000;
        c
    };
    // `new_async` takes (peripheral, scl, sda, ...).
    let i2c = I2c::new_async(p.I2C1, p.PIN_7, p.PIN_6, I2cIrqs, i2c_cfg);

    let mag1_pwr = Output::new(p.PIN_3, Level::Low);
    let mag2_pwr = Output::new(p.PIN_4, Level::Low);
    let mag3_pwr = Output::new(p.PIN_2, Level::Low);

    static I2C_BUS: StaticCell<sensors::SharedBus<embassy_rp::peripherals::I2C1>> =
        StaticCell::new();
    let bus = I2C_BUS.init(Mutex::new(i2c));

    let sensors_init = with_timeout(
        Duration::from_secs(3),
        sensors::Sensors::init(bus, mag1_pwr, mag2_pwr, mag3_pwr),
    )
    .await;

    let mut sensors = match sensors_init {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            warn!("Sensor init error: {}", defmt::Debug2Format(&e));
            fault().await
        }
        Err(_) => {
            warn!("Sensor init timed out");
            fault().await
        }
    };

    // ── Readout ──
    //
    // Deliberately unpaced. Reads block on clock stretching until the sensor
    // has converted, so the loop already self-clocks at the rate the hardware
    // sustains. A device-side timer here would be a second clock beating
    // against it, and `Ticker` compounds that by never re-syncing to `now`, so
    // any stall is repaid as an unpaced burst of back-to-back samples that
    // breaks the uniform sampling the estimator assumes.
    //
    // `seq` counts every attempted sample, successful or not, so a gap in the
    // host's log unambiguously means a lost frame.
    //
    // # Why the estimator gets a mean and not every sample
    //
    // This loop runs faster than the estimator can consume it -- measured
    // around 2957 Hz against a 391 us filter step -- so [`estimator::submit`],
    // which keeps only the newest value, used to drop roughly one sample in
    // six. That is the worst of both worlds: the bus time and the conversion
    // time are paid for every sample and the information in the dropped ones
    // is thrown away.
    //
    // Averaging instead is free and strictly better. For a random walk with
    // process PSD `q` observed with variance `R` every `T`, the steady-state
    // posterior is `sqrt(q*T*R)`. Dropping every second sample gives
    // `sqrt(q*2T*R)` -- 19 % more standard deviation. Averaging pairs gives
    // `sqrt(q*2T*R/2)`, which is exactly the full-rate figure back again.
    //
    // The mean is rounded back to `i16` so that nothing downstream changes,
    // and so the rest calibration keeps its exact integer statistics. That
    // rounding costs a little of what averaging buys -- it adds 1/12 count^2
    // of quantisation, so two samples of 1.0-count noise land at 0.77 rather
    // than the ideal 0.71 -- but the calibration then measures 0.77 directly,
    // so `R` stays honest without anybody scaling it by hand. Measuring what
    // is actually fed to the filter is worth more than the last 8 %.
    //
    // The cost is a half-window of group delay, about 170 us. The HID tick is
    // 1 ms and the axes are absolute rather than incremental, so it is not
    // observable.
    let mut seq: u16 = 0;
    let mut accumulator = [0i32; 9];
    let mut accumulated: u32 = 0;
    let mut errors: u32 = 0;
    let mut consecutive_errors: u32 = 0;
    let mut sent: u32 = 0;
    let mut window_start = Instant::now();
    let mut window_samples: u32 = 0;
    let mut window_steps: u32 = estimator::steps_completed();

    // Started only once the sensors are up: bring-up has its own three-second
    // timeout and ends in `fault()`, which is a diagnosed stop with a red ring
    // rather than something a reset would improve.
    watchdog.start(WATCHDOG_PERIOD);
    let mut core1_steps = estimator::steps_completed();
    let mut core1_deadline = Instant::now() + CORE1_LIVENESS_WINDOW;
    let mut core1_alive = true;
    let mut fault_shown = false;

    loop {
        // Before the fallible read, so that a failing sensor does not also
        // starve the watchdog -- see `ERRORS_BEFORE_FAULT`.
        let now = Instant::now();
        if now >= core1_deadline {
            let steps = estimator::steps_completed();
            core1_alive = steps != core1_steps;
            core1_steps = steps;
            core1_deadline = now + CORE1_LIVENESS_WINDOW;
        }
        if core1_alive {
            watchdog.feed();
        }

        let counts = match sensors.read_raw().await {
            Ok(r) => r,
            Err(e) => {
                // Under master-controlled triggering a read that yields
                // nothing is a genuine fault, not the routine "sensor hasn't
                // converted yet" that the old free-running configuration
                // produced constantly. Log the first, then sample, so a
                // persistent fault is visible without flooding the transport.
                errors += 1;
                consecutive_errors += 1;
                if errors == 1 || errors % 256 == 0 {
                    warn!("read error #{} (sent {}): {}", errors, sent, e);
                }
                if consecutive_errors == ERRORS_BEFORE_FAULT && !fault_shown {
                    warn!("{} consecutive read errors; ring to red", consecutive_errors);
                    led::set(Pattern::Solid(led::RED));
                    fault_shown = true;
                }
                seq = seq.wrapping_add(1);
                // Back off only on the error path: a failing read can return
                // in microseconds, and without this the loop would spin on the
                // bus and starve everything else.
                Timer::after_millis(1).await;
                continue;
            }
        };

        consecutive_errors = 0;
        if fault_shown {
            // Core 1 owns the ring during a calibration and will reassert its
            // own pattern on its next state change, so restoring green here is
            // at worst briefly wrong and never sticky.
            info!("sensor reads recovered");
            led::set(Pattern::Solid(led::GREEN));
            fault_shown = false;
        }

        let t_us = Instant::now().as_micros() as u32;

        for (slot, &c) in accumulator.iter_mut().zip(counts.iter()) {
            *slot += c as i32;
        }
        accumulated += 1;
        if accumulated >= SAMPLES_PER_ESTIMATE {
            let mut mean = [0i16; 9];
            for (m, &sum) in mean.iter_mut().zip(accumulator.iter()) {
                *m = rounded_div(sum, SAMPLES_PER_ESTIMATE as i32);
            }
            // `t_us` is the last sample of the window rather than its middle:
            // the interval between submissions is what `predict` integrates,
            // and that is the same either way. The constant offset is the
            // group delay noted above.
            estimator::submit(estimator::Sample {
                seq,
                t_us,
                counts: mean,
            });
            accumulator = [0; 9];
            accumulated = 0;
        }

        // Whatever core 1 has made of the recent past. Not necessarily this
        // sample -- see `estimator` on why that is the intended behaviour.
        let estimate = estimator::latest();
        let frame = Frame {
            seq,
            t_us,
            counts,
            pose: estimate.map(|e| e.pose).unwrap_or_default(),
            nis: estimate.map(|e| e.nis).unwrap_or_default(),
            // Button state is read here rather than carried through the
            // estimator: it is not part of the estimate, and core 1 runs
            // behind this loop, so routing it through would report presses
            // late.
            status: estimate.map(|e| e.status).unwrap_or_default()
                | if buttons::left_pressed() { status::BUTTON_LEFT } else { 0 }
                | if buttons::right_pressed() { status::BUTTON_RIGHT } else { 0 },
            progress: estimate.map(|e| e.progress).unwrap_or_default(),
        };

        if data_class.dtr() {
            let mut buf = [0u8; FRAME_LEN];
            frame.encode(&mut buf);
            match with_timeout(USB_WRITE_TIMEOUT, data_class.write_packet(&buf)).await {
                Ok(Ok(())) => sent += 1,
                Ok(Err(_)) => {} // host went away; keep sampling
                Err(_) => {}     // host stopped reading; drop the frame
            }
        }

        seq = seq.wrapping_add(1);
        window_samples += 1;

        // Report the achieved rate independently of the host, so the readout
        // can be judged without trusting record.py.
        let elapsed = window_start.elapsed();
        if elapsed >= Duration::from_secs(5) {
            let hz = window_samples * 1000 / elapsed.as_millis() as u32;
            let steps_now = estimator::steps_completed();
            let est_hz = steps_now.wrapping_sub(window_steps) * 1000 / elapsed.as_millis() as u32;
            window_steps = steps_now;
            let stale = estimate
                .map(|e| seq.wrapping_sub(e.seq))
                .unwrap_or(u16::MAX);
            info!(
                "stream: {} Hz, estimator {} Hz, {} sent, {} errors, {} frames behind, DIAG={:02x}",
                hz,
                est_hz,
                sent,
                errors,
                stale,
                sensors.diag_bytes()
            );
            window_start = Instant::now();
            window_samples = 0;
        }
    }
}

/// Nothing useful can happen without sensors, so say so and stop.
///
/// A solid red ring rather than a silent hang: the one thing worse than a dead
/// device is a dead device that looks like a working one.
async fn fault() -> ! {
    led::set(Pattern::Solid(led::RED));
    loop {
        Timer::after_secs(60).await;
    }
}

#[embassy_executor::task]
async fn usb_task(mut usb: MyUsbDevice) -> ! {
    usb.run().await
}
