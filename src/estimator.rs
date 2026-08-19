//! The pose estimator, and the only two things that cross between the cores.
//!
//! # The split
//!
//! Core 0 does I/O: the I2C readout, USB, the LEDs, the buttons. Core 1 does
//! nothing but arithmetic -- one iterated EKF step per sample, and the rest
//! calibration that occasionally runs alongside it.
//!
//! The reason is the deadline. A filter step costs a measured 60 892 cycles,
//! or 406 us at 150 MHz, and that is a *hard* deadline in a way that a USB
//! transfer is not: a late packet is retried, a late filter step is a sample
//! thrown away. Sharing a core with a stack that has its own interrupt-driven
//! timing means the two would occasionally collide, and the collision would
//! look like sensor noise.
//!
//! # The rate is not 2 kHz, and the step is not the whole cost
//!
//! Earlier revisions of this comment quoted a 2 kHz deadline and a 75 000-cycle
//! budget. Neither survives measurement. The readout loop is unpaced -- it runs
//! at whatever the I2C clock stretching allows -- and that is about 2100 Hz
//! with no debug host attached.
//!
//! More importantly, `filter_step` is not what this core costs per sample.
//! The step is 406 us; the task around it takes **644 us**, measured by the
//! `estimator` figure in the readout loop's rate log. The difference is
//! everything in the loop below, and all of it executes from flash -- only
//! `filter_step` is relocated. Optimising the step alone therefore has a
//! ceiling, and the 238 us outside it is where the remaining work is.
//!
//! Beware of measuring one core's rate and inferring the other's. The two are
//! coupled through [`ESTIMATE`]'s critical section, and the readout rate also
//! varies with the sensors' own conversion timing: figures between 2033 and
//! 2957 Hz were recorded across one session. Only ever compare rates from the
//! same run, and prefer the counters over `seq` lag.
//!
//! Core 0 therefore averages `SAMPLES_PER_ESTIMATE` readings and hands this
//! core one measurement it can always take, which sets the rate by
//! construction rather than by a race between the two -- see that constant for
//! the measurements behind the choice. Nothing here scales `R` to match: the
//! rest calibration measures the noise of the stream it is actually fed.
//!
//! # What crosses, and what happens when core 1 falls behind
//!
//! Two values, and both are *latest-value*, never queues:
//!
//! * [`submit`] hands core 0's newest sample to core 1.
//! * [`latest`] lets core 0 read core 1's newest estimate without waiting.
//!
//! If core 1 is late, [`submit`] overwrites the pending sample and core 1
//! simply never sees the one it missed. This is the right policy and not a
//! compromise: a queue would trade a dropped sample for a growing latency, and
//! a pose estimate that arrives late is not merely less useful than a fresh
//! one, it is wrong -- it describes where the knob *was*. Dropped samples show
//! up as a jump in `seq`, so the host can always tell.
//!
//! Note this also means core 1 never blocks core 0, so a fault on core 1
//! cannot stall the readout. It shows up instead as an estimate whose `seq`
//! stops advancing, which core 0 can see.

use core::cell::Cell;
use core::ptr::addr_of_mut;
use core::sync::atomic::{AtomicU32, Ordering};

use cadmouse_model::generated as consts;
use cadmouse_model::magnet::FieldTable;
use cadmouse_model::model::{MEAS_DIM, POSE_DIM, PoseModel};
use cadmouse_model::rest::{Abort, Calibration, Deadzone, RestCalibration, Step};
use cadmouse_model::tuning;
use defmt::{info, warn};
use embassy_sync::blocking_mutex::Mutex as BlockingMutex;
use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::signal::Signal;
use embassy_time::{Duration, Instant};
use iekf::IteratedEkf;

use crate::buttons;
use crate::led::{self, Pattern};
use crate::protocol::status;

/// One sensor sample, core 0 to core 1.
#[derive(Clone, Copy)]
pub struct Sample {
    pub seq: u16,
    pub t_us: u32,
    pub counts: [i16; MEAS_DIM],
}

/// One pose estimate, core 1 to core 0.
#[derive(Clone, Copy)]
pub struct Estimate {
    /// The sample this came from, so core 0 can tell fresh from stale.
    pub seq: u16,
    /// Zeroed and deadzoned: what the HID axes will be built from.
    pub pose: [f32; POSE_DIM],
    pub nis: f32,
    pub status: u8,
    pub progress: u8,
}

/// Filter steps completed since boot.
///
/// Core 1's own rate, which is not derivable from anything core 0 can see: the
/// readout rate says how fast samples are produced, and `seq` lag says how
/// stale the newest estimate is, but neither says how often this core actually
/// finishes a step. Deciding how many samples to average needs exactly that
/// number, so it is measured rather than inferred.
///
/// Relaxed, and read only as a difference over a five-second window -- a stale
/// or torn read costs nothing here.
static STEPS: AtomicU32 = AtomicU32::new(0);

static SAMPLE: Signal<CriticalSectionRawMutex, Sample> = Signal::new();
static ESTIMATE: BlockingMutex<CriticalSectionRawMutex, Cell<Option<Estimate>>> =
    BlockingMutex::new(Cell::new(None));

/// Hand a sample to core 1. Never blocks; overwrites anything not yet taken.
pub fn submit(sample: Sample) {
    SAMPLE.signal(sample);
}

/// The newest estimate, or `None` before the first one.
pub fn latest() -> Option<Estimate> {
    ESTIMATE.lock(|cell| cell.get())
}

/// Filter steps completed since boot; see [`STEPS`].
pub fn steps_completed() -> u32 {
    STEPS.load(Ordering::Relaxed)
}

/// The field table, copied out of flash into RAM.
///
/// Worth 1.65x on the measurement function -- 12 298 cycles against 20 331 --
/// because the bicubic's nine gathers per evaluation otherwise miss the XIP
/// cache. 87 kB against 520 kB of SRAM is a cheap trade for a third of the
/// filter's cost.
static mut TABLE_RAM: [u8; consts::TABLE_BYTES] = [0; consts::TABLE_BYTES];

/// One filter step, executing from SRAM rather than from flash.
///
/// This placement is worth more than every other optimisation in the
/// estimator put together, and it is not obvious, so: the RP2350 executes from
/// external QSPI flash through a small XIP cache. The measurement model and
/// the filter's linear algebra each fit in that cache on their own, but
/// together they do not, and every pass refetches from flash. Measured, one
/// whole step at the shipping iteration count:
///
/// | | cycles |
/// |---|---|
/// | model and filter, from flash | 89 230 |
/// | the same code, from SRAM | **60 892** |
///
/// The ratio was 2.1x when this comment was first written and is 1.5x now.
/// That is not the placement mattering less -- it is the code around it having
/// shrunk, so less of it misses the cache. Re-measure before quoting either
/// number; `bench_forward` prints both.
///
/// `.data` is copied to SRAM by the startup code, so a function placed there
/// executes from SRAM; `inline(never)` keeps it a real function so the
/// placement means something, and `IteratedEkf::update` and
/// `forward_and_jac_vector` are `inline(always)` so they come along rather
/// than staying behind in flash.
///
/// **Anything added to this function's call graph must be inlinable into it**,
/// or it silently moves back to flash and takes the 2x with it.
#[unsafe(link_section = ".data")]
#[inline(never)]
fn filter_step(
    ekf: &mut IteratedEkf<POSE_DIM, MEAS_DIM>,
    model: &PoseModel<'_>,
    dt: f32,
    z: &[f32; MEAS_DIM],
) -> Result<(), iekf::UpdateError> {
    if dt > 0.0 {
        ekf.predict(dt.min(MAX_PREDICT_DT));
    }
    ekf.update(model, z)
}

/// Longest gap the process noise is allowed to integrate over.
///
/// After a stall -- a bus fault, a USB stall, a long calibration -- the
/// elapsed time can be enormous, and letting `P` grow by `Q * dt` over a full
/// second would put the covariance far outside anything the linearisation
/// supports. Clamping is a lie, but it is a small and bounded one, and the
/// alternative is a filter that has to be reset after every hiccup.
const MAX_PREDICT_DT: f32 = 0.05;

/// How long the red abort flash stays up before the ring goes back to green.
const ABORT_FLASH: Duration = Duration::from_millis(1500);

/// Runs on core 1. Never returns.
#[embassy_executor::task]
pub async fn task() -> ! {
    // SAFETY: this task is the only thing that ever touches TABLE_RAM, and it
    // is spawned exactly once.
    let table = FieldTable::copy_into(unsafe { &mut *addr_of_mut!(TABLE_RAM) });
    let model = PoseModel::new(&table);
    info!("core 1: field table in RAM, estimator starting");

    let mut calibration = Calibration::fallback();
    let mut deadzone = Deadzone::new(calibration.deadzone);

    let mut ekf = IteratedEkf::<POSE_DIM, MEAS_DIM>::new(
        [0.0; POSE_DIM],
        tuning::initial_variance(),
        tuning::measurement_variance(&calibration.sigma),
    );
    ekf.set_process_noise(tuning::process_noise());
    ekf.set_iterations(tuning::ITERATIONS);

    // A calibration runs at boot, unprompted. Nothing is persisted across a
    // power cycle -- deliberately, since writing flash while this core runs
    // from XIP is a hazard not worth taking on for a measurement that costs a
    // second -- so this is what makes the device usable when it is plugged in.
    // The user is expected not to be holding the knob at that moment, and the
    // stillness check is what catches it when they are.
    let mut running = Some(RestCalibration::new());
    led::set(Pattern::Spinner(led::BLUE));

    let mut last_t_us: Option<u32> = None;
    let mut flags: u8 = 0;
    let mut revert_led_at: Option<Instant> = None;

    loop {
        let sample = SAMPLE.wait().await;

        // Put the ring back after an abort flash has had its time.
        if let Some(at) = revert_led_at
            && Instant::now() >= at
        {
            led::set(Pattern::Solid(led::GREEN));
            revert_led_at = None;
            flags &= !status::CALIBRATION_ABORTED;
        }

        // A request that arrives mid-calibration restarts it, which is what
        // the user pressing the buttons again almost certainly means.
        if buttons::take_calibration_request() {
            info!("core 1: rest calibration requested");
            running = Some(RestCalibration::new());
            led::set(Pattern::Spinner(led::BLUE));
            revert_led_at = None;
        }

        // ── predict and update ──
        let dt = match last_t_us {
            Some(previous) => sample.t_us.wrapping_sub(previous) as f32 * 1e-6,
            None => 0.0,
        };
        last_t_us = Some(sample.t_us);

        let mut z = [0.0f32; MEAS_DIM];
        for (i, &c) in sample.counts.iter().enumerate() {
            z[i] = c as f32;
        }

        match filter_step(&mut ekf, &model, dt, &z) {
            Ok(()) => flags = (flags | status::FILTER_VALID) & !status::DIVERGED,
            Err(e) => {
                // The innovation covariance stopped being positive definite,
                // which means the filter had already lost the pose rather than
                // that it is losing it now. Restart from the last known rest,
                // wide enough to find its way back.
                warn!("core 1: filter update rejected ({}), resetting", e);
                ekf.reset(calibration.rest_pose, wide_initial_variance());
                flags = (flags & !status::FILTER_VALID) | status::DIVERGED;
                continue;
            }
        }

        STEPS.fetch_add(1, Ordering::Relaxed);
        let estimate = *ekf.state();

        // ── calibration, if one is running ──
        let mut progress = 0u8;
        if let Some(cal) = running.as_mut() {
            flags |= status::CALIBRATING;
            progress = cal.progress();
            match cal.feed(&sample.counts, &estimate) {
                Step::Continue => {}
                Step::Finished(result) => {
                    info!(
                        "core 1: calibrated. rest {} um, deadzone {} um, sigma {} counts",
                        (result.rest_pose[0] * 1000.0) as i32,
                        (result.deadzone[0] * 1000.0) as i32,
                        result.sigma[0]
                    );
                    calibration = result;
                    deadzone.set_thresholds(calibration.deadzone);
                    ekf.set_measurement_variance(tuning::measurement_variance(&calibration.sigma));
                    running = None;
                    flags = (flags & !status::CALIBRATING & !status::CALIBRATION_ABORTED)
                        | status::CALIBRATED;
                    progress = 255;
                    led::set(Pattern::Solid(led::GREEN));
                }
                Step::Aborted(Abort::KnobMoved) => {
                    warn!("core 1: calibration abandoned, the knob moved");
                    running = None;
                    flags = (flags & !status::CALIBRATING) | status::CALIBRATION_ABORTED;
                    led::set(Pattern::Blink {
                        color: led::RED,
                        period_ms: 250,
                    });
                    revert_led_at = Some(Instant::now() + ABORT_FLASH);
                }
            }
        }

        // ── report ──
        let mut zeroed = [0.0f32; POSE_DIM];
        for i in 0..POSE_DIM {
            zeroed[i] = estimate[i] - calibration.rest_pose[i];
        }
        let (pose, at_rest) = deadzone.apply(&zeroed);
        if at_rest {
            flags |= status::IN_DEADZONE;
        } else {
            flags &= !status::IN_DEADZONE;
        }

        // `nis()` is a 9x9 triangular solve, and this lock is taken by core 0
        // on every readout for `submit` and `latest`. Computing it inside the
        // lock put core 1's arithmetic directly into core 0's sampling loop --
        // the two rates were measurably coupled. It is only ever read by the
        // debug frame, so it does not belong in a cross-core critical section.
        let nis = ekf.nis();
        ESTIMATE.lock(|cell| {
            cell.set(Some(Estimate {
                seq: sample.seq,
                pose,
                nis,
                status: flags,
                progress,
            }))
        });
    }
}

/// Covariance to restart from after a divergence.
///
/// Much wider than the boot value: at boot the pose is known to be rest, and
/// after a divergence it is known to be anything at all. A millimetre and two
/// degrees covers the mechanism's whole envelope.
fn wide_initial_variance() -> [f32; POSE_DIM] {
    let pos = 1.0 * 1.0;
    let rot = 0.035 * 0.035;
    [pos, pos, pos, rot, rot, rot]
}
