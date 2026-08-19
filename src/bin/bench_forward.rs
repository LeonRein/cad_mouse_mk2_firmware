//! Cycle-counts the measurement function on the real target.
//!
//!     cargo run --bin bench_forward

//!
//! The question this answers is whether a twelve-state filter fits in the 2 kHz
//! budget. At 150 MHz that budget is 75 000 cycles per step; a six-state UKF
//! needs thirteen `h()` evaluations and a twelve-state one needs twenty-five,
//! while an iterated EKF needs roughly two `forward_and_jac` calls. Estimating
//! ~3500 cycles for `h()` put the twelve-state UKF over budget and the rest
//! under it, which is too close to a boundary to leave as an estimate.
//!
//! Three things are measured beyond the headline number, because each could
//! move it by more than the margin:
//!
//! * **Flash against RAM.** The 85 kB table is read through the XIP cache.
//!   Whether that matters is a measurement, not a guess, and with 520 kB of
//!   SRAM copying it in is free if it helps.
//! * **Optimisation level.** Both profiles build at `opt-level = 3`; `z` was
//!   measured first and cost roughly 2-3x on this code.
//! * **`forward` against `forward_and_jac`.** The gradients come out of the
//!   same bicubic gather, so the iterated EKF's Jacobian should be nearly free.
//!   If it is not, the filter choice changes.

#![no_std]
#![no_main]

use cadmouse_model::generated as consts;
use cadmouse_model::magnet::FieldTable;
use cadmouse_model::model::{MEAS_DIM, POSE_DIM, Pose, PoseModel, forward, forward_and_jac};
use cadmouse_model::tuning;
use cortex_m::peripheral::DWT;
use iekf::IteratedEkf;
use defmt::info;
use embassy_executor::Spawner;
use {defmt_rtt as _, panic_probe as _};

/// Nominal system clock after `embassy_rp::init`, for turning cycles into time.
const CLOCK_HZ: u32 = 150_000_000;

/// Enough poses that the table lookups land in different cells, so the XIP
/// cache is exercised the way real motion would exercise it rather than
/// answering from one hot line.
const N_POSES: usize = 64;

/// Repeats per measurement, to swamp the timer overhead.
const N_REPS: usize = 32;

static mut TABLE_RAM: [u8; consts::TABLE_BYTES] = [0; consts::TABLE_BYTES];

/// Poses spanning the envelope the mechanism actually reaches: about +-1.2 mm
/// and +-4 degrees. Generated with a small LCG so the sequence is reproducible
/// across runs and profiles.
fn make_poses() -> [Pose; N_POSES] {
    let mut poses = [[0.0f32; POSE_DIM]; N_POSES];
    let mut state: u32 = 0x1234_5678;
    for pose in poses.iter_mut() {
        for (k, value) in pose.iter_mut().enumerate() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0 - 0.5; // [-0.5, 0.5)
            *value = if k < 3 { 2.4 * unit } else { 0.14 * unit };
        }
    }
    poses
}

/// Set (or clear) the FPU's flush-to-zero bit, FPSCR[24].
///
/// With it clear -- the reset default -- an operation with a subnormal operand
/// or result is handled by *support code*, not by the FPU, and costs orders of
/// magnitude more than a normal one. A Kalman filter's covariance is exactly
/// the place that happens: it collapses toward zero as the estimate settles,
/// and the products of two already-tiny entries fall off the bottom of the
/// normal range.
fn set_flush_to_zero(enable: bool) {
    unsafe {
        let mut fpscr: u32;
        core::arch::asm!("vmrs {}, fpscr", out(reg) fpscr);
        fpscr = if enable { fpscr | (1 << 24) } else { fpscr & !(1 << 24) };
        core::arch::asm!("vmsr fpscr, {}", in(reg) fpscr);
    }
}

/// One filter step, placed in RAM rather than in flash.
///
/// `.data` is copied from flash to SRAM by the startup code, so putting a
/// function there makes it execute from SRAM. `inline(never)` keeps it a real
/// function so the placement means something, and the callees it pulls in are
/// marked `inline(always)` so they come along rather than staying behind in
/// flash.
#[unsafe(link_section = ".data")]
#[inline(never)]
fn ram_filter_step(
    ekf: &mut IteratedEkf<POSE_DIM, MEAS_DIM>,
    model: &PoseModel<'_>,
    z: &[f32; MEAS_DIM],
) {
    ekf.predict(0.0005);
    let _ = ekf.update(model, z);
}

/// A measurement model that does no work, so a filter step using it costs only
/// the filter's own linear algebra.
///
/// Hoisted to module scope so the same type can be driven from flash and from
/// RAM. `inline(never)` is what makes it a fair subtraction: the point is to
/// keep the call, and only remove what is behind it.
struct TrivialModel {
    counts: [f32; MEAS_DIM],
    jac: [[f32; POSE_DIM]; MEAS_DIM],
}

impl iekf::MeasurementModel<POSE_DIM, MEAS_DIM> for TrivialModel {
    #[inline(never)]
    fn predict_and_jacobian(
        &self,
        _state: &[f32; POSE_DIM],
    ) -> ([f32; MEAS_DIM], [[f32; POSE_DIM]; MEAS_DIM]) {
        (self.counts, self.jac)
    }
}

/// The filter's own algebra with the code in RAM, for comparison against the
/// identical loop running from flash.
///
/// This exists to settle whether "filter only, no model" is measuring the
/// algebra or measuring the XIP cache. If the two differ by roughly the same
/// factor as the full step does, the flash figure was never a cost of the
/// arithmetic.
#[unsafe(link_section = ".data")]
#[inline(never)]
fn ram_trivial_step(
    ekf: &mut IteratedEkf<POSE_DIM, MEAS_DIM>,
    model: &TrivialModel,
    z: &[f32; MEAS_DIM],
) {
    ekf.predict(0.0005);
    let _ = ekf.update(model, z);
}

fn cycles<F: FnMut()>(mut body: F) -> u32 {
    let start = DWT::cycle_count();
    body();
    DWT::cycle_count().wrapping_sub(start)
}

fn report(label: &str, total: u32, calls: usize) {
    let per = total / calls as u32;
    let ns = (per as u64 * 1_000_000_000) / CLOCK_HZ as u64;
    info!(
        "{=str} : {=u32} cycles, {=u64} ns  -> {=u32} fit in the 75000-cycle 2 kHz budget",
        label,
        per,
        ns,
        75_000 / per.max(1)
    );
}

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let _p = embassy_rp::init(Default::default());
    let mut core = cortex_m::Peripherals::take().unwrap();
    core.DCB.enable_trace();
    core.DWT.enable_cycle_counter();

    let poses = make_poses();
    let flash_table = FieldTable::from_flash();
    // SAFETY: single-threaded, and this is the only reference taken.
    let ram_table = FieldTable::copy_into(unsafe { &mut *core::ptr::addr_of_mut!(TABLE_RAM) });

    info!("--- measurement function, {=usize} poses x {=usize} reps ---", N_POSES, N_REPS);

    let calls = N_POSES * N_REPS;

    let total = cycles(|| {
        for _ in 0..N_REPS {
            for pose in poses.iter() {
                core::hint::black_box(forward(core::hint::black_box(pose), &flash_table));
            }
        }
    });
    report("forward, table in flash", total, calls);

    let total = cycles(|| {
        for _ in 0..N_REPS {
            for pose in poses.iter() {
                core::hint::black_box(forward(core::hint::black_box(pose), &ram_table));
            }
        }
    });
    report("forward, table in RAM", total, calls);

    let total = cycles(|| {
        for _ in 0..N_REPS {
            for pose in poses.iter() {
                core::hint::black_box(forward_and_jac(core::hint::black_box(pose), &flash_table));
            }
        }
    });
    report("forward_and_jac, flash", total, calls);

    let total = cycles(|| {
        for _ in 0..N_REPS {
            for pose in poses.iter() {
                core::hint::black_box(forward_and_jac(core::hint::black_box(pose), &ram_table));
            }
        }
    });
    report("forward_and_jac, RAM", total, calls);

    // One bicubic lookup on its own, to show how much of the total is the
    // table and how much is the geometry around it.
    let total = cycles(|| {
        for _ in 0..N_REPS {
            for pose in poses.iter() {
                let rho = 8.0 + 4.0 * pose[0];
                let z = -9.0 + pose[2];
                core::hint::black_box(ram_table.sample(core::hint::black_box(rho), z));
            }
        }
    });
    report("one bicubic sample, RAM", total, calls);

    // The filter does not call `forward_and_jac` directly: it goes through
    // `forward_and_jac_vector`, which converts the Jacobian from the local
    // convention to the vector one. That conversion should be a 9x6 times 3x3.
    let total = cycles(|| {
        for _ in 0..N_REPS {
            for pose in poses.iter() {
                core::hint::black_box(cadmouse_model::model::forward_and_jac_vector(
                    core::hint::black_box(pose),
                    &ram_table,
                ));
            }
        }
    });
    report("forward_and_jac_vector, RAM", total, calls);

    let total = cycles(|| {
        for _ in 0..N_REPS {
            for pose in poses.iter() {
                core::hint::black_box(cadmouse_model::model::right_jacobian_so3(
                    core::hint::black_box(&[pose[3], pose[4], pose[5]]),
                ));
            }
        }
    });
    report("right_jacobian_so3 alone", total, calls);

    // The same evaluation, but reached the way the filter reaches it: through
    // the `MeasurementModel` trait. If this matches the line above, the trait
    // is free and any difference inside `update` is about code layout rather
    // than about the call.
    {
        use iekf::MeasurementModel;
        let model = PoseModel::new(&ram_table);
        let total = cycles(|| {
            for _ in 0..N_REPS {
                for pose in poses.iter() {
                    core::hint::black_box(
                        model.predict_and_jacobian(core::hint::black_box(pose)),
                    );
                }
            }
        });
        report("...via the MeasurementModel trait", total, calls);
    }

    // The number that actually decides whether the firmware makes its deadline:
    // one whole filter step, the measurement model plus all the linear algebra
    // around it, at the shipping iteration count. Everything above is a
    // component of this.
    {
        let model = PoseModel::new(&ram_table);
        let sigma = [tuning::FALLBACK_SIGMA_COUNTS; MEAS_DIM];
        let mut ekf = IteratedEkf::<POSE_DIM, MEAS_DIM>::new(
            [0.0; POSE_DIM],
            tuning::initial_variance(),
            tuning::measurement_variance(&sigma),
        );
        ekf.set_process_noise(tuning::process_noise());
        ekf.set_iterations(tuning::ITERATIONS);

        // Measurements taken from the model itself, so the filter is tracking
        // a real trajectory through the table rather than diverging on noise.
        let mut measurements = [[0.0f32; MEAS_DIM]; N_POSES];
        for (k, pose) in poses.iter().enumerate() {
            measurements[k] = forward(pose, &ram_table);
        }

        let total = cycles(|| {
            for _ in 0..N_REPS {
                for z in measurements.iter() {
                    ekf.predict(0.0005);
                    let _ = ekf.update(&model, core::hint::black_box(z));
                }
            }
        });
        report("full IEKF step (2 iterations), RAM", total, calls);

        ekf.set_iterations(1);
        let total = cycles(|| {
            for _ in 0..N_REPS {
                for z in measurements.iter() {
                    ekf.predict(0.0005);
                    let _ = ekf.update(&model, core::hint::black_box(z));
                }
            }
        });
        report("full IEKF step (1 iteration), RAM", total, calls);
        ekf.set_iterations(tuning::ITERATIONS);

        // Same work, but with the covariance reset to its initial value before
        // every step. If this is much faster than the run above, the cost is
        // *data-dependent* -- which on this FPU means subnormal operands in a
        // covariance that has collapsed, not the arithmetic itself.
        let total = cycles(|| {
            for _ in 0..N_REPS {
                for z in measurements.iter() {
                    ekf.reset([0.0; POSE_DIM], tuning::initial_variance());
                    let _ = ekf.update(&model, core::hint::black_box(z));
                }
            }
        });
        report("full IEKF step, covariance reset each time", total, calls);

        // And the same again with the FPU flushing subnormals to zero, which
        // is the fix if that is what this is.
        set_flush_to_zero(true);
        let total = cycles(|| {
            for _ in 0..N_REPS {
                for z in measurements.iter() {
                    ekf.predict(0.0005);
                    let _ = ekf.update(&model, core::hint::black_box(z));
                }
            }
        });
        report("full IEKF step (2 iterations), flush-to-zero", total, calls);
        set_flush_to_zero(false);
    }

    // The filter with the measurement model taken out of it: same matrices,
    // same code path, but `predict_and_jacobian` returns a constant. Whatever
    // this costs is the filter's own overhead, and the difference from the run
    // above is what the model actually costs *when called from here* rather
    // than from a tight loop.
    {
        let (counts, jac) = cadmouse_model::model::forward_and_jac_vector(&poses[0], &ram_table);
        let trivial = TrivialModel { counts, jac };

        let sigma = [tuning::FALLBACK_SIGMA_COUNTS; MEAS_DIM];
        let mut ekf = IteratedEkf::<POSE_DIM, MEAS_DIM>::new(
            [0.0; POSE_DIM],
            tuning::initial_variance(),
            tuning::measurement_variance(&sigma),
        );
        ekf.set_process_noise(tuning::process_noise());

        for (label, iterations) in [("1 iteration", 1u8), ("2 iterations", 2)] {
            ekf.set_iterations(iterations);
            let total = cycles(|| {
                for _ in 0..N_REPS {
                    for _ in 0..N_POSES {
                        ekf.predict(0.0005);
                        let _ = ekf.update(&trivial, core::hint::black_box(&counts));
                    }
                }
            });
            match label {
                "1 iteration" => report("filter only, no model (1 iteration)", total, calls),
                _ => report("filter only, no model (2 iterations)", total, calls),
            }
        }

        // The identical loop with the code in SRAM. Same matrices, same call,
        // same trivial model -- the only variable is where the instructions
        // live, so the ratio between this and the run above is a pure
        // measurement of the XIP cache rather than of the linear algebra.
        ekf.set_iterations(1);
        let total = cycles(|| {
            for _ in 0..N_REPS {
                for _ in 0..N_POSES {
                    ram_trivial_step(&mut ekf, &trivial, core::hint::black_box(&counts));
                }
            }
        });
        report("filter only, no model (1 iteration), CODE IN RAM", total, calls);
    }

    // The same step, but with the code itself in SRAM instead of executing
    // from external flash through the XIP cache. Everything above says the
    // model and the filter are each fast alone and slow together, which is the
    // signature of a working set that no longer fits that cache.
    {
        let model = PoseModel::new(&ram_table);
        let sigma = [tuning::FALLBACK_SIGMA_COUNTS; MEAS_DIM];
        let mut ekf = IteratedEkf::<POSE_DIM, MEAS_DIM>::new(
            [0.0; POSE_DIM],
            tuning::initial_variance(),
            tuning::measurement_variance(&sigma),
        );
        ekf.set_process_noise(tuning::process_noise());
        ekf.set_iterations(tuning::ITERATIONS);

        let mut measurements = [[0.0f32; MEAS_DIM]; N_POSES];
        for (k, pose) in poses.iter().enumerate() {
            measurements[k] = forward(pose, &ram_table);
        }

        let total = cycles(|| {
            for _ in 0..N_REPS {
                for z in measurements.iter() {
                    ram_filter_step(&mut ekf, &model, core::hint::black_box(z));
                }
            }
        });
        report("full IEKF step (2 iterations), CODE IN RAM", total, calls);

        // The same step with the FPU flushing subnormals to zero. This is the
        // configuration the firmware actually ships, so it is the one number
        // that answers "does it make the sample interval". Measuring
        // flush-to-zero only against the flash-resident build, as this
        // benchmark used to, answers a question nobody has.
        set_flush_to_zero(true);
        let total = cycles(|| {
            for _ in 0..N_REPS {
                for z in measurements.iter() {
                    ram_filter_step(&mut ekf, &model, core::hint::black_box(z));
                }
            }
        });
        report("full IEKF step, CODE IN RAM, flush-to-zero", total, calls);
        set_flush_to_zero(false);
    }

    // The linear algebra on its own, since the step above is far more than the
    // measurement model accounts for. Each of these is one of the pieces of
    // `IteratedEkf::update`, at the sizes it actually runs at.
    {
        use iekf::linalg;

        let jac = {
            let (_, j) = forward_and_jac(&poses[0], &ram_table);
            j
        };
        let p = {
            let mut p = [[0.0f32; POSE_DIM]; POSE_DIM];
            for (i, row) in p.iter_mut().enumerate() {
                row[i] = 1e-3;
            }
            p
        };

        let total = cycles(|| {
            for _ in 0..N_REPS {
                for _ in 0..N_POSES {
                    core::hint::black_box(linalg::matmul::<MEAS_DIM, POSE_DIM, POSE_DIM>(
                        core::hint::black_box(&jac),
                        &p,
                    ));
                }
            }
        });
        report("  J P            (9x6 * 6x6)", total, calls);

        let jp = linalg::matmul::<MEAS_DIM, POSE_DIM, POSE_DIM>(&jac, &p);
        let total = cycles(|| {
            for _ in 0..N_REPS {
                for _ in 0..N_POSES {
                    core::hint::black_box(linalg::matmul_transpose::<
                        MEAS_DIM,
                        POSE_DIM,
                        MEAS_DIM,
                    >(core::hint::black_box(&jp), &jac));
                }
            }
        });
        report("  J P J^T        (9x6 * 6x9)", total, calls);

        let mut s = linalg::matmul_transpose::<MEAS_DIM, POSE_DIM, MEAS_DIM>(&jp, &jac);
        linalg::add_diagonal(&mut s, &[1.2; MEAS_DIM]);
        let total = cycles(|| {
            for _ in 0..N_REPS {
                for _ in 0..N_POSES {
                    core::hint::black_box(linalg::cholesky(core::hint::black_box(&s)));
                }
            }
        });
        report("  cholesky       (9x9)", total, calls);

        let chol = linalg::cholesky(&s).unwrap();
        let total = cycles(|| {
            for _ in 0..N_REPS {
                for _ in 0..N_POSES {
                    core::hint::black_box(linalg::cholesky_solve_mat_transposed::<MEAS_DIM, POSE_DIM>(
                        core::hint::black_box(&chol),
                        &jp,
                    ));
                }
            }
        });
        report("  solve+transpose(9x9 \\ 9x6)", total, calls);
    }

    // A correctness check, so a benchmark that optimised the work away or read
    // a misaligned table cannot masquerade as a fast one. At the neutral pose
    // the prediction must match what the host reports for the same
    // calibration, to a few counts.
    let neutral = forward(&[0.0; POSE_DIM], &flash_table);
    info!("forward(0) = {}", neutral);
    info!("host says [8.1, 24.3, 510.7, 6.0, -27.0, 496.7, 55.6, -7.5, -409.5]");

    let ram_neutral = forward(&[0.0; POSE_DIM], &ram_table);
    let mut worst = 0.0f32;
    for k in 0..neutral.len() {
        let d = (neutral[k] - ram_neutral[k]).abs();
        if d > worst {
            worst = d;
        }
    }
    info!("flash vs RAM table agreement: {=f32} counts (must be 0)", worst);

    info!("--- done ---");
    loop {
        embassy_time::Timer::after_secs(60).await;
    }
}
