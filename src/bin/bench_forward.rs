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

use cad_mouse_mk2_firmware::magnet::FieldTable;
use cad_mouse_mk2_firmware::model::{forward, forward_and_jac, Pose, POSE_DIM};
use cad_mouse_mk2_firmware::generated as consts;
use cortex_m::peripheral::DWT;
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
