//! Pins the *current* numerical behaviour of the model and the filter.
//!
//! Not a correctness check -- `golden.rs` is that, when its vectors are in
//! step with `generated.rs`. This is an equivalence net for performance work:
//! it says "whatever this computed before the change, it computes now".
//!
//! Regenerate deliberately, and only when the numbers are *meant* to move:
//!     SNAPSHOT_WRITE=1 cargo test --target x86_64-unknown-linux-gnu \
//!         -p cadmouse-model --test snapshot -- --nocapture

use cadmouse_model::magnet::FieldTable;
use cadmouse_model::model::{
    forward, forward_and_jac_vector, MEAS_DIM, POSE_DIM, Pose, PoseModel,
};
use cadmouse_model::tuning;
use iekf::IteratedEkf;
use std::fmt::Write as _;

const N_POSES: usize = 64;

/// The same envelope and the same LCG `bench_forward` uses, so the snapshot
/// covers the poses the device actually reaches.
fn poses() -> Vec<Pose> {
    let mut out = Vec::with_capacity(N_POSES);
    let mut state: u32 = 0x1234_5678;
    for _ in 0..N_POSES {
        let mut pose = [0.0f32; POSE_DIM];
        for (k, value) in pose.iter_mut().enumerate() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / 16_777_216.0 - 0.5;
            *value = if k < 3 { 2.4 * unit } else { 0.14 * unit };
        }
        out.push(pose);
    }
    out
}

fn render() -> String {
    let table = FieldTable::from_flash();
    let mut s = String::new();

    for pose in poses() {
        let counts = forward(&pose, &table);
        for c in counts {
            writeln!(s, "f {c:.6e}").unwrap();
        }
        let (counts, jac) = forward_and_jac_vector(&pose, &table);
        for c in counts {
            writeln!(s, "c {c:.6e}").unwrap();
        }
        for row in jac {
            for v in row {
                writeln!(s, "j {v:.6e}").unwrap();
            }
        }
    }

    // A full filter run. The golden FRAMES are stale as *targets* but remain a
    // perfectly good measurement sequence to drive the filter with.
    #[allow(dead_code)]
    mod golden {
        include!("data/golden_data.rs");
    }
    let model = PoseModel::new(&table);
    let mut ekf = IteratedEkf::<POSE_DIM, MEAS_DIM>::new(
        golden::INITIAL_POSE,
        tuning::initial_variance(),
        tuning::measurement_variance(&golden::SIGMA),
    );
    ekf.set_process_noise(tuning::process_noise());
    ekf.set_iterations(tuning::ITERATIONS);

    for k in 0..golden::N_FRAMES {
        let mut z = [0.0f32; MEAS_DIM];
        for c in 0..MEAS_DIM {
            z[c] = golden::FRAMES[k][c] as f32;
        }
        if golden::DTS[k] > 0.0 {
            ekf.predict(golden::DTS[k]);
        }
        ekf.update(&model, &z).expect("filter stayed positive definite");
        for v in ekf.state() {
            writeln!(s, "x {v:.6e}").unwrap();
        }
        writeln!(s, "n {:.6e}", ekf.nis()).unwrap();
        for row in ekf.covariance() {
            for v in row {
                writeln!(s, "p {v:.6e}").unwrap();
            }
        }
    }
    s
}

const PATH: &str = "tests/data/snapshot.txt";

#[test]
fn behaviour_is_unchanged() {
    let got = render();
    if std::env::var("SNAPSHOT_WRITE").is_ok() {
        std::fs::write(PATH, &got).unwrap();
        println!("wrote {} lines to {PATH}", got.lines().count());
        return;
    }
    let want = std::fs::read_to_string(PATH)
        .expect("no snapshot; run once with SNAPSHOT_WRITE=1");

    // Mixed absolute/relative, per quantity: "relative" is meaningless next to
    // zero and these span twelve orders of magnitude. The absolute floor is in
    // each case a scale below which a difference cannot matter -- a thousandth
    // of a count against a 1.08-count noise floor, a nanometre against a
    // posterior that is micrometres wide.
    //
    // Reassociating a sum changes the last bits; that is expected and is what
    // these budgets are for. What they must catch is a *wrong* answer.
    fn tolerance(tag: u8) -> (f64, f64) {
        match tag {
            b'f' | b'c' => (1e-5, 1e-3),  // counts
            // The table's derivative terms carry a cancellation in the cubic
            // derivative weights, so an f32 bicubic is only good to about
            // 1e-3 relative against an f64 evaluation of the same stencil --
            // measured, over 4000 random points, for both the per-point and
            // the separable summation order. A budget tighter than that is
            // measuring the summation order, not the answer.
            b'j' => (2e-3, 1e-3),         // counts per mm / per rad
            b'x' => (1e-4, 1e-6),         // pose: mm and rad
            b'n' => (1e-3, 1e-4),         // NIS, dimensionless
            // Loosest, deliberately. The covariance is a second-order
            // quantity -- it steers the gain, it is not reported to anyone --
            // and it integrates every rounding change in the Jacobian over the
            // whole run. The pose budget above is the one that constrains what
            // the device actually outputs.
            b'p' => (1e-2, 1e-15),        // covariance: mm^2 and rad^2
            _ => panic!("unknown tag {tag}"),
        }
    }

    assert_eq!(
        got.lines().count(),
        want.lines().count(),
        "snapshot shape changed"
    );

    // Worst offender per quantity, each scored against its own budget, so one
    // loose tolerance cannot hide a tight one that is about to break.
    let mut worst: std::collections::BTreeMap<u8, (usize, String, String, f64)> =
        Default::default();
    let mut n_diff = 0usize;
    for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
        if g == w {
            continue;
        }
        n_diff += 1;
        let tag = g.as_bytes()[0];
        assert_eq!(tag, w.as_bytes()[0], "snapshot shape changed at line {i}");
        let (rel_tol, abs_tol) = tolerance(tag);
        let gv: f64 = g[2..].parse().unwrap();
        let wv: f64 = w[2..].parse().unwrap();
        let score = (gv - wv).abs() / (wv.abs() * rel_tol + abs_tol);
        let slot = worst.entry(tag).or_insert((i, g.into(), w.into(), 0.0));
        if score > slot.3 {
            *slot = (i, g.into(), w.into(), score);
        }
    }

    if n_diff == 0 {
        println!("snapshot bit-identical ({} values)", got.lines().count());
        return;
    }

    println!(
        "{n_diff}/{} values differ. Worst per quantity, as a fraction of its budget:",
        got.lines().count()
    );
    let mut broke = None;
    for (tag, (line, g, w, score)) in &worst {
        println!(
            "  {}  {score:8.4} of budget   line {line}: {g} vs {w}",
            *tag as char
        );
        if *score > 1.0 {
            broke = Some((*tag as char, *line, g.clone(), w.clone(), *score));
        }
    }
    if let Some((tag, line, g, w, score)) = broke {
        panic!("'{tag}' moved to {score:.2}x its budget at line {line}: {g} vs {w}");
    }
}
