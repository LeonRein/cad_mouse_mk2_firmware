//! Shared estimator code for the CAD Mouse MK2.
//!
//! A library rather than modules inside `main.rs` so the firmware and the
//! benchmark measure the same code. A benchmark that times a copy of the hot
//! path is worth very little the moment the copy drifts.
//!
//! Ported from `scripts/cadmouse/`, which is where the model is developed and
//! where the calibration is fitted. `generated.rs` and `gen/field_table.bin`
//! are emitted by `cadmouse.export` from the same objects the host uses.

#![no_std]

pub mod generated;
pub mod magnet;
pub mod model;
