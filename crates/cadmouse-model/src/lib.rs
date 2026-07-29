//! The knob's measurement model: where the magnets are, what the sensors see.
//!
//! A crate rather than modules inside the firmware so that the firmware, the
//! benchmark and the host-side golden-vector tests all measure the same code.
//! A benchmark that times a copy of the hot path is worth very little the
//! moment the copy drifts.
//!
//! Ported from `scripts/cadmouse/`, which is where the model is developed and
//! where the calibration is fitted. [`generated`] and `gen/field_table.bin` are
//! emitted by `cadmouse.export` from the same objects the host uses, so the
//! firmware cannot drift away from what the calibration was fitted against.
//!
//! Builds for the host as well as the target, which is what lets
//! `cargo test --target x86_64-unknown-linux-gnu -p cadmouse-model` check the
//! port against vectors produced by the Python.

#![cfg_attr(not(test), no_std)]

pub mod generated;
pub mod magnet;
pub mod model;
pub mod rest;
pub mod tuning;

pub use magnet::FieldTable;
pub use model::{MEAS_DIM, POSE_DIM, Pose, PoseModel, forward, forward_and_jac_vector};
pub use rest::{Calibration, Deadzone, RestCalibration};
