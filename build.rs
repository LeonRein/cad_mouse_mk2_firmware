//! This build script copies the `memory.x` file from the crate root into
//! a directory where the linker can always find it at build time.
//! For many projects this is optional, as the linker always searches the
//! project root directory -- wherever `Cargo.toml` is. However, if you
//! are using a workspace or have a more complicated build setup, this
//! build script becomes required. Additionally, by requesting that
//! Cargo re-run the build script whenever `memory.x` is changed,
//! updating `memory.x` ensures a rebuild of the application with the
//! new memory settings.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    // Put `memory.x` in our output directory and ensure it's
    // on the linker search path.
    let out = &PathBuf::from(env::var_os("OUT_DIR").unwrap());
    File::create(out.join("memory.x"))
        .unwrap()
        .write_all(include_bytes!("memory.x"))
        .unwrap();
    println!("cargo:rustc-link-search={}", out.display());

    // By default, Cargo will re-run a build script whenever
    // any file in the project changes. By specifying `memory.x`
    // here, we ensure the build script is only re-run when
    // `memory.x` is changed.
    println!("cargo:rerun-if-changed=memory.x");

    println!("cargo:rustc-link-arg-bins=--nmagic");
    println!("cargo:rustc-link-arg-bins=-Tlink.x");
    println!("cargo:rustc-link-arg-bins=-Tdefmt.x");

    warn_if_not_release();
}

/// Say so, loudly, when this is not a release build.
///
/// A flashed binary carries no sign of which profile produced it, and the two
/// behave differently enough to send someone chasing a hardware fault: `dev`
/// keeps `debug-assertions` and `overflow-checks`, and on this board that runs
/// the sensor loop at 1339 Hz against release's 2264 Hz. That looked like a
/// regression in the firmware for some time before it turned out to be a
/// profile.
///
/// `PROFILE` is `debug` for `dev` and anything inheriting it, `release`
/// otherwise. `rerun-if-env-changed` keeps the warning honest across a switch
/// between the two without forcing a rebuild of the linker script work above.
fn warn_if_not_release() {
    println!("cargo:rerun-if-env-changed=PROFILE");
    if env::var("PROFILE").as_deref() == Ok("release") {
        return;
    }
    println!(
        "cargo:warning=building WITHOUT --release: overflow checks and debug \
         assertions are on, and the sensor loop drops to ~1339 Hz from ~2264 Hz."
    );
    println!(
        "cargo:warning=Fine for iterating. Flash `cargo run --release` before \
         measuring anything, recording a session, or using the device for real."
    );
}
