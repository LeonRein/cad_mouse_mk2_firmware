#!/usr/bin/env bash
# Build the firmware and package it as a UF2, for flashing without a probe.
#
# The RP2350's bootrom exposes a mass-storage device when it comes up in
# BOOTSEL, and copying a UF2 onto it programs the flash. That is the whole
# update path once the debug probe is gone -- no probe-rs, no SWD, nothing
# attached to the board but USB.
#
#     scripts/mkuf2.sh                 # -> target/cad-mouse-mk2-firmware.uf2
#
# To install it: hold BOOTSEL while plugging the board in, then copy the file
# onto the "RP2350" drive that appears. The board reboots into the new
# firmware on its own once the copy completes.
#
# Two details that will otherwise waste an afternoon:
#
#   * `picotool` insists on recognising the input by *file extension*, and
#     cargo's output has none. Hence the copy to `.elf` below rather than
#     pointing picotool straight at the binary.
#
#   * The family must be `rp2350-arm-s` -- ARM, secure. The RP2350 also has a
#     RISC-V core and a non-secure ARM image type, and a UF2 built for the
#     wrong one is copied without complaint and simply does not boot.
set -euo pipefail

cd "$(dirname "$0")/.."

TARGET=thumbv8m.main-none-eabihf
BIN=cad-mouse-mk2-firmware
ELF="target/$TARGET/release/$BIN"
OUT="target/$BIN.uf2"

if ! command -v picotool >/dev/null; then
    echo "picotool not found -- see https://github.com/raspberrypi/picotool" >&2
    exit 1
fi

echo "building $BIN (release)"
cargo build --release --bin "$BIN"

# Release, always: `dev` carries overflow checks and costs the sensor loop a
# third of its rate. `build.rs` warns about it, but a UF2 has no build log
# attached to it by the time anyone flashes it, so refuse rather than warn.
cp "$ELF" "target/$BIN.elf"
picotool uf2 convert "target/$BIN.elf" "$OUT" --family rp2350-arm-s
rm -f "target/$BIN.elf"

echo
picotool info "$OUT"
echo
ls -l "$OUT"
