# CAD Mouse MK2 — firmware

A six-degree-of-freedom input device (a "space mouse"): a knob on a spring
suspension carrying three magnets, three Hall sensors watching them, and a
Kalman filter turning nine magnetic field readings into a pose. Rust,
`no_std`, Embassy, on a Seeed XIAO RP2350.

The device reports **absolute** axis positions over USB HID, not deltas, and
measures its own zero at every power-up.

---

## Hardware

| function | pin | GPIO |
|---|---|---|
| right button | D0 | 26 |
| LED level shifter | D1 | 27 |
| left button | D2 | 28 |
| LED data | D3 | **5** |
| I2C1 SDA | D4 | 6 |
| I2C1 SCL | D5 | 7 |
| MAG3 power | D8 | 2 |
| MAG2 power | D9 | 4 |
| MAG1 power | D10 | 3 |

D3 is GPIO5 on the XIAO RP2350 and GPIO29 on the XIAO RP2040 — the one D-pin
that differs between the two boards.

Three TLI493D-A2B6 Hall sensors share one I2C bus at 1 MHz, brought up one at a
time and reassigned to distinct addresses (A2/A1/A0). Eight WS2812s form the
LED ring, driven over PIO.

---

## Flashing

### Without a debug probe (the normal path)

```sh
scripts/mkuf2.sh          # builds release, writes target/cad-mouse-mk2-firmware.uf2
```

Get the board into BOOTSEL, then copy the `.uf2` onto the `RP2350` drive that
appears. It reboots into the new firmware on its own.

Two ways in, no probe required for either:

- **Hold both side buttons for ten seconds.** See [the gesture](#button-gesture).
- Hold the board's BOOTSEL button while plugging it in.

The UF2 family must be `rp2350-arm-s` — the RP2350 also has a RISC-V core and a
non-secure ARM image type, and a UF2 built for the wrong one is copied without
complaint and simply does not boot. `mkuf2.sh` gets this right.

### With a debug probe

```sh
cargo run --release                        # flash + run, streams defmt over RTT
cargo run --release --bin bench_forward    # on-target cycle counts
```

The runner is `probe-rs run --chip RP235x`, configured in `.cargo/config.toml`.

**Always `--release`.** The `dev` profile carries overflow checks and costs the
sensor loop about a third of its rate; `build.rs` prints a warning on every
`dev` build, and timing numbers from one are meaningless.

---

## Using it

USB ID `1209:0001` (pid.codes test PID). The kernel's generic HID driver turns
the descriptor into an evdev device with `ABS_X`..`ABS_RZ`, which is what
`spacenavd` consumes — but `spacenavd` matches a built-in table of
3Dconnexion/Logitech IDs and will not pick this up on sight. Add to
`/etc/spnavrc`:

```
device-id = 1209:0001
```

A second USB interface (CDC ACM) carries a binary debug stream: raw counts,
pose, NIS, status flags, one 56-byte frame per sample. `scripts/record.py` and
`scripts/view.py` read it. Nothing is sent unless a host raises DTR — but note
that opening it **does** change the sample rate, since the frames compete for
the same USB frames as HID.

---

## Host calibration

Fits the 27 parameters of the mechanism — magnet positions, moments, axes —
from a recorded session. Once per device. Needs [uv](https://docs.astral.sh/uv/).

```sh
cd scripts

uv run record.py --check                                    # link health first
uv run record.py -o data/session1.csv                       # guided ~90 s capture
uv run calibrate.py data/session1.csv -o calibrations/calibration.json
uv run view.py calibrations/calibration.json                # watch it live, check signs
uv run export.py calibrations/calibration.json              # bake into the firmware
```

`export.py` writes `crates/cadmouse-model/src/generated.rs` and
`crates/cadmouse-model/gen/field_table.bin`.

**Whenever the calibration or the model changes, regenerate the golden vectors
too** — they are the only check that the Rust port still agrees with the Python
it was ported from, and they go stale silently:

```sh
uv run python -m cadmouse.golden data/session1.csv calibrations/calibration.json
```

### Recording

`record.py` prompts through segments: `rest` (hands off — defines pose zero and
measures sensor noise), `tx`/`ty`/`tz`, `rx`/`ry`/`rz`, then `free` (arbitrary
motion, held out of the fit and used to score it).

**Move slowly**, several seconds per traverse. The three sensors are read
sequentially over one bus, so fast motion smears the three readings across
different poses.

### Fitting

A bundle adjustment: the 27 parameters and the pose of every fitted frame,
solved together, because there is no ground truth anywhere in the session. The
`rest` blocks pin pose zero; fixed sensor positions break the remaining gauge
freedom. Takes ~40 s.

**Check `converged`.** A stopped fit does not look stopped — cut off early this
problem still scores well on held-out data while placing magnets 2 mm from
where they converge to. The CLI refuses to write a non-converged calibration;
`--force` overrides.

---

## On-device calibration

A different thing from the host calibration, and easy to confuse with it:

| | host (`calibrate.py`) | device |
|---|---|---|
| fits | 27 mechanism parameters | rest pose, per-channel noise, deadzone |
| needs | ~90 s recording, all six axes | nothing; hands off the knob |
| takes | ~40 s | ~1 s |
| how often | once per device | every power-up, and on demand |
| stored | baked into the firmware | RAM only, never flash |

It runs unprompted at boot and on the button gesture. 256 frames to settle, then
1024 frames collected; from those it takes the mean pose as zero, the
per-channel standard deviation as `R`, and `3σ` as the per-axis deadzone.

It **refuses to finish** if the pose moved more than 0.05 mm or 0.005 rad while
measuring — a bad zero is silently wrong for as long as the device stays
powered. If that happens at boot it flashes red and **tries again**, because
until a calibration completes the HID axes report zeros, and a device sitting
on solid green reporting nothing looks like broken hardware.

Nothing is written to flash, deliberately: it costs a second to redo, and
writing flash while core 1 executes from XIP is a hazard not worth taking on.
It is also why there is no thermal drift model — a re-zero corrects the bias
whatever caused it.

---

## Implementation

### Cores

| | core 0 | core 1 |
|---|---|---|
| I2C readout, USB (HID + CDC), LEDs, buttons | ✓ | |
| the pose filter, the rest calibration | | ✓ |

Two values cross, both *latest-value*, never queued: core 0 hands over the
newest sample, core 1 publishes the newest estimate. Core 1 never blocks
core 0, so a fault there shows up as an estimate whose `seq` stops advancing
rather than as a stalled readout.

### Rates

| stage | rate | set by |
|---|---|---|
| sensor readout | **~2100–2600 Hz** | unpaced; I2C clock stretching until each conversion finishes |
| pose filter | **readout ÷ 2** (~1050–1300 Hz) | `SAMPLES_PER_ESTIMATE` |
| HID reports | **1 kHz** | `poll_ms = 1`, plus its own ticker |

The readout rate is not a constant — it varies with the sensors' own conversion
timing, and figures from 2033 to 2957 Hz have been recorded on one board in one
session. Only ever compare rates measured in the same run. The `stream:` log
line prints readout rate, filter rate and lag together for exactly this reason.

Nothing is synchronised to anything else, and deliberately so: the axes are
absolute rather than incremental, so timing jitter produces a slightly stale
value rather than accumulated drift.

### Averaging before the filter

The readout runs about twice as fast as core 1 can consume, so core 0 averages
`SAMPLES_PER_ESTIMATE = 2` readings into one measurement instead of letting the
filter drop every second sample.

This is free and strictly better. For a random walk with process PSD `q`
observed with variance `R` every `T`, the steady-state posterior goes as
`√(q·T·R)`. Dropping every second sample gives `√(q·2T·R)` — 19 % more standard
deviation. Averaging pairs gives `√(q·2T·R/2)`, which is the full-rate figure
back again.

Measured on the device: the rest calibration reports **0.71 counts** of noise
with averaging against **0.94** without, and the filter consumes exactly every
second sample with none dropped.

The mean is accumulated in `i32` and rounded back to `i16`, so nothing
downstream changes and the calibration keeps its exact integer statistics. That
rounding adds 1/12 count² — but the calibration then measures the rounded
stream directly, so **`R` stays honest without anybody scaling it by hand**.

### The filter

An **iterated extended Kalman filter** (`crates/iekf`), six states against nine
measurements:

- **State**: three translations (mm) and a rotation vector (rad) about the
  knob's neutral centre.
- **Measurement**: nine ADC counts, MAG1/2/3 × x,y,z.
- **Process model**: random walk. `Q = diag(0.02 mm²/s, 3e-4 rad²/s)`.
- **Iterations**: 2 Gauss-Newton passes per update, relinearising about the
  current estimate while holding the prior fixed. The extra `(prior_x − x)`
  term in the residual is what makes it *iterated* rather than merely repeated.
- **Covariance**: Joseph form plus explicit symmetrisation, which is what makes
  `f32` hold up over long runs.

Iterated rather than unscented because the measurement function dominates the
cost: a UKF at `N = 6` evaluates `h` thirteen times per step, this evaluates it
twice.

### The measurement model

`crates/cadmouse-model`. Each sensor sees the sum of all three magnets — the two
far ones are worth 8–14 counts against a one-count noise floor, so none can be
dropped. Nine field evaluations per Jacobian.

Each evaluation interpolates a precomputed `(ρ, z)` table of the field of one
axially magnetised cylinder: 153 × 81 grid at 0.25 mm, bicubic (Keys, a = −1/2).
Value and both partial derivatives fall out of the same 4×4 gather, which is
what makes an analytic Jacobian nearly free.

The Jacobian is taken with respect to a *local* perturbation and converted to
the vector convention by the right Jacobian of SO(3) — mixing the two produces
a filter that works beautifully near zero and degrades with no obvious cause.

### Code and data placement

The RP2350 executes from external QSPI flash through a small XIP cache, and the
working set does not fit.

- The 97 kB field table is **copied into SRAM at boot**.
- `filter_step` is placed in `.data`, so it **executes from SRAM**. Measured,
  one whole step: 89 230 cycles from flash against **60 892 from SRAM** — 406 µs
  at 150 MHz.

**Anything added to `filter_step`'s call graph must be inlinable into it**, or
it silently moves back to flash and takes the difference with it. This has
already happened once unnoticed.

Note that the step is not what core 1 costs per sample: the task *around* it
takes **644 µs**, and the difference is code that still runs from flash.

### Robustness without a debug probe

Once the board is installed, several things that are merely inconvenient on the
bench become indistinguishable from broken hardware.

| | on the bench | installed |
|---|---|---|
| a panic | breaks into the debugger | logs, then resets |
| either core wedging | visible in the log | watchdog resets after 2 s |
| boot calibration aborting | a `warn!` | retries rather than sitting on green reporting nothing |
| a dead sensor | a flood of `warn!` | ring goes solid red after ~500 consecutive failures |
| defmt logging | drained by probe-rs | RTT is non-blocking; frames drop, nothing stalls |

The panic handler picks between halting and resetting by reading `C_DEBUGEN`, so
it is the same binary in both cases — there is no separate "release" behaviour
that was never tested.

The watchdog is fed **only while both cores are making progress**: core 0 being
alive says nothing about the estimator, and an estimator that has stopped is
exactly the failure worth catching. It is disabled before entering the
bootloader, since the loop that feeds it is about to stop existing.

### Button gesture

Hold both side buttons:

| held | ring | release here |
|---|---|---|
| 0–5 s | green filling | nothing |
| **5 s** | **solid blue** | **rest calibration** |
| 5–10 s | yellow filling over blue | rest calibration |
| **10 s** | **solid red** | **reboots into the USB bootloader** |

The colour says what releasing *now* would do; the fill says how far away the
next thing is. The red persists across the reboot — WS2812s hold their last
value — so a device waiting for a UF2 never looks like a dead one.

The calibration fires on **release**, not on reaching five seconds, so that the
calibration point can be passed on the way to the bootloader without also
triggering one.

---

## Tests

```sh
cargo test --target x86_64-unknown-linux-gnu -p cadmouse-model -p iekf
```

The explicit `--target` is needed because `.cargo/config.toml` pins builds to
the RP2350. Both library crates build for the host, so the arithmetic is
checked without a board attached.

| suite | checks |
|---|---|
| `tests/golden.rs` | the Rust port against f64 NumPy: forward model, Jacobian, a 400-frame filter trajectory |
| `tests/snapshot.rs` | that behaviour has not moved — 21 808 values, tolerances per quantity |
| unit tests | deadzone, axis shaping, rest calibration, linear algebra |

The host side has its own suite, and it is the more interesting one — it checks
properties of the *estimator* (innovation whiteness, NIS consistency, drift)
rather than of the port:

```sh
cd scripts && uv run --group dev pytest -q      # 67 tests, ~60 s
```

Two things worth knowing:

- **Nothing in `src/` is tested.** The binary has `test = false` and the crate
  cannot build for the host, so any `#[test]` there would never execute. The
  readout loop, the LED rendering and the button gesture are verified on the
  device instead — by fault injection where that is possible.
- **`filter_trajectory_matches_the_python` passes a deliberately wide NIS
  band.** A well-tuned filter puts the normalised innovation squared at 9 (the
  channel count); this one measures 4.34 on the recording and 5.64 on hardware,
  meaning `R` and/or `Q` are larger than the data warrants and the filter
  smooths more than it needs to. A defensible trade for a knob, but an
  unexamined one. See the comment there.

---

## Layout

```
src/                     firmware: bring-up, readout loop, cores, USB, LEDs, buttons
crates/iekf/             the filter — general purpose, knows nothing about magnets
crates/cadmouse-model/   this knob's measurement model, field table, calibration
scripts/                 host tooling: record, calibrate, view, export, golden vectors
scripts/mkuf2.sh         package a release build as a UF2
```

`scripts/README.md` covers the host side in far more detail, including why the
calibration is shaped the way it is and what goes wrong when it is not.
