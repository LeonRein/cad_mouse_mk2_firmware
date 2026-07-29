# CAD Mouse MK2 — host-side tooling

The firmware streams nine raw magnetic field readings over USB; this is where
the model behind them is developed, fitted and checked. The estimation itself
now runs on the device — see [Two calibrations](#two-calibrations) and
[Exporting to the firmware](#exporting-to-the-firmware) — and the host tooling
is what produces the calibration it runs with, and the reference it is checked
against.

Present and working end to end: the magnet model, the measurement function and
its Jacobian, the session loader, the calibration fit, the filter, the golden
vectors, the Rust port they check, and HID output. Absent: the sleep state.

```
uv run pytest                            # the gates for the Python
cd .. && cargo test --target x86_64-unknown-linux-gnu   # ...and for the port
```

Everything here is Python, run with [uv](https://docs.astral.sh/uv/).
**Run every command from this `scripts/` directory** — it is the uv project
root, so `uv run record.py` works and `uv run scripts/record.py` from the repo
root does not.

## The pipeline

Four steps, in order, one script each:

```
uv run record.py    -o data/session1.csv       # capture ~90 s from the device
uv run calibrate.py data/session1.csv -o calibration.json
uv run view.py      calibration.json           # watch it live, check the signs
uv run export.py    calibration.json           # emit table + calibration for the firmware
```

Each has `--help`, and each has a section below. `calibrate.py` and `export.py`
are thin wrappers over `cadmouse/calibrate.py` and `cadmouse/export.py`, which
is where the work and the explanations live.

Two more sit outside that flow. The first answers a design question rather than
producing an artefact; the second produces the vectors the Rust port is checked
against, and should be re-run whenever the model or the calibration changes:

```
uv run python -m cadmouse.filter data/session1.csv calibration.json  # IEKF vs UKF
uv run python -m cadmouse.golden data/session1.csv calibration.json  # golden vectors
```

## Two calibrations

Which is which matters, because they are easy to confuse and they answer
different questions.

| | on the PC (`calibrate.py`) | on the device |
|---|---|---|
| fits | 27 parameters of the mechanism | rest pose, sensor noise, deadzone |
| needs | a ~90 s recorded session, all six axes exercised | nothing; hands off the knob |
| costs | an optimiser and a few minutes | about a second |
| how often | once per device | every power-up, and on demand |
| delivered by | `export.py` baking it into the firmware | measured live, held in RAM |

The device one is triggered by **holding both buttons for five seconds** (the
ring fills one pixel per second so the hold is visible), and it also runs
unprompted at boot. It refuses to finish if the pose moved while it was
measuring — a bad zero is worse than no zero, since it is silently wrong for as
long as the device stays powered. `uv run record.py --check` reports whether
the attached device is calibrated, and `view.py` shows it live.

Nothing about the device calibration is written to flash, deliberately: it
costs a second to redo, and writing flash while core 1 executes from XIP is a
hazard worth not taking on for it. It is also why there is no thermal drift
model — a re-zero corrects the bias whatever caused it.

## Recording a session

Flash firmware that includes the binary stream (`format_frame` in
`src/sensors.rs`), then:

```
uv run record.py --check                 # link health: rate, loss, clipping
uv run record.py -o data/session1.csv    # guided capture, ~90 s
```

`record.py` prompts through the sequence. **Move slowly** — several seconds per
traverse. The three sensors are read sequentially over one I2C bus, so fast
motion smears the three readings across different poses.

| segment | what to do |
|---|---|
| `rest` | hands off — defines pose zero *and* measures the sensor noise |
| `tx` `ty` `tz` | slide the knob along each axis, full travel |
| `rx` `ry` `rz` | tilt forward/back, tilt left/right, twist |
| `free` | arbitrary six-axis motion — intended to be held out of any fit |

You are asked to move one axis at a time; a fit should not assume you succeeded.
Whatever the mechanism bleeds into the other five axes is a property of the
hardware worth measuring, not something to force to zero.

The output CSV is one row per frame:
`segment, seq, t_us, mag1x … mag3z`.

`record.py` is self-contained on purpose — stdlib plus pyserial, no shared
package — so it is unaffected by the estimator rewrite.

**Watch for clipping.** The sensor may rail past roughly 1.2 mm of z travel or
7° of tilt at the `A2B6Sensitivity::Short` (2×) setting the firmware currently
uses — an N35 disc reaches ~141 mT at a 4 mm gap against a ±133 mT range.
Whether your build actually clips depends on the real magnet grade and ring
heights; `record.py --check` reports peak counts and settles it in seconds. The
remedy is `A2B6Sensitivity::Full`, doubling the range for half the resolution.

## Wire format

The device sends fixed 26-byte little-endian frames, one per USB packet
(`format_frame`, `src/sensors.rs`):

| offset | size | field |
|---|---|---|
| 0 | 2 | magic `0xA55A` |
| 2 | 2 | `seq`, wrapping frame counter |
| 4 | 4 | `t_us`, device uptime |
| 8 | 18 | nine `int16` raw counts, MAG1/2/3 x,y,z |

`seq` increments on every *attempted* read including failed ones, so a gap means
a sample was genuinely lost rather than the hand having paused — the difference
between a usable velocity estimate and a quietly wrong one.

Counts are raw sign-extended 12-bit ADC values, not millitesla. With
`A2B6Sensitivity::Short` (2×) the TLI493D-A2B6 gives 7.7 × 2 = 15.4 counts/mT
(`../tli493d/src/variant.rs`); full scale is ±2047 counts.

## Board geometry

Useful for whatever replaces the estimator. Board frame: origin at the knob's
neutral centre of rotation, `+x` right, `+y` toward the rear, `+z` up.

- Sensor ring: radius 16.51 mm at z = −18 mm, ring angles −90°, 150°, 30°, so
  MAG1 (0, −16.51), MAG2 (−14.298, +8.255), MAG3 (+14.298, +8.255). MAG1 faces
  the user. Both rings are *generated* from radius, height and a shared set of
  angles (`ring()` in `geometry.py`) rather than written out as coordinates —
  each magnet sits directly over its own sensor, so the angles are one fact,
  not two that have to be kept in step by hand.
- Magnet ring at rest: nominal radius 16 mm at z = −12 mm. The 12 mm depth is a
  design value, not measured.
- Magnets are 6 × 3 mm discs, **one per position** (grade not stated in the BOM;
  N35 gives ≈0.079 A·m²). This was confirmed by opening the knob on
  2026-07-30, settling a question the fit had already raised — see
  [Calibrating](#calibrating). The drawing had been read as two discs stacked,
  and `MAGNET_HEIGHT` said 6 mm.
- The magnet's *bottom face* is what the mechanism fixes, at z = −12 mm, so
  halving the height moved the centre from z = −9 to z = −10.5 and put it
  7.5 mm from the sensor plane rather than 9. Two things downstream care:
  the field table's extent (`DEFAULT_Z_RANGE` in `magnet.py`, which had to grow
  upward, and clamps silently if it is too small) and the face quadrature
  order, which had to rise to 16 × 32 to stay exact that close in.
- A point dipole is *not* good enough. One disc is half as long as it is wide,
  so the leading multipole correction no longer cancels and the dipole
  overestimates by 18 % on axis at the operating gap — about 88 counts against
  a 1-count noise floor. `cadmouse.magnet` uses an interpolated finite-size
  model; the dipole survives only as a fallback. (While the magnet was believed
  to be a stacked pair this was a 2 % near-miss, which is why the file used to
  say the dipole was "better than the rule of thumb suggests".)

Measured against `data/session1.csv`, with nominal geometry (see
`tests/test_model.py`, which pins each of these):

- Noise floor ≈ 1 count/channel; rest drifts < 10 counts over the 90 s session.
- Nominal geometry is ~25 % too strong, and ~40 % too strong for magnet 3.
- **Magnet 3 is physically reversed** — not sensor 3 mounted upside down. The
  two hypotheses are distinguishable and were tested; see `geometry.py`.
- The two far magnets contribute 8–14 counts at each sensor, so the model sums
  over all three.
- All six DOF are strongly observable: the Jacobian's singular values span only
  a factor of 2.5, putting the single-frame noise floor at ~6 µm and ~0.013°.

## Calibrating

```
uv run calibrate.py data/session1.csv -o calibration.json
```

A bundle adjustment: 27 calibration parameters and the pose of every fitted
frame, solved together, because **there is no ground truth anywhere in the
session**. The recording asks for one axis at a time but the mechanism bleeds
into the others, so the poses are latent variables rather than known inputs.
What pins the solution is the `rest` blocks (pose zero, the datum), fixed
sensor positions (breaking the remaining gauge freedom), and weak priors aimed
at the commanded axis. `free` is excluded throughout and scored at the end.

Takes ~40 s. Current result on `session1.csv`:

| | rest | tx…rz | **free (held out)** |
|---|---|---|---|
| rms, counts | 1.41 | 0.57–0.78 | **0.81** |
| rms / σ | 1.26 | 0.56–0.77 | **0.78** |

Two things that will bite if you touch this:

- **Check `converged`.** A stopped fit does not look stopped. Cut off at 60
  evaluations this problem still scores 1.49 counts on held-out data while
  placing the magnets 2 mm from where they converge to. The CLI refuses to
  write a calibration that did not converge; pass `--force` to override.
- **Magnet polarity is detected, not assumed.** Start the fit with a sign wrong
  and it does not degrade — it drives every moment to zero within one iteration
  and parks there, because the best explanation of badly-posed frames is "there
  is no field", and a zero field has a zero Jacobian. A signed moment makes the
  sign *representable*, not *findable*.

The fit is prior-insensitive: varying the magnet-depth prior over a 12× range
moves predictions by 0.14 counts rms, so the geometry is coming from the data.

**Settled, 2026-07-30: there is one magnet per position, not two.** The fit
raised it first — the moments came out at 55 % of a stacked N35 pair, which is
not a plausible remanence spread — and opening the knob confirmed it.
`MAGNET_HEIGHT` is now 3 mm, and the held-out residual fell from 0.886 to 0.81
counts. Nominal geometry is correspondingly no longer 25 % too strong: magnets
1 and 2 now sit within a few percent of nominal and magnet 3 at ~0.83, which is
a per-device fact rather than a modelling error.

The moral is worth keeping: the calibration was *already telling* us the
geometry was wrong, in the one parameter free to absorb it. A fit that has to
move a physical constant by 45 % to explain the data is reporting a hardware
fact, not converging.

## Filtering

```
uv run python -m cadmouse.filter data/session1.csv calibration.json
```

Two estimators behind one interface, on the six-element pose with a
random-walk process model. **Ship the IEKF.** On the full held-out segment:

| | mean NIS (target 9) | in 95 % band | innovation rms | time |
|---|---|---|---|---|
| IEKF | 9.50 | 88.3 % | 1.39 counts | 15.8 s |
| UKF (FilterPy) | 9.47 | 88.5 % | 1.39 counts | 48.4 s |

These improved when the magnet geometry was corrected (they read 10.67 / 85.3 %
/ 1.44 while `MAGNET_HEIGHT` was 6 mm), which is the expected direction: the
filter's consistency is limited by calibration error, so a better model shows
up here before it shows up anywhere else.

They agree to **0.16 µm rms** in translation (max 3.3 µm, all of it in the
startup transient) and 0.001° in rotation — some thirty times finer than the
5.6 µm a single frame resolves — for half the cost. The choice is therefore
about compute, not modelling, exactly as `cond(J) = 2.5` predicted.

FilterPy is here as an **oracle, not a dependency of the shipped filter**: the
RP2350 implementation will be hand-written (no mature allocation-free `no_std`
Rust UKF exists, and nothing offers an IEKF), so it needs something independent
to be checked against. `tests/test_filter.py::test_iekf_and_ukf_agree` is that
check.

Three things worth knowing before touching the tuning:

- **Q is not a free knob.** Three orders of magnitude too large puts the UKF's
  sigma points millimetres from the mean, outside both the mechanism and the
  field table, and it diverges. The IEKF hides the same error by linearising at
  the mean — so a UKF that diverges where the IEKF looks fine is evidence about
  Q, not about the UKF.
- **The initial covariance is not harmlessly conservative** for the same
  reason: sigma points spread as `sqrt((n + lambda) P)`.
- **`alpha = 1e-3` is wrong here.** At n = 6 it makes lambda −5.999994 and the
  mean weight about −10⁶. Use `alpha = 1, kappa = 0`; since h is nearly linear
  over the posterior, the tighter spread costs nothing.

The band tops out near 88 %, not 95 %, and no Q or R reaches it. **The main
cause is residual calibration error.** The evidence:

- Innovations on `free` stay autocorrelated at ρ ≈ 0.4–0.5 out past 100 frames
  (50 ms). White sensor noise would give ρ ≈ 0, so the excess is systematic and
  varies with pose.
- On `rest` the innovations are white and their rms is 0.99 counts against a
  sensor σ of 1.08 — the sensor model is right where the pose is pinned.
- The arithmetic closes: √(1.39² − 1.08²) = 0.88 counts of systematic excess,
  against a held-out calibration residual of 0.81.

**Open, and newly so: the sequential readout may now be a visible second
term.** This used to be dismissed outright — NIS correlated with knob speed at
r = 0.03 — but with the corrected magnet the correlation is **0.185**, and
`tests/test_filter.py::test_nis_does_not_depend_on_knob_speed` fails its 0.15
gate. The threshold has deliberately *not* been relaxed to make it pass.

The likely reading is unmasking rather than regression: the speed-dependent
term did not grow, the much larger pose-dependent term shrank around it.
Correcting the magnet cut innovation rms by only 2.6 % (1.366 → 1.331 on the
reduced fit used by that test) while the correlation rose 38 %, which is the
signature of a fixed small term becoming a larger share of a smaller total. At
the measured peak of 19.5 mm/s and ~333 µs of skew the knob moves 6.5 µm, worth
0.7–1.4 counts on the most sensitive channel — no longer comfortably below a
0.88-count systematic. Deciding this needs either a skew-aware measurement
model or a recording made deliberately slowly, and until then the gate stays
red on purpose.

Skew does matter in one place, but it is not this one: at the instant the hand
*releases* the knob the pose changes discontinuously between the three reads,
and that single frame lands up to 138 counts out. See `SETTLE_S` in
`dataset.py`.

So the way to close the remaining gap is a better model, not a bigger R — which
puts the magnet-stack question above at the top of the list.

## Watching it work

```
uv run view.py calibration.json                      # live, from the device
uv run view.py calibration.json --replay data/session1.csv
```

A 3-D wireframe of the knob, six live traces, and a panel naming what the
filter thinks you are doing ("tx +0.35 mm BACK"). The filter runs on every
frame — it sustains ~2500 Hz in Python against the device's 2000 Hz — and only
the drawing is decimated.

This is not decoration. **Every automated check in this project is
self-consistent**: held-out residual, NIS, innovation whiteness all compare the
model against itself, so all of them pass just as happily if the board frame is
mirrored or two axes are swapped. The one test tying segment labels to axes
takes `abs()` of a cosine, confirming the response lies *along* the predicted
axis but not which way it points. For a 3-D mouse that is the bug that ships,
and a person moving the knob finds it in seconds.

The conventions were confirmed by hand on 2026-07-29 and are now frozen in
`test_model.py::EXPECTED_JACOBIAN_SIGNS`, which fails on mirroring the magnet
ring in x or z, flipping any magnet's polarity, or permuting channels — all of
which leave every residual- and consistency-based test in the suite green.

The `abs()` on the segment-direction cosine stays, and is not the gap it looks
like: the sign of a principal component is arbitrary, and the operator moved
each axis both ways, so there is no direction in the recording to compare
against. Even `tz` is 41 % positive. A person watching the screen is the only
anchor to physical reality here, so the frozen table records what they saw.

The 3-D view exaggerates motion (`--gain`, default 8x) because a millimetre on
a 33 mm body is invisible; the traces are always in real units.

The readout shows the filter rate and the display rate separately, because they
are unrelated and a slow *picture* otherwise reads as a slow *filter*. Getting
the picture usable took three changes, all found by profiling rather than
guessing:

| | ms/frame | fps |
|---|---|---|
| six autoscaled axes, 12 000 pts each, full redraw | 71.9 | 14 |
| two fixed axes, 600 pts, full redraw | 66.0 | 15 |
| the same, blitted | **20.3** | **49** |

Most of it was never the data: six sets of axis furniture cost ~7 ms each to
redraw whatever they contain, and autoscaling forces a full redraw every frame.
Blitted, the 3-D wireframe costs 1.2 ms and the text readout is the single most
expensive artist. Pass `--no-blit` if the blitted 3-D leaves artefacts on your
backend.

## Exporting to the firmware

```
uv run export.py calibration.json
```

Writes `crates/cadmouse-model/gen/field_table.bin` (97 kB of raw little-endian
`f32`, pulled in with `include_bytes!`) and
`crates/cadmouse-model/src/generated.rs` (grid metadata, geometry, the fitted
calibration). Both come from the same objects this package uses, so the
firmware cannot drift away from what the calibration was fitted against.

The Rust is a workspace, split along the line of what is reusable:

| crate | holds | tested by |
|---|---|---|
| `crates/iekf` | the filter itself, `no_std`, no allocation, const-generic over states and measurements. Knows nothing about magnets. | its own unit tests |
| `crates/cadmouse-model` | `magnet.rs` and `model.rs` — direct ports of `cadmouse/magnet.py` and `cadmouse/model.py` — plus the generated calibration, the tuning, and the rest calibration | `tests/golden.rs`, against `cadmouse.golden` |
| the root crate | the firmware: pins, sensors, LEDs, buttons, the wire format, and the core-1 estimator task | the board |

The two ported files have to stay *direct* ports, or the golden vectors stop
being a check of one function and become a comparison of two.

### What the golden vectors actually say

`cargo test --target x86_64-unknown-linux-gnu` runs the port against vectors
this package generated. Measured agreement, f32 Rust against f64 NumPy:

| | disagreement | for scale |
|---|---|---|
| `forward` | 0.0006 counts | the noise floor is 1.08 counts |
| `forward_and_jac_vector` | 5.5e-6 relative | — |
| 400 filtered frames | 0.002 µm, 1e-5° | one frame resolves 5.6 µm |

So the port costs nothing measurable, and the tolerances in `golden.rs` are set
about ten times looser than what was measured — a failure there means one of
the two implementations moved, not that f32 finally ran out of bits.

To time it on target:

```
cargo run --bin bench_forward
```

> **Stale as of 2026-07-30 and not yet re-measured.** Correcting the magnet
> geometry grew the field table from 85 kB to 97 kB (153 × 81 rather than
> 141 × 77), and every number below is a measurement of how a working set falls
> across a small XIP cache — see [the one that matters](#the-one-that-matters-put-the-code-in-ram-too).
> A 14 % larger table is exactly the kind of change these figures are sensitive
> to, and the RAM-table rows in particular now ask for more SRAM. Re-run
> `cargo run --bin bench_forward` on the board before trusting any of it.

Measured on the RP2350 at 150 MHz, `opt-level = 3`. The budget is 75 000 cycles
at 2 kHz, or 150 000 at 1 kHz.

| | cycles |
|---|---|
| `forward`, table in flash | 21 100 |
| `forward`, table in RAM | **11 900** |
| `forward_and_jac`, RAM | 20 100 |
| `forward_and_jac_vector`, RAM | 24 400 |
| one bicubic sample, RAM | 676 |
| filter's own algebra, no model (1 iteration) | 24 200 |
| **whole IEKF step, 2 iterations, code in flash** | **298 000** |
| **whole IEKF step, 2 iterations, code in RAM** | **141 300** |

### The one that matters: put the *code* in RAM too

The headline result is not about the algorithm. The model costs 24 400 cycles
in a tight loop and the filter's algebra costs 24 200 in a tight loop, so a
step should cost about 50 000 — and it measured 298 000. Neither the trait
call, nor the covariance's condition, nor subnormal operands accounted for any
of it, and each was ruled out by measurement rather than by argument.

The cause is that this part executes from **external QSPI flash through a
small XIP cache**. Each of those two loops fits in it; together they do not,
so every pass refetches. Placing one `#[link_section = ".data"]` function —
`estimator::filter_step` — and making its callees `inline(always)` so they come
with it cut the step by **2.1x**, with no change to the arithmetic.

Two consequences worth carrying forward:

- **Anything added to `filter_step`'s call graph must be inlinable into it**,
  or it quietly moves back to flash and takes the 2x with it.
- **Benchmark numbers here are layout-sensitive.** Adding an unrelated
  benchmark moved others by 20-40 %, because what is really being measured is
  how the working set falls across the cache. Compare configurations within one
  run, not across runs.

### Where that leaves the rate

At 141 300 cycles a step is 0.94 ms, so the filter sustains about 1060 Hz. The
readout runs at 2000-2235 Hz, so **core 1 currently sees roughly every other
sample** — it reports itself as "6 frames behind" in the device log, and the
host sees no loss because dropping is the designed behaviour, not a failure.
This costs less than it sounds (the filter is fed twice the samples its noise
model needs) but it is not the intended steady state. What is left, in the
order the benchmark says to do it:

1. Pull the rest of the call graph into RAM — `field_and_grad`,
   `rotation_from_rotvec` and `right_jacobian_so3` are still in flash.
2. Drop to one relinearisation pass, worth ~60 000 cycles, if step 1 is not
   enough. On the recorded data one to five passes are indistinguishable.
3. The bicubic, which is 9 × 676 = 6 100 of the model's cost.

The port is validated against the host in the same run: `forward(0)` returns
`[8.124218, 24.257837, 510.74368, …]` in f32 against `[8.1, 24.3, 510.7, …]`
from f64 NumPy, and the flash and RAM tables agree to 0.0 counts.

## Using it as a mouse: HID and spacenavd

The device enumerates as a **Generic Desktop multi-axis controller** — six
`int16` axes at ±350 in report 1, two buttons in report 3, polled at 1 kHz.
The descriptor is a byte-for-byte port of the original C++ firmware's, in
`src/hid.rs`, and it must stay that way: host software decides what to expect
from a device's descriptor, and older 3-D mice split translation and rotation
across two reports where this sends all six in one.

### USB identity

```
1209:0001   pid.codes community VID, test PID
```

Deliberately **not** 3Dconnexion's `256f:c631`, which the original firmware
claimed so the vendor's Windows driver would bind. On Linux nothing needs
that, so the device answers to its own name. `0x1209` is
[pid.codes](https://pid.codes)' community vendor ID; `0x0001` is one of its
test PIDs, which is fine for a personal build. A permanent PID is free for an
open-source project, which is the only condition attached to the VID.

### Making spacenavd pick it up

`spacenavd` matches a built-in table of 3Dconnexion and Logitech USB IDs, so
it will not find this device on sight. It takes one line in `/etc/spnavrc`:

```
device-id = 1209:0001
```

That is the whole difference between "a free VID" and "a cloned one" on Linux.
Restart the daemon and it is treated exactly like a retail device. `spacenavd
-v -d` prints what it considered and why if it still does not appear.

The kernel side needs nothing: `hid-generic` binds the descriptor on its own
and creates an evdev node with `ABS_X`..`ABS_RZ` and `BTN_0`/`BTN_1`, which is
what the daemon reads. Confirmed on this machine:

```
hid-generic 0003:1209:0001.0012: USB HID v1.10 Multi-Axis Controller
B: ABS=3f          # ABS_X .. ABS_RZ, each min=-350 max=350
```

### Checking the signs

```
uv run hidmon.py
```

The one thing no automated test in this repository can catch. Everything else
compares the device against itself or against the Python, and both would be
just as happy with two axes swapped or one pointing backwards. `hidmon.py`
reads evdev directly and draws a bar per axis, so pushing the knob right and
watching which number moves settles it in seconds.

If an axis is backwards, flip its entry in `AXIS_SIGN` in
`crates/cadmouse-model/src/shaping.rs`. They all start at `+1` because the
board frame and the usual multi-axis convention agree *on paper* — that is a
hypothesis, and this is how it gets tested.

### Scaling

`shaping.rs` maps **125 mm** and **100°** to full scale, one factor for
translation and one for rotation. **Not one per axis**, even though the
measured envelope differs per axis (1.24, 2.47, 0.84 mm), because normalising
each axis to its own peak warps direction: a diagonal push comes out at the
wrong angle and the device feels like it pulls to one side. The cost is that
the stiffer axes do not reach the rails, which is the right trade.

Those full-scale figures are deliberately far outside anything the mechanism
can reach — they set *sensitivity*, not range. They started at the measured
2.5 mm and 10° envelope, which put full scale within a gentle push and was
unusably fast in practice, and were then divided by fifty (translation) and
ten (rotation) by hand.

The consequence is worth knowing: the knob's real travel now reaches only
**±7 of 350** in translation and **±35** in rotation, so translation arrives in
about fifteen discrete steps. If that quantisation shows up as steppiness,
the fix is *not* to raise the sensitivity back — it is to leave the firmware
alone and turn the feel down downstream, where `spacenavd`'s own
`sensitivity` and `sensitivity-translation-*` options are floats and cost no
resolution. `tests/the_mechanisms_own_travel_reaches_only_part_of_the_range`
in `shaping.rs` pins these numbers so the trade stays visible.

The original firmware's `GAIN_T`/`GAIN_R` do **not** carry over — they scaled
raw magnetic deltas in counts, and nothing here is in counts.

## Not done here

The sleep state (the original slept after two minutes); the sequential-sampling
skew between the three sensors, beyond trimming the transition frames it
corrupts (`SETTLE_S` in `dataset.py`).

Deliberately dropped: thermal drift compensation — a re-zero corrects bias
drift whatever its cause, and the device now measures its own rest at every
power-up.
