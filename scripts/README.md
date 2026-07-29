# CAD Mouse MK2 — host-side tooling

The firmware streams nine raw magnetic field readings over USB; this is where
the model behind them is developed, fitted and checked. The estimation itself
now runs on the device — see [Two calibrations](#two-calibrations) and
[Exporting to the firmware](#exporting-to-the-firmware) — and the host tooling
is what produces the calibration it runs with, and the reference it is checked
against.

Present and working end to end: the magnet model, the measurement function and
its Jacobian, the session loader, the calibration fit, the filter, the golden
vectors, and the Rust port they check. Absent: HID output, and the sleep state.

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

- Sensor ring: radius 16.51 mm at z = −18 mm — real pick-and-place coordinates,
  MAG1 (0, −16.51), MAG2 (−14.30, +8.26), MAG3 (+14.30, +8.26), i.e. ring angles
  −90°, 150°, 30°. MAG1 faces the user.
- Magnet ring at rest: nominal radius 16 mm at z = −12 mm. The 12 mm depth is a
  design value, not measured.
- Magnets are 6 × 3 mm discs (grade not stated in the BOM; N35 gives ≈0.080 A·m²).
  They sit about one diameter from the sensors. A point dipole is better here
  than the usual rule of thumb suggests — the stacked pair is as long as it is
  wide, which nearly cancels the leading multipole correction, leaving ~2 % on
  axis at the operating gap rather than the ~30 % claimed here previously.
  Still ~15 counts against a 1-count noise floor, so `cadmouse.magnet` uses an
  interpolated finite-size model; the dipole survives only as a fallback.

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
| rms, counts | 1.41 | 0.57–0.81 | **0.89** |
| rms / σ | 1.26 | 0.56–0.80 | **0.85** |

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

**Worth checking physically:** the fitted moments are 55 % of what a stacked
pair of N35 discs would give, but 87 % of what a *single* 3 mm disc would give,
and modelling one disc lowers the held-out residual from 0.886 to 0.812. That
is suggestive rather than conclusive, but if there is one magnet per position
rather than two, `MAGNET_HEIGHT` in `geometry.py` is wrong.

## Filtering

```
uv run python -m cadmouse.filter data/session1.csv calibration.json
```

Two estimators behind one interface, on the six-element pose with a
random-walk process model. **Ship the IEKF.** On the full held-out segment:

| | mean NIS (target 9) | in 95 % band | innovation rms | time |
|---|---|---|---|---|
| IEKF | 10.67 | 85.3 % | 1.44 counts | 23.5 s |
| UKF (FilterPy) | 10.64 | 85.5 % | 1.44 counts | 49.3 s |

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

The band tops out near 85 %, not 95 %, and no Q or R reaches it. **The cause is
residual calibration error, not the sequential readout.** The evidence:

- Innovations on `free` stay autocorrelated at ρ ≈ 0.4–0.5 out past 100 frames
  (50 ms). White sensor noise would give ρ ≈ 0, so the excess is systematic and
  varies with pose.
- On `rest` the innovations are white and their rms is 0.99 counts against a
  sensor σ of 1.08 — the sensor model is right where the pose is pinned.
- The arithmetic closes: √(1.44² − 1.08²) = 0.95 counts of systematic excess,
  against a held-out calibration residual of 0.89.

The sequential readout is *not* it, despite the warning higher up this file:
NIS correlates with knob speed at r = 0.03, and the band is slightly better at
10–20 mm/s (88 %) than at 3–10 (85 %). At the measured peak of 19.5 mm/s and
~333 µs of skew the knob moves 6.5 µm, worth 0.7–1.4 counts on the most
sensitive channel and ~0.3–0.6 counts typically — at or below the noise floor.

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

Writes `crates/cadmouse-model/gen/field_table.bin` (85 kB of raw little-endian
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
`[8.121126, 24.265835, 510.74326, …]` in f32 against `[8.1, 24.3, 510.7, …]`
from f64 NumPy, and the flash and RAM tables agree to 0.0 counts.

## Not done here

HID output and the buttons that will ride along with it; the sleep state; the
sequential-sampling skew between the three sensors, beyond trimming the
transition frames it corrupts (`SETTLE_S` in `dataset.py`).

Deliberately dropped: thermal drift compensation — a re-zero corrects bias
drift whatever its cause, and the device now measures its own rest at every
power-up.
