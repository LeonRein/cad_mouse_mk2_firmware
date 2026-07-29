# CAD Mouse MK2 — host-side tooling

The firmware streams nine raw magnetic field readings over USB; turning those
into the knob's six degrees of freedom happens on the host. The previous
estimator was removed as buggy and is being rewritten from scratch; the
`cadmouse` package is that rewrite, in progress.

Present and working end to end: the magnet model, the measurement function and
its Jacobian, the session loader, the calibration fit, and the filter. Absent:
the f32 pass, the golden vectors, and the Rust port.

```
uv run pytest                            # the gates for what exists so far
```

Everything here is Python, run with [uv](https://docs.astral.sh/uv/).
**Run every command from this `scripts/` directory** — it is the uv project
root, so `uv run record.py` works and `uv run scripts/record.py` from the repo
root does not.

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
uv run python -m cadmouse.calibrate data/session1.csv -o calibration.json
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
uv run python -m cadmouse.export calibration.json
```

Writes `gen/field_table.bin` (85 kB of raw little-endian `f32`, pulled in with
`include_bytes!`) and `src/generated.rs` (grid metadata, geometry, the fitted
calibration). Both come from the same objects this package uses, so the
firmware cannot drift away from what the calibration was fitted against.

The Rust side lives in `src/magnet.rs` and `src/model.rs` — direct ports of
`cadmouse/magnet.py` and `cadmouse/model.py`, and they have to stay direct
ports or the golden vectors stop being a check of one function and become a
comparison of two.

To time it on target:

```
cargo run --bin bench_forward
```

Measured on the RP2350 at 150 MHz, `opt-level = 3`, against a 2 kHz budget of
75 000 cycles:

| | cycles | ns |
|---|---|---|
| `forward`, table in flash | 20 331 | 135 540 |
| `forward`, table in RAM | **12 298** | 81 986 |
| `forward_and_jac`, flash | 29 877 | 199 180 |
| `forward_and_jac`, RAM | **20 383** | 135 886 |
| one bicubic sample, RAM | 672 | 4 480 |

What follows from it:

- **Only the EKF family fits.** A six-state UKF needs 13 × 12 298 = 159 900
  cycles, over twice the budget; twelve-state needs 307 000. An IEKF at one to
  two iterations needs 20 400–40 800, i.e. 27–54 %. The earlier estimate of
  ~3500 cycles per `h()` was 3.5–6x optimistic and had the six-state UKF
  fitting — it does not.
- **Copy the table into RAM.** It is worth 1.65x, and 85 kB against 520 kB of
  SRAM is cheap. XIP cache misses dominate otherwise.
- **The bicubic is about half the cost** — 9 × 672 = 6050 of 12 300 cycles —
  so that is where any optimisation should start. `field_and_grad` is currently
  not inlined (nine `bl` calls per `forward`), the index arithmetic carries
  bounds checks, and `sqrtf` is a software call; all three are recoverable.

The port is validated against the host in the same run: `forward(0)` returns
`[8.120877, 24.266603, 510.744, …]` in f32 against `[8.1, 24.3, 510.7, …]` from
f64 NumPy, and the flash and RAM tables agree to 0.0 counts.

## Not done here

On-device Rust estimation; HID output; thermal drift
compensation (deliberately dropped — a runtime re-zero corrects bias drift
whatever its cause); the sequential-sampling skew between the three sensors,
beyond trimming the transition frames it corrupts (`SETTLE_S` in `dataset.py`).
