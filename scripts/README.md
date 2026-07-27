# CAD Mouse MK2 — 6-DOF pose estimation

Host-side tooling that turns the nine magnetic field readings into the knob's
six degrees of freedom, replacing the hand-tuned linear mix in the C++ reference
firmware (`misc/original_firmware/src/controllers/MotionController.cpp`).

Everything here is Python, run with [uv](https://docs.astral.sh/uv/). Nothing
runs on the device yet — the firmware streams samples and the host does the maths.

**Run every command from this `scripts/` directory** — it is the uv project root,
so `uv run observability.py` works and `uv run scripts/observability.py` from the
repo root does not.

```
cd scripts
uv run pytest                    # fast tests (~1 min)
uv run pytest -m slow            # full calibration round trip (minutes)
uv run observability.py          # what the geometry can resolve, before any hardware
```

## How it works

A physics forward model predicts what the sensors *should* read for a given knob
pose, and the estimator inverts it.

| module | role |
|---|---|
| `cadmouse/geometry.py` | board frame, ring positions, the pose convention |
| `cadmouse/magnet.py` | magnet field models: point dipole and finite-size cylinder |
| `cadmouse/forward.py` | pose → nine sensor counts. The single source of truth |
| `cadmouse/params.py` | the 45 calibration parameters, packing, JSON |
| `cadmouse/calibrate.py` | fits those parameters from a recorded session |
| `cadmouse/ukf.py` | pose from counts: Gauss-Newton and the UKF |
| `cadmouse/stream.py` | decodes the device's binary USB frames |
| `cadmouse/simulate.py` | synthetic sessions, for testing without hardware |

### Two facts worth knowing before you read the code

**The magnets are not point dipoles.** They are 6 mm across and sit 6 mm from
the sensors — exactly one diameter, where a point dipole is **+30 % wrong** on
axis, rising to +77 % when the knob is pressed down. That is a systematic shape
error, so it biases the pose rather than averaging out. `SubdipoleCylinder` is
therefore the default model, not a fallback.

**There is no ground truth during calibration.** Nothing measures the knob's
actual pose, so the per-sample poses are solved *jointly* with the calibration
parameters — see the fitting section for how that is made to converge.

**Pose scale needs a physical anchor.** Known ring geometry is *not* enough to
make the recovered pose metric: scaling every pose by *s* and dividing every gain
by *s* changes the prediction only through the model's curvature, which over a
few millimetres is under 1%. Left free, the fit reaches the noise floor while
reporting a 2 mm press as 12 mm. Both factors setting the true scale are known
independently of any fit — the TLI493D's datasheet sensitivity (15.4 counts/mT)
and the magnet's moment from its size and grade (0.080 A·m² for a 6×3 mm N35
disc) — so their product, 1.24e5 counts per model unit, is applied as a soft
±30% prior. The width is dominated by the magnet grade, which the BOM does not
state.

**Watch for clipping.** At that magnet strength the sensor rails past roughly
1.2 mm of z travel or 7° of tilt, at the `A2B6Sensitivity::Short` (2×) setting
the firmware currently uses — an N35 disc reaches ~141 mT at a 4 mm gap against
a ±133 mT range. Whether your build actually clips depends on the real magnet
grade and ring heights; `record.py --check` reports peak counts and settles it in
seconds. The remedy is `A2B6Sensitivity::Full`, doubling the range for half the
resolution.

## Recording a calibration session

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
| `free` | arbitrary six-axis motion — **never fitted**, the honest test |

You are asked to move one axis at a time, but the fit does not assume you
succeeded. Each segment gets a fitted motion *direction*, so whatever the
mechanism bleeds into the other five axes is measured and reported rather than
forced to zero — forcing it would bake the mechanism's imperfection into the
sensor calibration.

## Fitting

```
uv run fit.py data/session1.csv -o calib.json --plot
```

Read the residual report, not just the exit code:

- **Overall RMS at the noise floor** (the `rest` segments' own spread) — good.
- **RMS several times the floor** — the model is wrong for this hardware. Check
  the residual-vs-pose plot; structure there means model error, noise does not
  have structure.
- **The held-out `free` score** is the number that matters. It was excluded from
  the fit entirely, so a low residual there cannot be overfitting.

A fit takes about 45 seconds.

### How the fit works, and why it is shaped that way

Every part of this was forced by something that did not work.

**Poses are not free per sample.** Giving each sample its own 6-vector leaves
4500 residuals against 3045 unknowns — 1.5 each. The fit then drives the residual
*below* the noise floor by absorbing noise into the poses, and lands on a
completely wrong calibration while looking healthy. A guided sweep does not
explore six dimensions: the knob traces a 1-D curve. So each segment gets one
motion **direction**, shared by all its samples, plus a scalar amplitude per
sample — 45 + 30 + N unknowns, about ten residuals each.

**A cold nonlinear start does not converge.** What rescues it is that the model
is linear in the unknowns one at a time — in the gains given poses, and in the
moments given gains — so alternating exact linear solves reaches a good starting
point with no search. Keeping that linearity is why `SubdipoleCylinder` orients
its sub-dipoles by the magnet's *body* axis rather than its moment.

**The starting amplitude scale comes from physics.** A generic full-travel guess
is up to 3× wrong (this knob moves ~1 mm in z), and that error propagates into
the linear gain solve, which then fights the scale prior. Each segment is instead
scaled so the predicted field excursion matches the measured one.

**The gauge is applied before the solve, not by it.** `‖m₁‖ = 1` carries weight
1e3, so starting at 1.4 contributes a squared residual of 160 000 and the
optimiser spends its budget fixing the gauge instead of the fit.

Note what is *not* searched: the sensor solder orientation. A free 3×3 gain
matrix can represent any orientation, so it falls out of the linear solve
continuously — an earlier design that enumerated 16 shared orientations
discriminated nothing, because every candidate converged to the same answer. The
only genuinely discrete unknowns left are the six per-axis sign conventions,
searched exhaustively over 64 candidates.

### Does it work?

Measured on synthetic sessions with known parameters (`uv run pytest -m slow`),
fitting blind with no ground truth:

| quantity | recovered |
|---|---|
| pose error, translation | 0.02–0.03 mm |
| pose error, rotation | 0.06–0.09° |
| metric travel | within 1% of true |
| held-out `free` residual | at or below the noise floor |

Those pose errors sit at the Cramér–Rao bound for this geometry (below), so the
fit extracts essentially all the information the measurement contains.

The fit also reports two things about your hardware rather than the model: the
per-sensor **gain anisotropy** (how far each sensor is from a clean scaled
rotation — real per-axis mismatch or cross-axis sensitivity) and the mechanism's
**cross-coupling** per axis, which is a measured version of the axis bleed the
original firmware's README complains about.

## Live view

```
uv run live.py --calib calib.json                 # UKF
uv run live.py --calib calib.json --gauss-newton  # no-dynamics baseline
uv run live.py --raw                              # counts only
```

Push the knob along one axis and watch whether the other five stay near zero.
That is the direct comparison against the axis bleed the original author
documented in `misc/original_firmware/README.md`.

## What the geometry can resolve

`uv run observability.py` computes the Cramér–Rao bound: the best any estimator
could do from a single frame, given the layout and the sensor noise. On nominal
geometry at 4 counts of noise:

| DOF | 1σ, single frame |
|---|---|
| tx, ty | 0.030 mm |
| tz | 0.014 mm |
| rx, ry | 0.068° |
| rz | 0.096° |

Condition number 28 — **well conditioned**. Rz (twist) is the weakest axis as
expected, since 1° of twist moves a magnet only 0.28 mm tangentially, but only
by about 1.4× versus the other rotations. The layout is not the problem: the
axis bleed in the original firmware was a processing artefact, not a physical
limit.

These are single-frame numbers. A filter averaging over the ~770 Hz stream does
considerably better for motion inside its bandwidth.

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
a sample was genuinely lost rather than the hand having paused. The previous CSV
format had no such counter, which made dropped frames invisible and any velocity
estimate quietly wrong.

Counts are raw 12-bit ADC values, not millitesla; the conversion
(`geometry.COUNTS_PER_MT`) lives on the host so the scale factor has one home.

## Not done here

On-device Rust estimator; HID output; thermal drift compensation (deliberately
dropped — a runtime re-zero corrects bias drift whatever its cause); runtime
auto-rezero; the sequential-sampling skew between the three sensors.
