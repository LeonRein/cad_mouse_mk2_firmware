# CAD Mouse MK2 — host-side tooling

The firmware streams nine raw magnetic field readings over USB; turning those
into the knob's six degrees of freedom happens on the host. The previous
estimator (forward model, joint calibration fit, UKF) has been **removed** — it
was buggy and is being rewritten from scratch. What is left is data capture.

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
  They sit about one diameter from the sensors, where a point-dipole model is
  ~30 % wrong on axis — a finite-size model is a requirement, not a refinement.

## Not done here

The estimator itself; on-device Rust estimation; HID output; thermal drift
compensation (deliberately dropped — a runtime re-zero corrects bias drift
whatever its cause); the sequential-sampling skew between the three sensors.
