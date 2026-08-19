use defmt::{info, warn};
use embassy_embedded_hal::shared_bus::I2cDeviceError;
use embassy_embedded_hal::shared_bus::asynch::i2c::I2cDevice;
use embassy_rp::gpio::Output;
use embassy_rp::i2c::{Async, Error as RpI2cError, I2c, Instance as I2cInstance};
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_time::{Delay, Duration, with_timeout};
use embedded_hal_async::i2c::I2c as _;
use tli493d::{
    A2B6, A2B6Sensitivity, AddressSlot, BxByBz, PowerMode, RawReading, Tli493d, TriggerMode,
};

/// I2C bus shared across the three sensors.
pub type SharedBus<T> = Mutex<NoopRawMutex, I2c<'static, T, Async>>;

/// An individual sensor's I2C device handle.
type SensorI2c<T> = I2cDevice<'static, NoopRawMutex, I2c<'static, T, Async>>;

/// One sensor, configured for X/Y/Z output with temperature disabled.
///
/// Spelled out rather than via `Tli493dA2b6`, whose measurement-shape parameter
/// is fixed at the driver's `BxByBzTemp` default.
type Sensor<T> = Tli493d<SensorI2c<T>, A2B6, BxByBz>;

/// Error type for sensor read operations.
pub type SensorError = tli493d::Error<I2cDeviceError<RpI2cError>>;

/// Ceiling on a single sensor read.
///
/// Reads block on clock stretching until the conversion finishes (user manual
/// §2.2.2), which is the mechanism the whole readout strategy rests on -- but it
/// means a sensor that stops releasing SCL would hang the bus forever. This
/// bounds that. It is a fault backstop, not a tuning knob: a healthy conversion
/// is orders of magnitude below it.
const READ_TIMEOUT: Duration = Duration::from_millis(10);

/// Why a sensor read did not produce a sample.
#[derive(Debug)]
pub enum ReadError {
    /// The driver rejected the frame, or the bus transaction failed.
    Sensor(SensorError),
    /// The sensor held the bus past [`READ_TIMEOUT`].
    Timeout,
}

impl defmt::Format for ReadError {
    fn format(&self, f: defmt::Formatter) {
        match self {
            // `tli493d::Error` carries a bus error that has no `defmt::Format`,
            // so the inner value goes through `Debug`.
            Self::Sensor(e) => defmt::write!(f, "{}", defmt::Debug2Format(e)),
            Self::Timeout => defmt::write!(f, "bus timeout after {} ms", READ_TIMEOUT.as_millis()),
        }
    }
}

/// Three TLI493D-A2B6 Hall sensors on a shared I2C bus.
pub struct Sensors<T: I2cInstance + 'static> {
    mag1: Sensor<T>,
    mag2: Sensor<T>,
    mag3: Sensor<T>,
}

impl<T: I2cInstance + 'static> Sensors<T> {
    /// Initialize all three sensors.
    ///
    /// `bus` must be a `&'static` reference to a shared I2C mutex (e.g. from
    /// `StaticCell`). The caller is responsible for creating and storing it.
    ///
    /// Powers sensors up one by one and reassigns I2C addresses:
    ///
    /// | Sensor | Power pin | Final address |
    /// |--------|-----------|----------------|
    /// | MAG1   | mag1_pwr  | A2 (0x78)     |
    /// | MAG2   | mag2_pwr  | A1 (0x22)     |
    /// | MAG3   | mag3_pwr  | A0 (0x35)     |
    ///
    /// This mirrors the C++ firmware sequence:
    /// - all power rails off
    /// - power on MAG1 at A0, move to A2
    /// - power on MAG2 at A0, move to A1
    /// - power on MAG3 at A0 (kept at A0)
    ///
    /// Sensors run in master-controlled mode with trigger-on-read; see
    /// [`bring_up`] for why.
    pub async fn init(
        bus: &'static SharedBus<T>,
        mut mag1_pwr: Output<'static>,
        mut mag2_pwr: Output<'static>,
        mut mag3_pwr: Output<'static>,
    ) -> Result<Self, SensorError> {
        // Match C++ startup: force all rails low first.
        mag1_pwr.set_low();
        mag2_pwr.set_low();
        mag3_pwr.set_low();
        embassy_time::Timer::after_millis(5).await;

        let mut delay = Delay;

        // Bring each sensor up one at a time: each powers on at the default
        // address A0, then (except MAG3) is moved to a distinct slot before the
        // next one is powered on, so they don't collide on the shared bus.
        let mag1 = bring_up(
            bus,
            &mut delay,
            &mut mag1_pwr,
            Some(AddressSlot::A2),
            "MAG1",
        )
        .await?;
        let mag2 = bring_up(
            bus,
            &mut delay,
            &mut mag2_pwr,
            Some(AddressSlot::A1),
            "MAG2",
        )
        .await?;
        let mag3 = bring_up(bus, &mut delay, &mut mag3_pwr, None, "MAG3").await?;

        info!("Sensors ready");
        Ok(Self { mag1, mag2, mag3 })
    }

    /// Read raw 12-bit values from all three sensors.
    ///
    /// Returns `[mag1_x, mag1_y, mag1_z,  mag2_x, mag2_y, mag2_z,  mag3_x, mag3_y, mag3_z]`.
    ///
    /// Each read returns the conversion this sensor started at the end of the
    /// *previous* call and starts the next one, so the three samples come from
    /// conversions staggered by one readout -- roughly 190 µs at 400 kHz,
    /// against the 1.3 ms of unknown phase the free-running low-power mode gave.
    /// That skew is well under the single-frame position noise floor, so the
    /// three vectors can be treated as simultaneous.
    pub async fn read_raw(&mut self) -> Result<[i16; 9], ReadError> {
        let r1 = read_one(&mut self.mag1).await?;
        let r2 = read_one(&mut self.mag2).await?;
        let r3 = read_one(&mut self.mag3).await?;
        Ok([r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, r3.x, r3.y, r3.z])
    }

    /// Raw `DIAG` bytes of the last successful read of each sensor, MAG1..MAG3.
    pub fn diag_bytes(&self) -> [u8; 3] {
        [
            self.mag1.diagnostics().raw,
            self.mag2.diagnostics().raw,
            self.mag3.diagnostics().raw,
        ]
    }
}

/// Read one sensor, bounding how long it may hold the shared bus.
///
/// Note the timeout cancels an in-flight I2C transaction. That is not clean --
/// aborting mid-byte can leave SDA held by the sensor -- but it only happens
/// when the bus is already broken, and hanging forever is the worse failure.
async fn read_one<T: I2cInstance + 'static>(
    sensor: &mut Sensor<T>,
) -> Result<RawReading, ReadError> {
    match with_timeout(READ_TIMEOUT, sensor.read_raw()).await {
        Ok(Ok(reading)) => Ok(reading),
        Ok(Err(e)) => Err(ReadError::Sensor(e)),
        Err(_) => Err(ReadError::Timeout),
    }
}

/// Power on a single sensor and configure it.
///
/// The sensor powers up at the default address A0; if `target` is `Some`, it is
/// moved to that slot. `label` is used only for log messages.
///
/// # Readout strategy
///
/// The sensor is put in **master-controlled mode** with **trigger-on-read**, and
/// relies on the **clock stretching** the driver already enables (`CA=0`,
/// `INT=1`). Together these make the read loop self-clocking:
///
/// - Addressing the sensor while a conversion is running makes it hold SCL low
///   until the conversion completes (user manual §2.2.2), so a read can never
///   return stale data and never has to be retried.
/// - `TRIG=AfterReg05` makes that same read start the next conversion
///   (§1.2.3, Figure 5), so no separate trigger transaction is needed.
///
/// The loop therefore settles at exactly the sensor's sustainable rate with no
/// wasted bus transactions, whatever that rate turns out to be. The previous
/// low-power configuration polled a free-running sensor instead, so most reads
/// found the frame counter unchanged and were discarded -- 92% of them.
///
/// Temperature is disabled (`DT=1`): nothing downstream uses it, and dropping it
/// shortens the conversion cycle. Short-range sensitivity (2x) is kept
/// deliberately even though `X2` lengthens the ADC integration time, because the
/// pose fit sits at its noise-limited accuracy bound and halving the signal would
/// cost more than the extra rate is worth.
async fn bring_up<T: I2cInstance + 'static>(
    bus: &'static SharedBus<T>,
    delay: &mut Delay,
    pwr: &mut Output<'static>,
    target: Option<AddressSlot>,
    label: &str,
) -> Result<Sensor<T>, SensorError> {
    info!("{}: power on", label);
    pwr.set_high();
    embassy_time::Timer::after_millis(5).await;

    let new_result = tli493d::Tli493d::new(
        I2cDevice::new(bus),
        delay,
        AddressSlot::A0,
        PowerMode::MasterControlled,
    )
    .await;
    let mut sensor = match new_result {
        Ok(s) => s,
        Err(e) => {
            // `pwr` is still driven high here (unlike in main.rs, where the
            // pin is dropped/floated once `Sensors::init` returns its error),
            // so this scan actually sees whatever is on the bus for `label`.
            warn!("{}: init failed, scanning bus", label);
            scan_bus(bus).await;
            return Err(e);
        }
    };

    if let Some(slot) = target {
        sensor.set_address_slot(slot).await?;
    }
    // A2B6 supports Full and Short (2x); EXTRA_SHORT is not available.
    sensor.set_sensitivity(A2B6Sensitivity::Short).await?;
    // Every read starts the next conversion.
    sensor.set_trigger_mode(TriggerMode::AfterReg05).await?;
    // No `set_update_rate`: PRD only sets the *low-power* update period, which
    // has no effect once conversions are master-triggered.

    // Disable temperature. This must come last of the config writes: it consumes
    // the driver to change the measurement-shape type, and it preserves the
    // register cache, so TRIG and X2 set above survive.
    let mut sensor = sensor.into_measurement_mode::<BxByBz>().await?;

    // Prime the pipeline. Nothing has converted yet in master-controlled mode,
    // so without this the first read of the main loop would find PD0 = 0 and be
    // rejected. One explicit trigger plus one discarded read leaves the sensor
    // in the steady state trigger-on-read expects: a conversion always in flight.
    sensor.trigger().await?;
    match with_timeout(READ_TIMEOUT, sensor.read_raw()).await {
        Ok(Ok(_)) => {
            // DIAG is logged because PD3 is documented as "temperature ADC
            // conversion complete" (§1.2.5) and we have just disabled the
            // temperature conversion. The driver no longer gates on PD3 for
            // this measurement shape; this shows what the sensor actually does
            // with it. Expect PD0 (0x04) set on a healthy frame.
            info!("{}: primed, DIAG=0x{:02x}", label, sensor.diagnostics().raw);
        }
        Ok(Err(e)) => warn!(
            "{}: priming read failed: {}",
            label,
            defmt::Debug2Format(&e)
        ),
        Err(_) => warn!(
            "{}: priming read timed out — clock stretching may not be working",
            label
        ),
    }

    info!("{}: ready", label);
    Ok(sensor)
}

/// Probe every valid 7-bit I2C address and log which ones ACK.
///
/// Diagnostic helper for when sensor init fails: distinguishes "nothing on
/// the bus" (wiring/power/pull-up problem) from "something answered, but not
/// at the expected address" (address-sampling or driver-state problem).
async fn scan_bus<T: I2cInstance + 'static>(bus: &'static SharedBus<T>) {
    let mut dev = I2cDevice::new(bus);
    let mut buf = [0u8; 1];
    info!("I2C bus scan:");
    let mut found = false;
    for addr in 0x08u8..=0x77 {
        if dev.read(addr, &mut buf).await.is_ok() {
            info!("  ACK at 0x{:02x}", addr);
            found = true;
        }
    }
    if !found {
        info!("  no devices responded");
    }
}

// The wire format that used to live here is now in `protocol.rs`: it is a
// contract with the host tooling rather than a property of the sensors, and it
// grew fields the sensors know nothing about.
