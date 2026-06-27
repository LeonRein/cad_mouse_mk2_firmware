use core::fmt::Write;

use defmt::info;
use embassy_embedded_hal::shared_bus::asynch::i2c::I2cDevice;
use embassy_embedded_hal::shared_bus::I2cDeviceError;
use embassy_rp::gpio::Output;
use embassy_rp::i2c::{Async, Error as RpI2cError, I2c, Instance as I2cInstance};
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_time::Delay;
use tli493d::{AddressSlot, PowerMode, A2B6Sensitivity, Tli493dA2b6};

/// I2C bus shared across the three sensors.
pub type SharedBus<T> = Mutex<NoopRawMutex, I2c<'static, T, Async>>;

/// An individual sensor's I2C device handle.
type SensorI2c<T> = I2cDevice<'static, NoopRawMutex, I2c<'static, T, Async>>;

/// Error type for sensor read operations.
pub type SensorError = tli493d::Error<I2cDeviceError<RpI2cError>>;

/// Three TLI493D-A2B6 Hall sensors on a shared I2C bus.
pub struct Sensors<T: I2cInstance + 'static> {
    mag1: Tli493dA2b6<SensorI2c<T>>,
    mag2: Tli493dA2b6<SensorI2c<T>>,
    mag3: Tli493dA2b6<SensorI2c<T>>,
}

impl<T: I2cInstance + 'static> Sensors<T> {
    /// Initialize all three sensors.
    ///
    /// `bus` must be a `&'static` reference to a shared I2C mutex (e.g. from
    /// `StaticCell`). The caller is responsible for creating and storing it.
    ///
    /// Powers sensors up one by one, running the reset sequence and
    /// reassigning I2C addresses:
    ///
    /// | Sensor | Power pin | Final address |
    /// |--------|-----------|----------------|
    /// | MAG1   | mag1_pwr  | A2 (0x78)     |
    /// | MAG2   | mag2_pwr  | A1 (0x22)     |
    /// | MAG3   | mag3_pwr  | A0 (0x35)     |
    ///
    /// All are configured for Master Controlled mode with Short sensitivity.
    pub async fn init(
        bus: &'static SharedBus<T>,
        mut mag1_pwr: Output<'static>,
        mut mag2_pwr: Output<'static>,
        mut mag3_pwr: Output<'static>,
    ) -> Result<Self, SensorError> {
        // ── Power-off all sensors (long enough to fully discharge) ──
        mag1_pwr.set_low();
        mag2_pwr.set_low();
        mag3_pwr.set_low();
        embassy_time::Timer::after_millis(500).await;

        let mut delay = Delay;

        // ── MAG 1 (bottom): discover current address, then reassign to A2 ──
        info!("MAG1: power on, discovering…");
        mag1_pwr.set_high();
        embassy_time::Timer::after_millis(50).await;
        let mut mag1 = Self::find_sensor(bus, &mut delay).await?;
        mag1.set_address_slot(AddressSlot::A2).await?;
        mag1.set_sensitivity(A2B6Sensitivity::Short).await?;
        info!("MAG1: ready at A2");

        // ── MAG 2 (top-left): discover current address, then reassign to A1 ──
        info!("MAG2: power on, discovering…");
        mag2_pwr.set_high();
        embassy_time::Timer::after_millis(50).await;
        let mut mag2 = Self::find_sensor(bus, &mut delay).await?;
        mag2.set_address_slot(AddressSlot::A1).await?;
        mag2.set_sensitivity(A2B6Sensitivity::Short).await?;
        info!("MAG2: ready at A1");

        // ── MAG 3 (top-right): discover current address, stays as-is ──
        info!("MAG3: power on, discovering…");
        mag3_pwr.set_high();
        embassy_time::Timer::after_millis(50).await;
        let mut mag3 = Self::find_sensor(bus, &mut delay).await?;
        mag3.set_sensitivity(A2B6Sensitivity::Short).await?;
        info!("MAG3: ready");

        info!("Sensors ready");
        Ok(Self { mag1, mag2, mag3 })
    }

    /// Try each address slot; return the first sensor that initializes.
    async fn find_sensor(
        bus: &'static SharedBus<T>,
        delay: &mut Delay,
    ) -> Result<Tli493dA2b6<SensorI2c<T>>, SensorError> {
        for &slot in &AddressSlot::ALL {
            match tli493d::Tli493d::new(
                I2cDevice::new(bus),
                delay,
                slot,
                PowerMode::MasterControlled,
            )
            .await
            {
                Ok(s) => {
                    info!("  found at 0x{:02X}", slot.as_7bit());
                    return Ok(s);
                }
                Err(_) => continue,
            }
        }
        // Last attempt to get a proper error
        tli493d::Tli493d::new(
            I2cDevice::new(bus),
            delay,
            AddressSlot::A0,
            PowerMode::MasterControlled,
        )
        .await
    }

    /// Read raw 12-bit values from all three sensors.
    ///
    /// Returns `[mag1_x, mag1_y, mag1_z,  mag2_x, mag2_y, mag2_z,  mag3_x, mag3_y, mag3_z]`.
    pub async fn read_raw(&mut self) -> Result<[i16; 9], SensorError> {
        let r1 = self.mag1.read_raw().await?;
        let r2 = self.mag2.read_raw().await?;
        let r3 = self.mag3.read_raw().await?;
        Ok([r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, r3.x, r3.y, r3.z])
    }
}

/// Format 9 raw sensor values as a CSV line into `buf`.
///
/// Returns the number of bytes written. If the buffer is too small,
/// the output is truncated.
pub fn format_csv(raw: &[i16; 9], buf: &mut [u8]) -> usize {
    struct BufWriter<'a> {
        buf: &'a mut [u8],
        pos: usize,
    }
    impl Write for BufWriter<'_> {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let bytes = s.as_bytes();
            let rem = self.buf.len() - self.pos;
            let n = bytes.len().min(rem);
            self.buf[self.pos..self.pos + n].copy_from_slice(&bytes[..n]);
            self.pos += n;
            if n < bytes.len() {
                Err(core::fmt::Error)
            } else {
                Ok(())
            }
        }
    }

    let mut w = BufWriter { buf, pos: 0 };
    for (i, &v) in raw.iter().enumerate() {
        if i > 0 {
            let _ = write!(w, ",");
        }
        let _ = write!(w, "{}", v);
    }
    let _ = write!(w, "\n");
    w.pos
}
