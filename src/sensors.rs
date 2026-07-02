use core::fmt::Write;

use defmt::info;
use embassy_embedded_hal::shared_bus::asynch::i2c::I2cDevice;
use embassy_embedded_hal::shared_bus::I2cDeviceError;
use embassy_rp::gpio::Output;
use embassy_rp::i2c::{Async, Error as RpI2cError, I2c, Instance as I2cInstance};
use embassy_sync::blocking_mutex::raw::NoopRawMutex;
use embassy_sync::mutex::Mutex;
use embassy_time::Delay;
use tli493d::{AddressSlot, A2B6Sensitivity, PowerMode, Tli493dA2b6, UpdateRate};

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
    /// Sensors run in low-power mode with fast update-rate bit.
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

        // ── MAG1 (bottom): start at A0, then move to A2 ──
        info!("MAG1: power on");
        mag1_pwr.set_high();
        embassy_time::Timer::after_millis(5).await;
        let mut mag1 = tli493d::Tli493d::new(
            I2cDevice::new(bus),
            &mut delay,
            AddressSlot::A0,
            PowerMode::LowPower,
        )
        .await?;
        mag1.set_address_slot(AddressSlot::A2).await?;
        // A2B6 supports Full and Short (x2). EXTRA_SHORT is not available.
        mag1.set_sensitivity(A2B6Sensitivity::Short).await?;
        mag1.set_update_rate(UpdateRate::Fast).await?;
        info!("MAG1: ready at A2");

        // ── MAG2 (top-left): start at A0, then move to A1 ──
        info!("MAG2: power on");
        mag2_pwr.set_high();
        embassy_time::Timer::after_millis(5).await;
        let mut mag2 = tli493d::Tli493d::new(
            I2cDevice::new(bus),
            &mut delay,
            AddressSlot::A0,
            PowerMode::LowPower,
        )
        .await?;
        mag2.set_address_slot(AddressSlot::A1).await?;
        mag2.set_sensitivity(A2B6Sensitivity::Short).await?;
        mag2.set_update_rate(UpdateRate::Fast).await?;
        info!("MAG2: ready at A1");

        // ── MAG3 (top-right): start and remain at A0 ──
        info!("MAG3: power on");
        mag3_pwr.set_high();
        embassy_time::Timer::after_millis(5).await;
        let mut mag3 = tli493d::Tli493d::new(
            I2cDevice::new(bus),
            &mut delay,
            AddressSlot::A0,
            PowerMode::LowPower,
        )
        .await?;
        mag3.set_sensitivity(A2B6Sensitivity::Short).await?;
        mag3.set_update_rate(UpdateRate::Fast).await?;
        info!("MAG3: ready");

        info!("Sensors ready");
        Ok(Self { mag1, mag2, mag3 })
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
