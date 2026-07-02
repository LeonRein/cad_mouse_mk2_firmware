#![no_std]
#![no_main]

use defmt::{info, unwrap, warn};
use embassy_executor::Spawner;
use embassy_rp::bind_interrupts;
use embassy_rp::gpio::{Level, Output};
use embassy_rp::i2c::{Config as I2cConfig, I2c, InterruptHandler};
use embassy_rp::peripherals::USB;
use embassy_rp::usb::{Driver, InterruptHandler as UsbInterruptHandler};
use embassy_sync::mutex::Mutex;
use embassy_time::{Duration, Ticker, with_timeout};
use embassy_usb::UsbDevice;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State, Sender};
use static_cell::StaticCell;
use tli493d::Error as TliError;
use defmt_embassy_usbserial as _;
use {panic_probe as _};

mod reset_interface;
mod sensors;

bind_interrupts!(struct UsbIrqs {
    USBCTRL_IRQ => UsbInterruptHandler<USB>;
});

bind_interrupts!(struct I2cIrqs {
    I2C1_IRQ => InterruptHandler<embassy_rp::peripherals::I2C1>;
});

type MyUsbDriver = Driver<'static, USB>;
type MyUsbDevice = UsbDevice<'static, MyUsbDriver>;

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    // Hold LED pins low so they don't float and light up unexpectedly.
    // XIAO RP2350: PIN_LED_DATA = D3 = GPIO29, PIN_LED_LS = D1 = GPIO27.
    let _led_data = embassy_rp::gpio::Output::new(p.PIN_29, embassy_rp::gpio::Level::Low);
    let _led_ls = embassy_rp::gpio::Output::new(p.PIN_27, embassy_rp::gpio::Level::Low);

    // ── USB setup ──
    let driver = Driver::new(p.USB, UsbIrqs);

    let config = {
        let mut config = embassy_usb::Config::new(0xc0de, 0xcafe);
        config.manufacturer = Some("CAD Mouse");
        config.product = Some("CAD Mouse MK2");
        config.serial_number = Some("00000001");
        config.max_power = 100;
        config.max_packet_size_0 = 64;
        config.composite_with_iads = true;
        config.device_class = 0xEF;
        config.device_sub_class = 0x02;
        config.device_protocol = 0x01;
        config
    };

    let mut builder = {
        static CONFIG_DESCRIPTOR: StaticCell<[u8; 512]> = StaticCell::new();
        static BOS_DESCRIPTOR: StaticCell<[u8; 256]> = StaticCell::new();
        static CONTROL_BUF: StaticCell<[u8; 128]> = StaticCell::new();
        embassy_usb::Builder::new(
            driver,
            config,
            CONFIG_DESCRIPTOR.init([0; 512]),
            BOS_DESCRIPTOR.init([0; 256]),
            &mut [],
            CONTROL_BUF.init([0; 128]),
        )
    };

    // CDC ACM for raw sensor data
    let mut data_class = {
        static STATE: StaticCell<State> = StaticCell::new();
        let state = STATE.init(State::new());
        CdcAcmClass::new(&mut builder, state, 64)
    };

    // CDC ACM for defmt logs
    let defmt_class = {
        static STATE: StaticCell<State> = StaticCell::new();
        let state = STATE.init(State::new());
        CdcAcmClass::new(&mut builder, state, 64)
    };

    reset_interface::ResetHandler::install(&mut builder);
    let usb = builder.build();

    // Background tasks
    unwrap!(spawner.spawn(usb_task(usb)));
    let (defmt_sender, _) = defmt_class.split();
    unwrap!(spawner.spawn(defmt_logger_task(defmt_sender)));

    // ── Wait for USB enumeration ──
    embassy_time::Timer::after_millis(1500).await;
    info!("Hello — defmt online");

    // ── Sensor init with 3 s timeout ──
    info!("Initializing sensors…");

    // Configure I2C bus and power outputs (only main.rs knows the pins).
    let i2c_cfg = {
        let mut c = I2cConfig::default();
        c.frequency = 400_000;
        c
    };
    // XIAO RP2350: I2C1 on D5 = GPIO7 (SCL) and D4 = GPIO6 (SDA).
    // new_async takes (peripheral, scl, sda, ...).
    let i2c = I2c::new_async(p.I2C1, p.PIN_7, p.PIN_6, I2cIrqs, i2c_cfg);

    // Per-sensor supply switches: D10 = GPIO3 (MAG1), D9 = GPIO4 (MAG2), D8 = GPIO2 (MAG3).
    let mag1_pwr = Output::new(p.PIN_3, Level::Low);
    let mag2_pwr = Output::new(p.PIN_4, Level::Low);
    let mag3_pwr = Output::new(p.PIN_2, Level::Low);

    static I2C_BUS: StaticCell<sensors::SharedBus<embassy_rp::peripherals::I2C1>> = StaticCell::new();
    let bus = I2C_BUS.init(Mutex::new(i2c));

    let sensors_init = with_timeout(
        Duration::from_secs(3),
        sensors::Sensors::init(bus, mag1_pwr, mag2_pwr, mag3_pwr),
    )
    .await;

    let mut sensors = match sensors_init {
        Ok(Ok(s)) => {
            info!("Sensors ready");
            Some(s)
        }
        Ok(Err(e)) => {
            warn!("Sensor init error: {}", defmt::Debug2Format(&e));
            None
        }
        Err(_) => {
            warn!("Sensor init timed out");
            None
        }
    };

    // ── Main loop: stream sensor data over data CDC ──
    loop {
        data_class.wait_connection().await;
        info!("Data CDC connected");

        match sensors {
            Some(ref mut s) => {
                let mut poll = Ticker::every(Duration::from_millis(1));
                loop {
                    let raw = match s.read_raw().await {
                        Ok(r) => r,
                        Err(e) => {
                            match e {
                                TliError::AdcLockup | TliError::DataNotReady => {}
                                _ => warn!("read error: {}", defmt::Debug2Format(&e)),
                            }
                            poll.next().await;
                            continue;
                        }
                    };
                    let mut buf = [0u8; 128];
                    let n = sensors::format_csv(&raw, &mut buf);
                    if data_class.write_packet(&buf[..n]).await.is_err() {
                        break;
                    }
                    info!("Sent data");
                    poll.next().await;
                }
            }
            None => {
                info!("No sensors — data CDC idle");
                // Sleep so we don't busy-loop; host disconnect will
                // naturally reset us to wait_connection on next iteration.
                embassy_time::Timer::after_secs(1).await;
            }
        }

        info!("Data CDC disconnected");
    }
}

// ── Background tasks ──

#[embassy_executor::task]
async fn usb_task(mut usb: MyUsbDevice) -> ! {
    usb.run().await
}

#[embassy_executor::task]
async fn defmt_logger_task(sender: Sender<'static, MyUsbDriver>) {
    defmt_embassy_usbserial::logger(sender).await;
}