#![no_std]
#![no_main]

use defmt::{info, unwrap, warn};
use embassy_executor::Spawner;
use embassy_rp::bind_interrupts;
use embassy_rp::gpio::{Level, Output};
use embassy_rp::i2c::{Config as I2cConfig, I2c, InterruptHandler};
use embassy_rp::peripherals::{PIO0, USB};
use embassy_rp::pio::{InterruptHandler as PioInterruptHandler, Pio};
use embassy_rp::pio_programs::ws2812::{Grb, PioWs2812, PioWs2812Program};
use embassy_rp::usb::{Driver, InterruptHandler as UsbInterruptHandler};
use embassy_sync::mutex::Mutex;
use embassy_time::{Duration, Ticker, with_timeout};
use embassy_usb::UsbDevice;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use smart_leds::RGB8;
use static_cell::StaticCell;
use tli493d::Error as TliError;
use defmt_rtt as _;
use {panic_probe as _};

mod sensors;

/// LED ring on the CAD Mouse MK2: 8 WS2812 pixels (see original_firmware
/// `Config::LED_COUNT`).
const LED_COUNT: usize = 8;

bind_interrupts!(struct UsbIrqs {
    USBCTRL_IRQ => UsbInterruptHandler<USB>;
});

bind_interrupts!(struct I2cIrqs {
    I2C1_IRQ => InterruptHandler<embassy_rp::peripherals::I2C1>;
});

bind_interrupts!(struct PioIrqs {
    PIO0_IRQ_0 => PioInterruptHandler<PIO0>;
});

type MyUsbDriver = Driver<'static, USB>;
type MyUsbDevice = UsbDevice<'static, MyUsbDriver>;

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    // LED ring: 8x WS2812 on D3/GPIO5 (data, driven via PIO) with a level
    // shifter enable on D1/GPIO27. Blinks continuously from boot as a
    // hardware-alive indicator, independent of USB/sensor state.
    //
    // Note: D3 is GPIO5 on the XIAO RP2350, *not* GPIO29 like on the XIAO
    // RP2040 the original C firmware's pin table was written for — the two
    // boards share the same D-to-GPIO mapping everywhere else (D1, D4, D5,
    // D8, D9, D10 are all identical), but D3 was remapped on the RP2350.
    let led_ls = Output::new(p.PIN_27, Level::High);
    let Pio {
        mut common, sm0, ..
    } = Pio::new(p.PIO0, PioIrqs);
    let ws2812_program = PioWs2812Program::new(&mut common);
    let ws2812: PioWs2812<'_, PIO0, 0, LED_COUNT, Grb> =
        PioWs2812::new(&mut common, sm0, p.DMA_CH0, p.PIN_5, &ws2812_program);
    unwrap!(spawner.spawn(led_task(led_ls, ws2812)));

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

    let usb = builder.build();

    // Background tasks
    unwrap!(spawner.spawn(usb_task(usb)));

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

/// Blinks the whole LED ring dim green as a hardware-alive heartbeat.
/// `_led_ls` is held for the task's lifetime so the level-shifter enable
/// pin stays driven high (dropping it would float the pin).
#[embassy_executor::task]
async fn led_task(_led_ls: Output<'static>, mut ws2812: PioWs2812<'static, PIO0, 0, LED_COUNT, Grb>) -> ! {
    info!("LED task started");
    let on = [RGB8::new(0, 20, 0); LED_COUNT];
    let off = [RGB8::default(); LED_COUNT];
    loop {
        ws2812.write(&on).await;
        info!("LED on");
        embassy_time::Timer::after_millis(300).await;
        ws2812.write(&off).await;
        info!("LED off");
        embassy_time::Timer::after_millis(300).await;
    }
}
