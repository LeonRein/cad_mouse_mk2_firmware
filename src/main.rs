#![no_std]
#![no_main]

use defmt::{info, panic, unwrap};
use embassy_executor::Spawner;
use embassy_rp::bind_interrupts;
use embassy_rp::peripherals::USB;
use embassy_rp::usb::{Driver, Instance, InterruptHandler};
use embassy_usb::UsbDevice;
// We now also import the Sender for the defmt split
use embassy_usb::class::cdc_acm::{CdcAcmClass, State, Sender}; 
use embassy_usb::driver::EndpointError;
use static_cell::StaticCell;

// 1. Swap defmt-rtt for the USB serial transport
use defmt_embassy_usbserial as _; 
use {panic_probe as _};

mod reset_interface;

bind_interrupts!(struct Irqs {
    USBCTRL_IRQ => InterruptHandler<USB>;
});

type MyUsbDriver = Driver<'static, USB>;
type MyUsbDevice = UsbDevice<'static, MyUsbDriver>;

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    let driver = Driver::new(p.USB, Irqs);

    let config = {
        let mut config = embassy_usb::Config::new(0xc0de, 0xcafe);
        config.manufacturer = Some("Embassy");
        config.product = Some("USB-serial example");
        config.serial_number = Some("12345678");
        config.max_power = 100;
        config.max_packet_size_0 = 64;

        // 2. REQUIRED: Enable IADs so the host OS recognizes both 
        // the Echo interface and the Defmt interface properly.
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

    // 3. Create the CDC ACM class for your ECHO task
    let mut echo_class = {
        static STATE: StaticCell<State> = StaticCell::new();
        let state = STATE.init(State::new());
        CdcAcmClass::new(&mut builder, state, 64)
    };

    // 4. Create a SECOND CDC ACM class for the DEFMT logs
    let defmt_class = {
        static STATE: StaticCell<State> = StaticCell::new();
        let state = STATE.init(State::new());
        CdcAcmClass::new(&mut builder, state, 64)
    };

    // Install the picotool reset interface (vendor-specific control interface)
    reset_interface::ResetHandler::install(&mut builder);

    let usb = builder.build();

    // Run the core USB device
    unwrap!(spawner.spawn(usb_task(usb)));

    // 5. Split the defmt class to get its Sender, and give it to the logger task
    let (defmt_sender, _defmt_receiver) = defmt_class.split();
    unwrap!(spawner.spawn(defmt_logger_task(defmt_sender)));

    unwrap!(spawner.spawn(hartbeat_task()));

    // Allow a brief moment for USB enumeration before pushing early logs
    embassy_time::Timer::after_millis(100).await;
    info!("Hello there! Defmt is alive over USB.");

    // Loop for Echo
    loop {
        echo_class.wait_connection().await;
        info!("Echo interface connected");
        let _ = echo(&mut echo_class).await;
        info!("Echo interface disconnected");
    }
}

#[embassy_executor::task]
async fn usb_task(mut usb: MyUsbDevice) -> ! {
    usb.run().await
}

// 6. Background task that listens for defmt logs and pushes them out via USB
#[embassy_executor::task]
async fn defmt_logger_task(sender: Sender<'static, MyUsbDriver>) {
    // Note: Depending on minor crate version adjustments, the logger function 
    // may optionally require the max_packet_size: `logger(sender, 64).await;`
    defmt_embassy_usbserial::logger(sender).await;
}

#[embassy_executor::task]
async fn hartbeat_task() {
    loop {
        info!("Heartbeat");
        embassy_time::Timer::after_millis(1000).await;
    }
}

struct Disconnected {}

impl From<EndpointError> for Disconnected {
    fn from(val: EndpointError) -> Self {
        match val {
            EndpointError::BufferOverflow => panic!("Buffer overflow"),
            EndpointError::Disabled => Disconnected {},
        }
    }
}

async fn echo<'d, T: Instance + 'd>(class: &mut CdcAcmClass<'d, Driver<'d, T>>) -> Result<(), Disconnected> {
    let mut buf = [0; 64];
    loop {
        let n = class.read_packet(&mut buf).await?;
        let data = &buf[..n];
        info!("data: {:x}", data);
        class.write_packet(data).await?;
        class.write_packet("X".as_bytes()).await?;
    }
}