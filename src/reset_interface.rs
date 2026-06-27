//! Reset Interface for picotool compatibility.
//!
//! This module implements the vendor-specific USB reset interface that picotool
//! uses to reboot the RP2350 into BOOTSEL mode (USB mass storage bootloader).
//!
//! Based on the original C implementation from pico-sdk:
//!   pico_stdio_usb/reset_interface.c

use embassy_usb::control::{OutResponse, Recipient, Request, RequestType};
use embassy_usb::driver::Driver;
use embassy_usb::types::InterfaceNumber;
use embassy_usb::{Builder, Handler};

/// Vendor-specific subclass for the reset interface.
const RESET_INTERFACE_SUBCLASS: u8 = 0x00;
/// Vendor-specific protocol for the reset interface.
const RESET_INTERFACE_PROTOCOL: u8 = 0x01;

/// Control request: reset to BOOTSEL (USB bootloader mode).
const RESET_REQUEST_BOOTSEL: u8 = 0x01;
/// Control request: reset to flash boot (regular boot).
const RESET_REQUEST_FLASH: u8 = 0x02;

/// Reset interface handler.
///
/// This handler responds to vendor-specific control requests on the
/// reset interface, allowing picotool to reboot the device into
/// BOOTSEL mode or perform a regular flash boot.
pub struct ResetHandler {
    /// The USB interface number assigned to the reset interface.
    iface: InterfaceNumber,
}

impl ResetHandler {
    /// Create a new ResetHandler.
    const fn new(iface: InterfaceNumber) -> Self {
        Self { iface }
    }

    /// Add the reset interface to the USB builder.
    ///
    /// This creates a vendor-specific interface (class 0xFF, subclass 0x00, protocol 0x01)
    /// that picotool recognizes as the reset interface.
    pub fn install<'d, D: Driver<'d>>(builder: &mut Builder<'d, D>) {
        // We need a static to hold the handler, since it must live as long as
        // the USB device runs.
        static HANDLER: static_cell::StaticCell<ResetHandler> = static_cell::StaticCell::new();

        // Create a vendor-specific function with our reset interface
        let mut function = builder.function(0xFF, RESET_INTERFACE_SUBCLASS, RESET_INTERFACE_PROTOCOL);
        let mut iface_builder = function.interface();
        let iface_number = iface_builder.interface_number();
        let _alt = iface_builder.alt_setting(
            0xFF, // class: vendor-specific
            RESET_INTERFACE_SUBCLASS,
            RESET_INTERFACE_PROTOCOL,
            None, // no string descriptor
        );
        drop(_alt);
        drop(iface_builder);
        drop(function);

        // Initialize the handler with the correct interface number
        let handler = HANDLER.init(ResetHandler::new(iface_number));

        // Register the handler with the USB builder
        builder.handler(handler);
    }
}

impl Handler for ResetHandler {
    /// Handle control OUT (Host-to-Device) requests on the reset interface.
    fn control_out(&mut self, req: Request, _data: &[u8]) -> Option<OutResponse> {
        // Only handle class or vendor requests to our interface
        if req.request_type != RequestType::Class
            && req.request_type != RequestType::Vendor
            || req.recipient != Recipient::Interface
            || req.index as u8 != self.iface.0
        {
            return None;
        }

        match req.request {
            RESET_REQUEST_BOOTSEL => {
                embassy_rp::rom_data::reset_to_usb_boot(0, 0);
                Some(OutResponse::Accepted)
            }
            RESET_REQUEST_FLASH => {
                // Trigger a plain watchdog reset for regular flash boot.
                let watchdog = embassy_rp::pac::WATCHDOG;
                let psm = embassy_rp::pac::PSM;
                psm.wdsel().write_value(embassy_rp::pac::psm::regs::Wdsel(
                    0x0001ffff & !(0x01 << 0) & !(0x01 << 1),
                ));
                watchdog.load().write_value(embassy_rp::pac::watchdog::regs::Load(1));
                watchdog.ctrl().modify(|w| w.set_enable(true));
                loop {
                    cortex_m::asm::nop();
                }
            }
            _ => None,
        }
    }
}
