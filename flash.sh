#!/bin/sh
set -e

DEFMT_DEV="usb-Embassy_USB-serial_example_12345678-if02"
DEFMT_PATH="/dev/serial/by-id/$DEFMT_DEV"

picotool reboot -u -f --vid 0xc0de --pid 0xcafe
sleep 3
picotool load -x -t elf "$1"

# Wait for the defmt CDC interface to re-appear after reboot
echo "Waiting for $DEFMT_PATH ..."
for _ in $(seq 1 30); do
    [ -e "$DEFMT_PATH" ] && break
    sleep 0.5
done

if [ -e "$DEFMT_PATH" ]; then
    exec defmt-print -e "$1" serial --path "$DEFMT_PATH"
else
    echo "ERROR: $DEFMT_PATH did not appear" >&2
    exit 1
fi
