#!/bin/sh
picotool reboot -u -f --vid 0xc0de --pid 0xcafe
sleep 3
picotool load -x -t elf "$@"
