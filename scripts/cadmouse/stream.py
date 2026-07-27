"""Reading the device's binary sample stream over USB CDC.

Wire format is defined by `format_frame` in ``src/sensors.rs`` and must be kept
in step with it:

===========  ====  ==========================================
offset       size  field
===========  ====  ==========================================
0               2  magic 0xA55A
2               2  seq, wrapping frame counter
4               4  t_us, device uptime in microseconds
8              18  nine int16 raw counts, MAG1/2/3 x,y,z
===========  ====  ==========================================

The sequence number is the point of the whole exercise. The firmware increments
it on every *attempted* read including failed ones, so a gap here means a sample
was genuinely lost rather than the hand having paused -- which is the difference
between a usable velocity estimate and a quietly wrong one.
"""

from __future__ import annotations

import glob
import struct
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

FRAME_MAGIC = 0xA55A
FRAME_LEN = 26
_FRAME = struct.Struct("<HHI9h")

assert _FRAME.size == FRAME_LEN, "struct format out of step with FRAME_LEN"

#: The firmware's USB identity (`src/main.rs`): VID 0xc0de, PID 0xcafe,
#: manufacturer "CAD Mouse", product "CAD Mouse MK2", serial "00000001".
DEVICE_GLOB = "/dev/serial/by-id/*CAD_Mouse*"


def find_port() -> str:
    """Locate the device's CDC data interface.

    Prefers the stable ``by-id`` symlink over ``/dev/ttyACM*``, which renumbers
    between plug-ins. The firmware exposes one CDC class, so the first match is
    the data interface.
    """
    matches = sorted(glob.glob(DEVICE_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"no CAD Mouse found at {DEVICE_GLOB}. Is it plugged in and running "
            "firmware with the binary stream (see src/sensors.rs format_frame)?"
        )
    return matches[0]


@dataclass
class Frame:
    seq: int
    t_us: int
    counts: np.ndarray  # (9,) int16 raw counts


@dataclass
class FrameDecoder:
    """Incremental byte-stream decoder with resynchronisation.

    USB CDC carries no record boundaries, so a reader that attaches mid-frame
    lands at an arbitrary offset. Scanning for the magic word recovers, and
    `resyncs` counts how often that was needed -- a nonzero value on a settled
    link means frames are being corrupted, not merely dropped.
    """

    buffer: bytearray = field(default_factory=bytearray)
    resyncs: int = 0

    def feed(self, chunk: bytes) -> Iterator[Frame]:
        self.buffer.extend(chunk)
        while len(self.buffer) >= FRAME_LEN:
            if not (
                self.buffer[0] == (FRAME_MAGIC & 0xFF)
                and self.buffer[1] == (FRAME_MAGIC >> 8)
            ):
                start = self.buffer.find(
                    FRAME_MAGIC.to_bytes(2, "little"), 1
                )
                if start < 0:
                    # Keep one byte: the magic may straddle the next chunk.
                    del self.buffer[:-1]
                    return
                del self.buffer[:start]
                self.resyncs += 1
                continue

            magic, seq, t_us, *counts = _FRAME.unpack_from(self.buffer, 0)
            del self.buffer[:FRAME_LEN]
            if magic != FRAME_MAGIC:  # defensive; the scan above should prevent it
                self.resyncs += 1
                continue
            yield Frame(seq=seq, t_us=t_us, counts=np.array(counts, dtype=np.int16))


@dataclass
class StreamStats:
    """Link health, accumulated as frames arrive."""

    received: int = 0
    dropped: int = 0
    resyncs: int = 0
    first_t_us: int | None = None
    last_t_us: int | None = None
    _last_seq: int | None = None

    def observe(self, frame: Frame) -> None:
        if self._last_seq is not None:
            gap = (frame.seq - self._last_seq - 1) & 0xFFFF
            # A huge "gap" is a rewind (device reset), not 65k lost frames.
            if gap < 0x8000:
                self.dropped += gap
        self._last_seq = frame.seq
        self.received += 1
        if self.first_t_us is None:
            self.first_t_us = frame.t_us
        self.last_t_us = frame.t_us

    @property
    def duration_s(self) -> float:
        if self.first_t_us is None or self.last_t_us is None:
            return 0.0
        return ((self.last_t_us - self.first_t_us) & 0xFFFFFFFF) / 1e6

    @property
    def rate_hz(self) -> float:
        return self.received / self.duration_s if self.duration_s > 0 else 0.0

    @property
    def loss_fraction(self) -> float:
        total = self.received + self.dropped
        return self.dropped / total if total else 0.0

    def summary(self) -> str:
        return (
            f"{self.received} frames in {self.duration_s:.1f} s "
            f"({self.rate_hz:.0f} Hz), {self.dropped} dropped "
            f"({self.loss_fraction:.2%}), {self.resyncs} resyncs"
        )


def read_frames(
    port: str | None = None,
    timeout: float = 1.0,
    chunk: int = 4096,
) -> Iterator[tuple[Frame, StreamStats]]:
    """Yield frames from the device indefinitely, with running link stats."""
    import serial  # imported lazily so the model code needs no serial port

    port = port or find_port()
    decoder = FrameDecoder()
    stats = StreamStats()

    # Baud rate is ignored by USB CDC but pyserial requires one.
    with serial.Serial(port, baudrate=115200, timeout=timeout) as handle:
        handle.reset_input_buffer()
        while True:
            data = handle.read(max(1, min(chunk, handle.in_waiting or 1)))
            if not data:
                continue
            for frame in decoder.feed(data):
                stats.resyncs = decoder.resyncs
                stats.observe(frame)
                yield frame, stats


def collect(
    n: int,
    port: str | None = None,
    on_frame=None,
) -> tuple[np.ndarray, np.ndarray, StreamStats]:
    """Read exactly `n` frames. Returns ``(counts (n, 9), t_us (n,), stats)``."""
    counts = np.empty((n, 9), dtype=np.int16)
    times = np.empty(n, dtype=np.int64)
    stats = StreamStats()

    for i, (frame, stats) in enumerate(read_frames(port)):
        counts[i] = frame.counts
        times[i] = frame.t_us
        if on_frame is not None:
            on_frame(i, frame, stats)
        if i + 1 >= n:
            break

    return counts, times, stats
