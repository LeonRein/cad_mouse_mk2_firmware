"""Frame decoding, drop detection, and resynchronisation.

The wire format is a contract with `format_frame` in ``src/sensors.rs``. These
tests pin it down from the host side so a firmware change that breaks it fails
here rather than silently producing garbage poses.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from cadmouse.stream import (
    FRAME_LEN,
    FRAME_MAGIC,
    FrameDecoder,
    StreamStats,
)


def encode(seq: int, t_us: int, counts) -> bytes:
    """Mirror of the firmware's `format_frame`."""
    return struct.pack("<HHI9h", FRAME_MAGIC, seq, t_us, *counts)


def test_frame_is_one_usb_packet():
    """A frame must fit a single 64-byte full-speed bulk packet.

    The firmware writes each frame with one `write_packet`; exceeding 64 bytes
    would truncate silently, which is exactly the trap the old CSV format fell
    into.
    """
    assert FRAME_LEN == 26
    assert FRAME_LEN <= 64
    assert len(encode(0, 0, range(9))) == FRAME_LEN


def test_round_trip():
    counts = [-2047, -1, 0, 1, 2047, 100, -100, 7, -7]
    frames = list(FrameDecoder().feed(encode(1234, 5_000_000, counts)))
    assert len(frames) == 1
    assert frames[0].seq == 1234
    assert frames[0].t_us == 5_000_000
    assert list(frames[0].counts) == counts


def test_split_across_chunks():
    """USB reads land on arbitrary boundaries, not frame boundaries."""
    payload = encode(7, 42, range(9))
    decoder = FrameDecoder()
    assert list(decoder.feed(payload[:11])) == []
    frames = list(decoder.feed(payload[11:]))
    assert len(frames) == 1 and frames[0].seq == 7


def test_multiple_frames_in_one_chunk():
    blob = b"".join(encode(i, i * 1000, [i] * 9) for i in range(5))
    frames = list(FrameDecoder().feed(blob))
    assert [f.seq for f in frames] == list(range(5))


def test_resync_after_attaching_mid_frame():
    """Attaching to a live stream lands at an arbitrary offset."""
    decoder = FrameDecoder()
    blob = encode(1, 10, range(9)) + encode(2, 20, range(9))
    frames = list(decoder.feed(blob[9:]))  # start partway into the first frame
    assert [f.seq for f in frames] == [2]
    assert decoder.resyncs >= 1


def test_magic_straddling_a_chunk_boundary_is_not_lost():
    """The scan must not discard a partial magic word at the buffer tail."""
    decoder = FrameDecoder()
    junk = b"\x00" * 30
    payload = encode(99, 1, range(9))
    assert list(decoder.feed(junk + payload[:1])) == []
    frames = list(decoder.feed(payload[1:]))
    assert [f.seq for f in frames] == [99]


# ------------------------------------------------------------------- statistics


class _F:
    def __init__(self, seq, t_us):
        self.seq, self.t_us = seq, t_us
        self.counts = np.zeros(9, dtype=np.int16)


def test_stats_counts_dropped_frames():
    stats = StreamStats()
    for seq in (0, 1, 2, 5, 6):  # 3 and 4 lost
        stats.observe(_F(seq, seq * 1300))
    assert stats.received == 5
    assert stats.dropped == 2
    assert stats.loss_fraction == pytest.approx(2 / 7)


def test_stats_handles_sequence_wraparound():
    """seq is a u16; wrapping must not read as 65k lost frames."""
    stats = StreamStats()
    stats.observe(_F(65534, 0))
    stats.observe(_F(65535, 1300))
    stats.observe(_F(0, 2600))
    stats.observe(_F(1, 3900))
    assert stats.dropped == 0


def test_stats_ignores_a_device_reset():
    """A rewound counter is a reboot, not a colossal loss burst."""
    stats = StreamStats()
    stats.observe(_F(5000, 0))
    stats.observe(_F(0, 1300))
    assert stats.dropped == 0


def test_stats_rate():
    stats = StreamStats()
    for i in range(771):
        stats.observe(_F(i, i * 1298))
    assert stats.rate_hz == pytest.approx(770, rel=0.02)
