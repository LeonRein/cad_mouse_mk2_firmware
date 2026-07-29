"""The host end of the wire-format contract.

The device end is `Frame::encode` in `src/protocol.rs`, which asserts its own
offsets at compile time. This checks the decoder against a frame built by hand
from the layout in the documentation, so the two ends can only agree by both
being right -- a test that built its input with `record._FRAME` would agree
with itself no matter what the firmware sends.

Frame loss and resynchronisation get their own tests because they are the two
things that actually happen on a real link, and the decoder silently doing the
wrong thing there is exactly the failure that produces a plausible-looking but
wrong recording.
"""

from __future__ import annotations

import struct

import pytest

import record


def build_frame(
    seq: int = 0x1234,
    t_us: int = 0x0102_0304,
    counts: tuple[int, ...] = (1, -2, 3, -4, 5, -6, 7, -8, 9),
    pose: tuple[float, ...] = (0.5, -0.25, 0.125, 0.001, -0.002, 0.003),
    nis: float = 9.25,
    status: int = record.STATUS_FILTER_VALID | record.STATUS_CALIBRATED,
    progress: int = 255,
) -> bytes:
    """One frame, assembled field by field from the documented offsets."""
    out = bytearray()
    out += struct.pack("<H", record.FRAME_MAGIC)
    out += struct.pack("<H", seq)
    out += struct.pack("<I", t_us)
    for c in counts:
        out += struct.pack("<h", c)
    for p in pose:
        out += struct.pack("<f", p)
    out += struct.pack("<f", nis)
    out += struct.pack("<B", status)
    out += struct.pack("<B", progress)
    assert len(out) == record.FRAME_LEN
    return bytes(out)


def test_layout_offsets_are_where_the_firmware_puts_them():
    frame = build_frame()
    assert frame[0:2] == struct.pack("<H", record.FRAME_MAGIC)
    assert struct.unpack_from("<h", frame, 8)[0] == 1  # first count
    assert struct.unpack_from("<f", frame, 26)[0] == pytest.approx(0.5)  # pose x
    assert struct.unpack_from("<f", frame, 50)[0] == pytest.approx(9.25)  # nis
    assert frame[54] == record.STATUS_FILTER_VALID | record.STATUS_CALIBRATED
    assert frame[55] == 255


def test_decoder_reads_every_field():
    decoder = record.FrameDecoder()
    (frame,) = list(decoder.feed(build_frame()))

    assert frame.seq == 0x1234
    assert frame.t_us == 0x0102_0304
    assert frame.counts == (1, -2, 3, -4, 5, -6, 7, -8, 9)
    assert frame.pose == pytest.approx((0.5, -0.25, 0.125, 0.001, -0.002, 0.003))
    assert frame.nis == pytest.approx(9.25)
    assert frame.status & record.STATUS_CALIBRATED
    assert frame.progress == 255
    assert decoder.resyncs == 0


def test_decoder_survives_being_attached_mid_frame():
    """A reader that opens the port lands at an arbitrary byte offset."""
    stream = build_frame(seq=1)[17:] + build_frame(seq=2) + build_frame(seq=3)
    decoder = record.FrameDecoder()
    frames = list(decoder.feed(stream))

    assert [f.seq for f in frames] == [2, 3]
    assert decoder.resyncs == 1


def test_decoder_reassembles_frames_split_across_reads():
    """USB reads return arbitrary chunk sizes, not whole frames."""
    stream = build_frame(seq=7) + build_frame(seq=8)
    decoder = record.FrameDecoder()

    frames = []
    for i in range(0, len(stream), 5):
        frames.extend(decoder.feed(stream[i : i + 5]))

    assert [f.seq for f in frames] == [7, 8]
    assert decoder.resyncs == 0


def test_status_names_cover_every_bit_the_firmware_can_set():
    """A bit the firmware sets and the host cannot name is a silent bit."""
    all_bits = 0
    for bit, _name in record.STATUS_NAMES:
        all_bits |= bit
    assert all_bits == 0b0011_1111
    assert record.describe_status(0) == "none"
    assert "DIVERGED" in record.describe_status(record.STATUS_DIVERGED)
