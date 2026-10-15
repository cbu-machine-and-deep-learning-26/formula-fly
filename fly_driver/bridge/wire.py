"""The bridge's wire format: a raw frame one way, a fixed control struct the other.

Stdlib ``struct`` over one TCP connection, network byte order. The car sends a header
followed by the frame's raw bytes; the header carries the shape, so the length is implied
and the fly side can check the shape against its driver rather than guess it. The fly
answers each frame with one fixed-size control record echoing the frame's sequence number.

The sender is not trusted to have clipped anything: :func:`decode_control` clips, and
rejects NaN, exactly as :meth:`~fly_driver.interface.ControlVector.clipped` does.
"""

from __future__ import annotations

import socket
import struct

import numpy as np

from fly_driver.interface import FRAME_DTYPE, ControlVector, Frame

__all__ = [
    "CONTROL",
    "DEFAULT_PORT",
    "FRAME_HEADER",
    "MAX_FRAME_BYTES",
    "decode_control",
    "recv_exactly",
    "recv_frame",
    "send_control",
    "send_frame",
]

#: Default TCP port the fly side listens on. Upstream AssettoCorsaGym uses 2345-2347.
DEFAULT_PORT = 2350

#: ``seq, is_reset, height, width, channels``. The frame's raw uint8 bytes follow.
FRAME_HEADER = struct.Struct("!I?HHH")

#: ``seq, steer, throttle, brake, compute_ms``. ``compute_ms`` is the fly side's own time
#: for the frame, so a slow round trip can be split into network and model time.
CONTROL = struct.Struct("!Idddd")

#: Refuse frames larger than this. A corrupt header would otherwise ask for gigabytes.
MAX_FRAME_BYTES = 16 * 1024 * 1024


def send_frame(sock: socket.socket, seq: int, frame: Frame, *, is_reset: bool) -> None:
    """Send one ``(height, width, channels)`` uint8 frame.

    Args:
        sock: Connected socket.
        seq: Frame sequence number; the reply echoes it.
        frame: The frame. Its shape goes on the wire as is.
        is_reset: True on the first frame of an episode, so the fly side resets its driver.
    """
    if frame.dtype != FRAME_DTYPE or frame.ndim != 3:
        raise ValueError(f"expected a (height, width, channels) uint8 frame, got {frame.dtype}")
    height, width, channels = frame.shape
    sock.sendall(FRAME_HEADER.pack(seq, is_reset, height, width, channels) + frame.tobytes())


def recv_frame(sock: socket.socket) -> tuple[int, bool, Frame] | None:
    """Receive one frame, or ``None`` if the peer closed cleanly between frames.

    Returns:
        ``(seq, is_reset, frame)``. The frame has the shape the header declared. It is
        not checked against anything here; the consumer validates it.
    """
    header = recv_exactly(sock, FRAME_HEADER.size)
    if header is None:
        return None
    seq, is_reset, height, width, channels = FRAME_HEADER.unpack(header)
    size = height * width * channels
    if size > MAX_FRAME_BYTES:
        raise ValueError(f"frame of {size} bytes exceeds the {MAX_FRAME_BYTES}-byte limit")
    payload = recv_exactly(sock, size)
    if payload is None:
        raise ConnectionError("peer closed mid-frame")
    frame = np.frombuffer(payload, dtype=FRAME_DTYPE).reshape(height, width, channels)
    return seq, bool(is_reset), frame


def send_control(sock: socket.socket, seq: int, control: ControlVector, compute_ms: float) -> None:
    """Answer frame ``seq`` with ``control``."""
    sock.sendall(CONTROL.pack(seq, control.steer, control.throttle, control.brake, compute_ms))


def decode_control(record: bytes) -> tuple[int, ControlVector, float]:
    """Unpack one control record into ``(seq, clipped control, compute_ms)``.

    Raises:
        ValueError: If a component is NaN or infinite.
    """
    seq, steer, throttle, brake, compute_ms = CONTROL.unpack(record)
    return seq, ControlVector.clipped(steer, throttle, brake), compute_ms


def recv_exactly(sock: socket.socket, size: int) -> bytearray | None:
    """Read exactly ``size`` bytes, or ``None`` on a clean close before the first byte."""
    buffer = bytearray(size)
    view = memoryview(buffer)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:])
        if count == 0:
            if received == 0:
                return None
            raise ConnectionError("peer closed mid-message")
        received += count
    return buffer
