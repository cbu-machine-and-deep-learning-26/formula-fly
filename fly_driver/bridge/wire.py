"""The bridge's wire format: a raw frame one way, a fixed control struct the other.

Stdlib ``struct`` over one TCP connection, network byte order. The car sends a header
followed by the frame's raw bytes; the header carries the shape, so the length is implied
and the fly side can check the shape against its driver rather than guess it. The fly
answers each frame with one fixed-size control record echoing the frame's sequence number.

The sender is not trusted to have clipped anything: :func:`decode_control` clips, and
rejects NaN, exactly as :meth:`~fly_driver.interface.ControlVector.clipped` does.

The training env (GH-29) speaks a second, lockstep protocol on its own port, the other way
round: the trainer asks and AC answers.

- On connect the server sends one :func:`send_hello` JSON record: track, car, frame shape
  and rate, so a trainer pointed at the wrong session fails before its first step.
- The trainer sends a :data:`REQUEST`: :data:`RESET`, or :data:`STEP` with a control.
- The server answers with an :data:`OBSERVATION` header (episode number, whether AC ended
  the episode, :class:`Telemetry`) followed by a :func:`send_frame` frame.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, NamedTuple

import numpy as np

from fly_driver.interface import FRAME_DTYPE, ControlVector, Frame

__all__ = [
    "CONTROL",
    "DEFAULT_ENV_PORT",
    "DEFAULT_PORT",
    "FRAME_HEADER",
    "MAX_FRAME_BYTES",
    "OBSERVATION",
    "REQUEST",
    "RESET",
    "STEP",
    "Telemetry",
    "decode_control",
    "recv_exactly",
    "recv_frame",
    "recv_hello",
    "recv_observation",
    "recv_request",
    "send_control",
    "send_frame",
    "send_hello",
    "send_observation",
    "send_request",
]

#: Default TCP port the fly side listens on. Upstream AssettoCorsaGym uses 2345-2347.
DEFAULT_PORT = 2350

#: Default TCP port the training env server listens on.
DEFAULT_ENV_PORT = 2351

#: Request kinds.
RESET, STEP = 1, 2

#: ``kind, steer, throttle, brake``. The control is ignored for :data:`RESET`.
REQUEST = struct.Struct("!Bddd")


class Telemetry(NamedTuple):
    """What the trainer needs from AC each tick, beyond the frame.

    Attributes:
        tick: AC's ego tick counter. Consecutive ticks differ by one; a larger gap means
            the trainer missed ticks. Restarts from zero at every reset.
        time_s: AC-side clock in seconds. AC runs in real time, so this is simulation time.
        lap_dist_m: Distance along AC's track spline, ``0`` at the start/finish line.
        track_length_m: Length of AC's spline.
        lateral_m: Distance from AC's racing line (upstream's ``gap``).
        speed_mps: Car speed.
        tyres_out: Tyres off the track surface, ``0`` to ``4`` (AC's own count).
        centre_on_track: Car centre inside upstream's track occupancy grid.
    """

    tick: int
    time_s: float
    lap_dist_m: float
    track_length_m: float
    lateral_m: float
    speed_mps: float
    tyres_out: int
    centre_on_track: bool


#: ``episode, ended`` and then the :class:`Telemetry` fields. ``episode`` counts the
#: server's resets, so a client sees when another client reset the car under it.
OBSERVATION = struct.Struct("!I?IdddddB?")

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


def send_hello(sock: socket.socket, hello: dict[str, Any]) -> None:
    """Send one length-prefixed JSON record."""
    payload = json.dumps(hello).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def recv_hello(sock: socket.socket) -> dict[str, Any]:
    """Receive the record :func:`send_hello` sent."""
    header = recv_exactly(sock, 4)
    if header is None:
        raise ConnectionError("server closed before its hello")
    (size,) = struct.unpack("!I", header)
    if size > MAX_FRAME_BYTES:
        raise ValueError(f"hello of {size} bytes exceeds the {MAX_FRAME_BYTES}-byte limit")
    payload = recv_exactly(sock, size)
    if payload is None:
        raise ConnectionError("server closed mid-hello")
    return json.loads(bytes(payload).decode("utf-8"))


def send_request(sock: socket.socket, kind: int, control: ControlVector) -> None:
    """Ask for a :data:`RESET` or a :data:`STEP` with ``control``."""
    sock.sendall(REQUEST.pack(kind, control.steer, control.throttle, control.brake))


def recv_request(sock: socket.socket) -> tuple[int, ControlVector] | None:
    """One request as ``(kind, clipped control)``, or ``None`` on a clean close."""
    record = recv_exactly(sock, REQUEST.size)
    if record is None:
        return None
    kind, steer, throttle, brake = REQUEST.unpack(record)
    if kind not in (RESET, STEP):
        raise ValueError(f"unknown request kind {kind}")
    return kind, ControlVector.clipped(steer, throttle, brake)


def send_observation(
    sock: socket.socket, episode: int, ended: bool, telemetry: Telemetry, frame: Frame
) -> None:
    """Answer a request with the tick's telemetry and frame."""
    sock.sendall(OBSERVATION.pack(episode, ended, *telemetry))
    send_frame(sock, telemetry.tick, frame, is_reset=False)


def recv_observation(sock: socket.socket) -> tuple[int, bool, Telemetry, Frame]:
    """Receive ``(episode, ended, telemetry, frame)``.

    Raises:
        ConnectionError: If the server closed, which it does when the car fails.
    """
    record = recv_exactly(sock, OBSERVATION.size)
    if record is None:
        raise ConnectionError("env server closed the connection; see its log")
    episode, ended, *fields = OBSERVATION.unpack(record)
    message = recv_frame(sock)
    if message is None:
        raise ConnectionError("env server closed mid-observation")
    return episode, ended, Telemetry(*fields), message[2]
