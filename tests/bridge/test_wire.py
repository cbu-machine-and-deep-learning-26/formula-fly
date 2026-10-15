"""The wire format: frames survive the trip, and controls are clipped on arrival."""

from __future__ import annotations

import socket

import numpy as np
import pytest

from fly_driver.bridge.wire import CONTROL, decode_control, recv_frame, send_frame
from fly_driver.interface import ControlVector


def test_frame_round_trip_keeps_seq_reset_shape_and_bytes() -> None:
    frame = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    left, right = socket.socketpair()
    with left, right:
        send_frame(left, 7, frame, is_reset=True)
        left.close()
        seq, is_reset, received = recv_frame(right)
        assert recv_frame(right) is None  # clean close between frames
    assert (seq, is_reset) == (7, True)
    np.testing.assert_array_equal(received, frame)


def test_decode_control_clips_out_of_range_components() -> None:
    seq, control, compute_ms = decode_control(CONTROL.pack(3, -4.0, 2.0, -1.0, 1.5))
    assert seq == 3
    assert control == ControlVector(steer=-1.0, throttle=1.0, brake=0.0)
    assert compute_ms == 1.5


def test_decode_control_rejects_nan() -> None:
    with pytest.raises(ValueError, match="finite"):
        decode_control(CONTROL.pack(0, float("nan"), 0.0, 0.0, 0.0))
