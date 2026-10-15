"""The fly side: frames are validated, never resized, and resets reach the driver."""

from __future__ import annotations

import socket

import numpy as np
import pytest

from fly_driver.bridge.fly_side import ConstantDriver, open_listener, serve
from fly_driver.bridge.wire import CONTROL, decode_control, recv_exactly, send_frame
from fly_driver.interface import ControlVector

SHAPE = (4, 6, 3)


class CountingDriver(ConstantDriver):
    def __init__(self) -> None:
        super().__init__(ControlVector(0.5, 0.2, 0.0), SHAPE)
        self.resets = 0

    def reset(self, seed: int | None = None) -> None:
        self.resets += 1


def _connect(listener: socket.socket) -> socket.socket:
    return socket.create_connection(listener.getsockname()[:2])


def test_answers_every_frame_and_resets_only_on_flagged_frames() -> None:
    driver = CountingDriver()
    with open_listener("127.0.0.1", 0) as listener, _connect(listener) as car:
        for seq, is_reset in enumerate([True, False, False]):
            send_frame(car, seq, np.zeros(SHAPE, np.uint8), is_reset=is_reset)
        car.shutdown(socket.SHUT_WR)

        assert serve(driver, listener) == 3
        answers = [decode_control(recv_exactly(car, CONTROL.size)) for _ in range(3)]
    assert driver.resets == 1
    assert [seq for seq, _, _ in answers] == [0, 1, 2]
    assert all(control == driver.control for _, control, _ in answers)


def test_wrong_frame_shape_raises_instead_of_resizing() -> None:
    with open_listener("127.0.0.1", 0) as listener, _connect(listener) as car:
        send_frame(car, 0, np.zeros((8, 6, 3), np.uint8), is_reset=True)
        with pytest.raises(ValueError, match="No implicit resize"):
            serve(CountingDriver(), listener)
