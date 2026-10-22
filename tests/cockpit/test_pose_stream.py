"""The pose survives the wire, and each wrapper sends exactly one pose per step."""

from __future__ import annotations

import socket
from collections.abc import Iterator

import numpy as np
import pytest

from fly_driver.brains.populations import PopulationCounts
from fly_driver.cockpit.live_pose import FlyPose
from fly_driver.cockpit.pose_stream import (
    PoseStream,
    PoseStreamDriver,
    PoseStreamEnv,
    decode_pose,
    encode_pose,
)
from fly_driver.interface import ControlVector

POPULATIONS = PopulationCounts(compass=812, steering=3252, descending=32)


class _Brain:
    populations = POPULATIONS

    def rates(self) -> np.ndarray:
        rates = np.zeros(POPULATIONS.total, dtype=np.float32)
        rates[POPULATIONS.compass // 4] = 1.0  # bump a quarter of the way round
        return rates


class _Driver:
    frame_shape = (96, 96, 3)
    brain = _Brain()

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, frame: np.ndarray) -> ControlVector:
        return ControlVector(0.5, 1.0, 0.0)


class _Env:
    frame_rate_hz = 50.0

    def reset(self, seed: int | None = None) -> tuple[None, dict]:
        return None, {}

    def step(self, action: object) -> tuple[None, float, bool, bool, dict]:
        return None, 0.0, False, False, {}


@pytest.fixture
def receiver() -> Iterator[socket.socket]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(2.0)
    yield sock
    sock.close()


def _drain(sock: socket.socket) -> list[tuple[int, FlyPose]]:
    received = []
    sock.settimeout(0.2)
    try:
        while True:
            received.append(decode_pose(sock.recv(512)))
    except TimeoutError:
        return received


def test_encode_decode_round_trip() -> None:
    pose = FlyPose(12.3456, 25.0, 0.0, -29.9999, 1.5, 0.25, -14.0, 7.125)
    seq, decoded = decode_pose(encode_pose(41, pose))
    assert seq == 41
    assert decoded.values() == pytest.approx(pose.values(), abs=1e-3)
    with pytest.raises(ValueError):
        decode_pose(b"FLY1 1 2 3")
    with pytest.raises(ValueError):
        decode_pose(encode_pose(1, pose).replace(b"FLY1", b"NOPE"))


def test_driver_wrapper_streams_one_pose_per_act(receiver: socket.socket) -> None:
    stream = PoseStream(*receiver.getsockname())
    driver = PoseStreamDriver(_Driver(), stream)
    driver.reset(seed=0)
    for _ in range(3):
        assert driver.act(np.zeros((96, 96, 3), np.uint8)) == ControlVector(0.5, 1.0, 0.0)
    received = _drain(receiver)
    stream.close()

    assert [seq for seq, _ in received] == [0, 1, 2]
    first = received[0][1]
    assert first.wheel_deg > 0 and first.throttle_press_deg > 0
    assert first.head_yaw_deg > 0  # the brain was read: bump at +90 degrees on the ring


def test_env_wrapper_streams_one_pose_per_step(receiver: socket.socket) -> None:
    stream = PoseStream(*receiver.getsockname())
    env = PoseStreamEnv(_Env(), stream)
    env.reset(seed=0)
    for _ in range(2):
        env.step(ControlVector(-1.0, 0.0, 1.0))
    received = _drain(receiver)
    stream.close()

    assert [seq for seq, _ in received] == [0, 1]
    assert received[-1][1].wheel_deg < 0 and received[-1][1].brake_press_deg > 0
    assert env.frame_rate_hz == 50.0  # everything else passes through


def test_sending_with_nobody_listening_does_not_raise() -> None:
    stream = PoseStream("127.0.0.1", 9)
    stream.publish(ControlVector.neutral())
    stream.close()
