"""The training env's remote protocol over localhost, against FakeAc."""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator

import numpy as np
import pytest

from fly_driver.bridge.env_link import RemoteCar, checked_reset, serve_env
from fly_driver.bridge.fake_ac import FakeAc
from fly_driver.bridge.fly_side import open_listener
from fly_driver.envs.assetto_corsa import AssettoCorsaTrack
from fly_driver.interface import FRAME_RATE_HZ, FRAME_SHAPE, ControlVector

HELLO = {
    "track": "ks_silverstone-gp",
    "car": "dallara_f317",
    "frame_shape": list(FRAME_SHAPE),
    "frame_rate_hz": FRAME_RATE_HZ,
}
FULL = ControlVector(0.0, 1.0, 0.0)


@pytest.fixture
def server() -> Iterator[tuple[FakeAc, int]]:
    car = FakeAc(realtime=False)
    listener = open_listener("127.0.0.1", 0)
    threading.Thread(target=serve_env, args=(car, listener, HELLO), daemon=True).start()
    yield car, listener.getsockname()[1]
    listener.close()


def test_reset_and_steps_round_trip_frames_and_telemetry(server) -> None:
    car, port = server
    remote = RemoteCar("127.0.0.1", port)
    assert remote.hello == HELLO
    frame = remote.reset()
    assert frame.shape == FRAME_SHAPE and remote.telemetry().tick == 0

    for tick in range(1, 6):
        remote.apply(FULL)
        frame, ended = remote.step()
        assert not ended
        assert remote.telemetry() == car.telemetry()
        assert remote.telemetry().tick == tick
    np.testing.assert_array_equal(frame, car._frame())
    assert car.control == FULL
    remote.close()


def test_a_reset_mid_run_restarts_the_tick_and_keeps_the_place_on_track(server) -> None:
    car, port = server
    env = AssettoCorsaTrack.connect("127.0.0.1", port)
    env.reset()
    for _ in range(30):
        env.step(FULL)
    position = env.car.telemetry().lap_dist_m
    assert position > 0

    _, info = env.reset()
    assert env.car.telemetry().tick == 0 and info["speed_mps"] == 0.0
    assert info["arclength_m"] == pytest.approx(position)
    assert car.resets == 2
    _, _, terminated, truncated, info = env.step(FULL)
    assert not (terminated or truncated) and info["missed_ticks"] == 0
    env.close()


def test_a_reset_by_another_client_ends_the_first_clients_episode(server) -> None:
    _, port = server
    training = AssettoCorsaTrack.connect("127.0.0.1", port)
    training.reset()
    training.step(FULL)

    evaluation = AssettoCorsaTrack.connect("127.0.0.1", port)
    evaluation.reset()
    evaluation.step(FULL)
    evaluation.close()

    _, _, terminated, truncated, info = training.step(FULL)
    assert truncated and not terminated and info["interrupted"]
    training.reset()
    assert not training.step(FULL)[3]
    training.close()


def test_a_hello_that_does_not_match_is_refused(server) -> None:
    _, port = server
    with pytest.raises(ValueError, match="serves"):
        AssettoCorsaTrack.connect("127.0.0.1", port, car="ks_ferrari_sf70h")


def test_disconnecting_releases_the_controls(server) -> None:
    car, port = server
    remote = RemoteCar("127.0.0.1", port)
    remote.reset()
    remote.apply(FULL)
    remote.step()
    remote.close()
    for _ in range(100):
        if car.control == ControlVector.neutral():
            break
        threading.Event().wait(0.01)
    assert car.control == ControlVector.neutral()


class StuckCar(FakeAc):
    """Comes back from every reset with all four tyres off the road."""

    def reset(self):
        frame = super().reset()
        self.lateral_m = 20.0
        return frame


def test_checked_reset_retries_then_gives_up_on_a_car_that_stays_off() -> None:
    car = StuckCar(realtime=False)
    with pytest.raises(RuntimeError, match="after 3 resets"):
        checked_reset(car)
    assert car.resets == 3


def test_a_failing_car_closes_the_client_instead_of_answering() -> None:
    listener = open_listener("127.0.0.1", 0)
    car = StuckCar(realtime=False)
    threading.Thread(target=serve_env, args=(car, listener, HELLO), daemon=True).start()
    remote = RemoteCar("127.0.0.1", listener.getsockname()[1], timeout_s=5.0)
    with pytest.raises((ConnectionError, socket.timeout)):
        remote.reset()
    listener.close()
