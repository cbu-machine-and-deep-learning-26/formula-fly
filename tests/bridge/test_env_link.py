"""The training env's remote protocol over localhost, against FakeAc."""

from __future__ import annotations

import socket
import threading
from collections.abc import Iterator

import numpy as np
import pytest

from fly_driver.bridge.env_link import (
    FIRST_GEAR,
    GEAR_STATS,
    RemoteCar,
    checked_reset,
    serve_env,
)
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


def _drop(remote: RemoteCar) -> None:
    """Swap in a connection whose far end has gone, as when the server process dies."""
    dead, far = socket.socketpair()
    far.close()
    remote._sock.close()
    remote._sock = dead


def test_a_dropped_server_ends_the_episode_and_the_next_reset_carries_on(server) -> None:
    """One server restart must not end an hours-long real-time run (GH-29)."""
    _, port = server
    remote = RemoteCar("127.0.0.1", port, timeout_s=5.0, reconnect_s=5.0)
    remote.reset()
    remote.apply(FULL)
    last, ended = remote.step()
    assert not ended
    _drop(remote)
    frame, ended = remote.step()
    assert ended and remote.reconnects == 1
    np.testing.assert_array_equal(frame, last)
    remote.reset()
    assert not remote.step()[1]
    remote.close()


def test_a_server_that_does_not_come_back_raises(server) -> None:
    _, port = server
    remote = RemoteCar("127.0.0.1", port, timeout_s=5.0, reconnect_s=0.3)
    remote.reset()
    # A port nothing listens on. Closing a listener whose serve_env thread is blocked in
    # accept() does not stop it accepting on Linux, so the server's own port won't do.
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    remote._address = probe.getsockname()
    probe.close()
    _drop(remote)
    with pytest.raises(ConnectionError, match="not back"):
        remote.step()


def test_a_reset_leaves_the_car_in_neutral_and_the_envs_reset_puts_it_in_first(server) -> None:
    """AC's reset leaves the SF70H in neutral, where 0.15 throttle never launches it."""
    car, port = server
    car.reset()
    assert car.gear() < FIRST_GEAR
    env = AssettoCorsaTrack.connect("127.0.0.1", port)
    env.reset()
    assert car.gear() >= FIRST_GEAR
    for _ in range(20):
        env.step(ControlVector(0.0, 0.15, 0.0))
    assert car.telemetry().speed_mps > 0.0
    env.close()


class NoShiftCar(FakeAc):
    """A gearbox that ignores gear-up."""

    def shift_up(self) -> None:
        pass


def test_a_car_that_will_not_shift_is_logged_not_failed(caplog) -> None:
    car = NoShiftCar(realtime=False)
    before = dict(GEAR_STATS)
    frame = checked_reset(car)
    assert frame.shape == FRAME_SHAPE and car.gear() < FIRST_GEAR
    assert GEAR_STATS["gear_ups"] == before["gear_ups"] + 2
    assert GEAR_STATS["failures"] == before["failures"] + 1
    assert "after two gear-ups" in caplog.text
