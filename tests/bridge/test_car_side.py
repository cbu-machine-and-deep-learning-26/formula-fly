"""The car side against a fake AC env and a real fly side on a localhost socket."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import numpy as np
import pytest

from fly_driver.bridge.car_side import FlyLink, run_episode
from fly_driver.bridge.fly_side import open_listener, serve
from fly_driver.interface import ControlVector, Frame

SHAPE = (4, 6, 3)


class FakeCar:
    """Frame ``i`` is filled with ``i``; records every control it is given."""

    def __init__(self, ticks: int) -> None:
        self.ticks = ticks
        self.tick = 0
        self.applied: list[ControlVector] = []

    def reset(self) -> Frame:
        self.tick = 0
        return np.zeros(SHAPE, np.uint8)

    def apply(self, control: ControlVector) -> None:
        self.applied.append(control)

    def step(self) -> tuple[Frame, bool]:
        self.tick += 1
        return np.full(SHAPE, self.tick, np.uint8), self.tick >= self.ticks


class ScriptedDriver:
    """Steers by the frame's fill value; sleeps ``delay_s`` on the first frame."""

    frame_shape = SHAPE

    def __init__(self, delay_s: float = 0.0, scale: float = 0.1) -> None:
        self.delay_s = delay_s
        self.scale = scale
        self.seen: list[int] = []
        self.resets = 0

    def reset(self, seed: int | None = None) -> None:
        self.resets += 1

    def act(self, frame: Frame) -> np.ndarray:
        if not self.seen:
            time.sleep(self.delay_s)
        self.seen.append(int(frame[0, 0, 0]))
        # A raw array, some of it out of range: the fly side must clip, not reject.
        return np.array([self.scale * frame[0, 0, 0], 1.5, -0.5])


@pytest.fixture
def connect() -> Iterator:
    links: list[FlyLink] = []
    threads: list[threading.Thread] = []

    def _connect(driver: ScriptedDriver, deadline_s: float) -> FlyLink:
        listener = open_listener("127.0.0.1", 0)
        thread = threading.Thread(target=lambda: serve(driver, listener), daemon=True)
        thread.start()
        threads.append(thread)
        link = FlyLink("127.0.0.1", listener.getsockname()[1], deadline_s)
        links.append(link)
        return link

    yield _connect
    for link in links:
        link.close()
    for thread in threads:
        thread.join(timeout=2)


def test_round_trip_applies_each_frames_clipped_control(connect) -> None:
    driver = ScriptedDriver()
    car = FakeCar(ticks=10)
    stats = run_episode(car, connect(driver, deadline_s=1.0), SHAPE, max_ticks=100)

    assert driver.seen == list(range(10))
    assert driver.resets == 1
    expected = [ControlVector.clipped(0.1 * tick, 1.5, -0.5) for tick in range(10)]
    assert car.applied == expected
    assert stats.ticks == 10 and stats.misses == 0
    assert all(rtt is not None and rtt > 0 for rtt in stats.rtt_ms)


def test_missed_deadline_repeats_last_control_without_blocking(connect) -> None:
    deadline_s = 0.02
    driver = ScriptedDriver(delay_s=0.3)
    link = connect(driver, deadline_s)
    car = FakeCar(ticks=40)

    for tick in range(40):
        tick_start = time.perf_counter()
        car.apply(link.exchange(np.full(SHAPE, tick, np.uint8), is_reset=tick == 0))
        assert time.perf_counter() - tick_start < deadline_s + 0.05

    # The slow first answer arrives about 15 ticks late; every tick before it coasts.
    late = next(tick for tick, rtt in enumerate(link.stats.rtt_ms) if rtt is not None)
    assert late >= 5 and link.stats.misses >= late
    assert car.applied[:late] == [ControlVector.neutral()] * late
    assert car.applied[late] == ControlVector.clipped(0.0, 1.0, 0.0)
    # Frames offered while one was in flight were dropped, not queued.
    assert len(driver.seen) < 40 and driver.seen[1] == late + 1
