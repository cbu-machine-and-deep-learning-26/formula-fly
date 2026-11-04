"""Steering wobble on a straight (GH-104)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fly_driver.analysis.steering_wobble import (
    drive_straight,
    longest_straights,
    recorded_wobble,
    square_wobble,
)
from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.envs.segments import Segment, find_segments


def _segment(index: int, length_m: float, *, is_corner: bool = False) -> Segment:
    radius = 50.0 if is_corner else math.inf
    return Segment(index, f"S{index}", is_corner, None, 0.0, length_m, radius)


def test_square_wobble_flips_on_the_hand_computed_frames() -> None:
    wobble = square_wobble(2, 0.5)
    assert [wobble(frame) for frame in range(6)] == [0.5, 0.5, -0.5, -0.5, 0.5, 0.5]


def test_square_wobble_refuses_a_flip_below_one_frame() -> None:
    with pytest.raises(ValueError, match="flip_frames"):
        square_wobble(0, 1.0)


def test_recorded_wobble_loops_from_its_offset_and_applies_its_gain() -> None:
    wobble = recorded_wobble(np.array([1.0, -1.0, 0.5]), offset=1, gain=0.5)
    assert [wobble(frame) for frame in range(4)] == [-0.5, 0.25, 0.5, -0.5]


def test_recorded_wobble_refuses_an_empty_recording() -> None:
    with pytest.raises(ValueError, match="empty"):
        recorded_wobble(np.array([]))


def test_longest_straights_ignores_corners_and_sorts_longest_first() -> None:
    segments = [
        _segment(0, 100.0),
        _segment(1, 900.0, is_corner=True),
        _segment(2, 300.0),
        _segment(3, 200.0),
    ]
    assert [item.index for item in longest_straights(segments, 2)] == [2, 3]


@pytest.mark.slow
def test_a_hard_wobble_stops_a_car_that_is_fast_without_it() -> None:
    env = PracticeTrack(random_start=True, max_steps=4000)
    try:
        try:
            env.reset(seed=0)
        except Exception as exc:  # pragma: no cover - depends on the machine
            pytest.skip(f"no offscreen GL context: {exc}")
        straight = longest_straights(find_segments(env.centerline), 1)[0]
        clean = drive_straight(env, straight, lambda frame: 0.0)
        wobbling = drive_straight(env, straight, square_wobble(20, 1.0))
        assert not clean.crashed
        assert clean.end_speed_kmh > 280.0
        assert wobbling.crashed
        assert wobbling.end_speed_kmh < clean.end_speed_kmh
    finally:
        env.close()
