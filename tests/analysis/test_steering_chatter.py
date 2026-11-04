"""Steering chatter measurement (GH-104), checked against hand-computed values."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from fly_driver.analysis.steering_chatter import (
    DriveTrace,
    SmoothedSteering,
    SteeringStats,
    combine_stats,
    count_flips,
    drive,
    steering_stats,
    straight_exit_speeds,
    straight_mask,
    trace_stats,
    zone_outcome,
)
from fly_driver.envs.segments import Segment
from fly_driver.interface import ControlVector


def _segment(index: int, start_m: float, end_m: float, *, is_corner: bool) -> Segment:
    return Segment(
        index=index,
        name=f"S{index}",
        is_corner=is_corner,
        turns_left=True if is_corner else None,
        start_m=start_m,
        end_m=end_m,
        min_radius_m=50.0 if is_corner else math.inf,
    )


SEGMENTS = (
    _segment(0, 0.0, 100.0, is_corner=False),
    _segment(1, 100.0, 200.0, is_corner=True),
)


@pytest.mark.parametrize(
    ("steer", "flips"),
    [
        ([], 0),
        ([0.5], 0),
        ([0.5, 0.5, 0.9], 0),
        ([1.0, -1.0], 1),
        ([1.0, -1.0, 1.0, -1.0], 3),
        ([1.0, 0.0, -1.0], 1),  # passing through centre is still one flip
        ([1.0, 0.0, 1.0], 0),  # back to the same side is none
        ([0.01, -0.01, 0.01, -0.01], 0),  # all inside the deadband
        ([0.0, 0.0], 0),
        ([0.06, -0.06], 1),  # just outside the deadband on each side
    ],
)
def test_count_flips_on_hand_computed_series(steer: list[float], flips: int) -> None:
    assert count_flips(steer) == flips


def test_count_flips_honours_the_deadband_argument() -> None:
    assert count_flips([0.2, -0.2], deadband=0.3) == 0
    assert count_flips([0.2, -0.2], deadband=0.1) == 1


def test_count_flips_rejects_a_negative_deadband() -> None:
    with pytest.raises(ValueError, match="deadband"):
        count_flips([1.0], deadband=-0.1)


def test_steering_stats_on_a_hand_computed_series() -> None:
    # 5 frames at 50 Hz is 0.1 s. Flips: +1 -> -1 -> +0.5 is 2, and the 0.0 keeps its side.
    stats = steering_stats([1.0, -1.0, 0.5, 0.0, 0.5], 50.0)
    assert stats.frames == 5
    assert stats.seconds == pytest.approx(0.1)
    assert stats.flips == 2
    assert stats.flips_per_s == pytest.approx(20.0)
    assert stats.mean_abs_steer == pytest.approx(3.0 / 5.0)  # 1 + 1 + 0.5 + 0 + 0.5
    assert stats.full_lock_share == pytest.approx(2.0 / 5.0)
    assert stats.mean_step == pytest.approx((2.0 + 1.5 + 0.5 + 0.5) / 4.0)


def test_a_change_of_steering_across_a_gap_is_not_a_flip_or_a_step() -> None:
    steer = [1.0, 1.0, -1.0, -1.0, -1.0]
    keep = np.array([True, True, False, True, True])
    stats = steering_stats(steer, 50.0, keep=keep)
    assert stats.frames == 4
    assert stats.flips == 0  # +1, +1 | gap | -1, -1
    assert stats.step_sum == 0.0
    assert stats.step_count == 2


def test_empty_stats_are_nan_rather_than_a_zero_that_looks_calm() -> None:
    stats = steering_stats([1.0], 50.0, keep=np.array([False]))
    assert stats == SteeringStats()
    assert math.isnan(stats.flips_per_s)
    assert math.isnan(stats.mean_abs_steer)
    assert math.isnan(stats.full_lock_share)
    assert math.isnan(stats.mean_step)


def test_steering_stats_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="frame_rate_hz"):
        steering_stats([1.0], 0.0)
    with pytest.raises(ValueError, match="keep"):
        steering_stats([1.0, 1.0], 50.0, keep=np.array([True]))


def test_combine_stats_pools_counts_not_averages() -> None:
    long_calm = steering_stats([0.5] * 100, 50.0)  # 2 s, no flips
    short_chatter = steering_stats([1.0, -1.0], 50.0)  # 0.04 s, 1 flip
    pooled = combine_stats([long_calm, short_chatter])
    assert pooled.frames == 102
    assert pooled.flips == 1
    assert pooled.flips_per_s == pytest.approx(1.0 / 2.04)
    assert pooled.mean_abs_steer == pytest.approx((50.0 + 2.0) / 102.0)


def test_straight_mask_uses_the_segments_half_open_ranges() -> None:
    positions = [0.0, 99.9, 100.0, 199.9, 200.0]
    assert straight_mask(positions, SEGMENTS).tolist() == [True, True, False, False, False]


class _FakeEnv:
    """Moves 10 m per step from 90 m and ends at a scripted step, with a chosen last info."""

    def __init__(self, steps: int, last_info: dict[str, Any], *, truncate: bool = False) -> None:
        self.steps = steps
        self.last_info = last_info
        self.truncate = truncate
        self.reset_seeds: list[int | None] = []
        self.position = 0.0
        self.taken = 0
        self.received: list[ControlVector] = []

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.reset_seeds.append(seed)
        self.position = 90.0
        self.taken = 0
        return np.zeros((2, 2, 3), dtype=np.uint8), {"arclength_m": self.position}

    def step(self, action: ControlVector) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.received.append(action)
        self.taken += 1
        self.position += 10.0
        info: dict[str, Any] = {
            "progress_m": 10.0,
            "speed_mps": 20.0 + self.taken,
            "arclength_m": self.position,
            "lateral_m": 0.5,
        }
        done = self.taken >= self.steps
        if done:
            info.update(self.last_info)
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        return frame, 0.0, done and not self.truncate, done and self.truncate, info


class _ScriptedAgent:
    def __init__(self, steer: list[float]) -> None:
        self.steer = steer
        self.index = 0
        self.reset_seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)
        self.index = 0

    def act(self, frame: np.ndarray) -> ControlVector:
        del frame
        steer = self.steer[self.index]
        self.index += 1
        return ControlVector(steer=steer, throttle=0.25, brake=0.75)


def test_drive_records_every_frame_and_passes_the_seed_to_both_resets() -> None:
    env = _FakeEnv(3, {"crashed": True})
    agent = _ScriptedAgent([1.0, -1.0, 0.5])
    trace = drive(env, agent, seed=7)
    assert env.reset_seeds == [7]
    assert agent.reset_seeds == [7]
    assert trace.seed == 7
    assert trace.start_m == 90.0
    assert trace.end_m == 120.0
    assert trace.distance_m == 30.0
    assert trace.outcome == "crashed"
    assert trace.steer.tolist() == [1.0, -1.0, 0.5]
    assert trace.throttle.tolist() == [0.25, 0.25, 0.25]
    assert trace.brake.tolist() == [0.75, 0.75, 0.75]
    assert trace.speed_mps.tolist() == [21.0, 22.0, 23.0]
    assert trace.arclength_m.tolist() == [100.0, 110.0, 120.0]
    assert trace.lateral_m.tolist() == [0.5, 0.5, 0.5]
    assert np.isnan(trace.kingpin_rad).all()  # the fake env has no MuJoCo car


@pytest.mark.parametrize(
    ("last_info", "truncate", "outcome"),
    [
        ({"crashed": True}, False, "crashed"),
        ({"stalled": True}, False, "stalled"),
        ({"diverged": True, "crashed": True}, False, "diverged"),
        ({"lap_complete": True}, False, "lap"),
        ({}, True, "truncated"),
    ],
)
def test_drive_reports_why_it_ended(
    last_info: dict[str, Any], truncate: bool, outcome: str
) -> None:
    trace = drive(_FakeEnv(1, last_info, truncate=truncate), _ScriptedAgent([0.0]), seed=0)
    assert trace.outcome == outcome


def test_drive_stops_at_max_steps_and_says_so() -> None:
    env = _FakeEnv(100, {})
    trace = drive(env, _ScriptedAgent([0.0] * 100), seed=0, max_steps=2)
    assert trace.steer.size == 2
    assert env.taken == 2
    assert trace.outcome == "capped"


def test_drive_rejects_a_max_steps_below_one() -> None:
    with pytest.raises(ValueError, match="max_steps"):
        drive(_FakeEnv(1, {}), _ScriptedAgent([0.0]), seed=0, max_steps=0)


def test_trace_stats_splits_flips_between_straight_and_corner() -> None:
    # The fake env starts at 90 m and moves 10 m per step, so frames land at 100-130 m: corner.
    agent = _ScriptedAgent([1.0, -1.0, 1.0, -1.0])
    trace = drive(_FakeEnv(4, {"crashed": True}), agent, seed=0)
    stats = trace_stats(trace, 50.0, SEGMENTS)
    assert stats["all"].flips == 3
    assert stats["corner"].flips == 3
    assert stats["straight"].frames == 0


def test_trace_stats_counts_only_the_straight_frames_as_straight() -> None:
    # A segment boundary at 115 m puts 100 and 110 m on the straight, 120 and 130 m in the corner.
    segments = (
        _segment(0, 0.0, 115.0, is_corner=False),
        _segment(1, 115.0, 200.0, is_corner=True),
    )
    agent = _ScriptedAgent([1.0, -1.0, -1.0, 1.0])
    trace = drive(_FakeEnv(4, {"crashed": True}), agent, seed=0)
    stats = trace_stats(trace, 50.0, segments)
    assert (stats["straight"].frames, stats["straight"].flips) == (2, 1)
    assert (stats["corner"].frames, stats["corner"].flips) == (2, 1)
    assert stats["all"].flips == 2  # the -1, -1 across the boundary is not a flip either way


def test_smoothing_follows_the_hand_computed_filter() -> None:
    smoothed = SmoothedSteering(_ScriptedAgent([1.0, 1.0, 1.0, -1.0]), blend=0.5)
    smoothed.reset(seed=0)
    steers = [smoothed.act(None).steer for _ in range(4)]
    # Starts centred: 0.5, 0.75, 0.875, then 0.875 + 0.5 * (-1 - 0.875) = -0.0625.
    assert steers == pytest.approx([0.5, 0.75, 0.875, -0.0625])


def test_a_blend_of_one_changes_nothing() -> None:
    script = [1.0, -1.0, 0.25]
    smoothed = SmoothedSteering(_ScriptedAgent(script), blend=1.0)
    smoothed.reset()
    assert [smoothed.act(None).steer for _ in script] == script


def test_smoothing_leaves_throttle_and_brake_alone() -> None:
    smoothed = SmoothedSteering(_ScriptedAgent([1.0]), blend=0.1)
    smoothed.reset()
    control = smoothed.act(None)
    assert (control.throttle, control.brake) == (0.25, 0.75)


def test_smoothing_shrinks_a_full_lock_chatter() -> None:
    script = [1.0, -1.0] * 50
    smoothed = SmoothedSteering(_ScriptedAgent(script), blend=0.2)
    smoothed.reset()
    filtered = [smoothed.act(None).steer for _ in script]
    # Settled, it swings between +-1/9: x = 0.2 - 0.8x gives x = 0.2 / 1.8.
    assert abs(filtered[-1]) == pytest.approx(1.0 / 9.0, abs=1e-6)
    assert steering_stats(filtered[-10:], 50.0).mean_abs_steer == pytest.approx(1.0 / 9.0, abs=1e-6)


def test_reset_recentres_the_filter_and_passes_the_seed_on() -> None:
    agent = _ScriptedAgent([1.0, 1.0])
    smoothed = SmoothedSteering(agent, blend=0.5)
    smoothed.reset(seed=3)
    assert smoothed.act(None).steer == pytest.approx(0.5)
    smoothed.reset(seed=4)
    assert smoothed.act(None).steer == pytest.approx(0.5)
    assert agent.reset_seeds == [3, 4]


@pytest.mark.parametrize("blend", [0.0, -0.1, 1.5])
def test_a_blend_outside_its_range_is_refused(blend: float) -> None:
    with pytest.raises(ValueError, match="blend"):
        SmoothedSteering(_ScriptedAgent([0.0]), blend)


def _trace_at(arclengths: list[float], speeds: list[float] | None = None) -> DriveTrace:
    size = len(arclengths)
    return DriveTrace(
        seed=0,
        start_m=arclengths[0] if arclengths else 0.0,
        distance_m=0.0,
        outcome="crashed",
        steer=np.zeros(size),
        throttle=np.zeros(size),
        brake=np.zeros(size),
        kingpin_rad=np.zeros(size),
        speed_mps=np.array(speeds if speeds is not None else [0.0] * size),
        arclength_m=np.array(arclengths, dtype=np.float64),
        lateral_m=np.zeros(size),
    )


@pytest.mark.parametrize(
    ("arclengths", "outcome"),
    [
        ([10.0, 20.0, 30.0], "not_reached"),
        ([90.0, 110.0, 150.0], "ended_in_zone"),
        ([90.0, 110.0, 150.0, 210.0], "passed"),
        ([90.0, 190.0, 5.0, 8.0], "passed"),  # left by crossing the start line
        ([150.0, 210.0], "started_in_zone"),  # never made the approach, so not a pass
        ([150.0, 160.0], "started_in_zone"),
    ],
)
def test_zone_outcome(arclengths: list[float], outcome: str) -> None:
    assert zone_outcome(_trace_at(arclengths), 100.0, 200.0) == outcome


def test_exit_speed_is_the_last_frame_on_the_straight() -> None:
    trace = _trace_at([50.0, 99.0, 105.0], [10.0, 20.0, 30.0])
    assert straight_exit_speeds(trace, SEGMENTS) == {"S0": [20.0]}


def test_a_straight_the_drive_ended_on_has_no_exit() -> None:
    trace = _trace_at([50.0, 60.0], [10.0, 20.0])
    assert straight_exit_speeds(trace, SEGMENTS) == {}


def test_a_straight_driven_twice_has_two_exits() -> None:
    trace = _trace_at([50.0, 105.0, 50.0, 60.0, 105.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    assert straight_exit_speeds(trace, SEGMENTS) == {"S0": [1.0, 4.0]}
