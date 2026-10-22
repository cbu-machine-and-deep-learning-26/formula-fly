"""The cockpit fly's pose: inside the rig's limits, steering and pedals the right way round,
and a head that follows the compass bump."""

from __future__ import annotations

import math
from dataclasses import fields

import numpy as np
import pytest

from fly_driver.brains.populations import PopulationCounts
from fly_driver.cockpit.live_pose import (
    HEAD_YAW_LIMIT_DEG,
    LIMITS_DEG,
    PEDAL_PRESS_LIMIT_DEG,
    FlyPose,
    fly_pose,
    smooth_pose,
)
from fly_driver.interface import ControlVector

#: The 4,096-neuron brain's split.
POPULATIONS = PopulationCounts(compass=812, steering=3252, descending=32)


def _bump(angle: float, width: float = 0.3) -> np.ndarray:
    """Rates with a von Mises bump on the compass ring at ``angle``, the rest silent."""
    rates = np.full(POPULATIONS.total, -0.2, dtype=np.float32)
    cells = 2 * np.pi * np.arange(POPULATIONS.compass) / POPULATIONS.compass
    rates[: POPULATIONS.compass] = np.exp((np.cos(cells - angle) - 1) / width**2)
    return rates


def _inside(pose: FlyPose) -> bool:
    return all(
        LIMITS_DEG[field.name][0] <= getattr(pose, field.name) <= LIMITS_DEG[field.name][1]
        for field in fields(FlyPose)
    )


def test_every_pose_stays_inside_the_rig_limits() -> None:
    assert [field.name for field in fields(FlyPose)] == list(LIMITS_DEG)
    rng = np.random.default_rng(0)
    for steer, throttle, brake in [(-1, 1, 1), (1, 0, 0), (0.3, 0.5, 0.2)]:
        control = ControlVector(steer, throttle, brake)
        for rates in (
            np.ones(POPULATIONS.total),
            -np.ones(POPULATIONS.total),
            rng.uniform(-1, 1, POPULATIONS.total),
        ):
            assert _inside(fly_pose(control, rates, POPULATIONS))


def test_wheel_turns_with_steer_in_the_cars_sense() -> None:
    # ControlVector steer is positive to the right; the wheel angle is positive clockwise.
    assert fly_pose(ControlVector(1.0, 0, 0)).wheel_deg == LIMITS_DEG["wheel_deg"][1]
    assert fly_pose(ControlVector(-0.5, 0, 0)).wheel_deg < 0
    assert fly_pose(ControlVector.neutral()).wheel_deg == 0


def test_throttle_presses_one_foot_and_brake_the_other() -> None:
    gas = fly_pose(ControlVector(0, 1.0, 0))
    assert (gas.throttle_press_deg, gas.brake_press_deg) == (PEDAL_PRESS_LIMIT_DEG, 0.0)
    brake = fly_pose(ControlVector(0, 0, 0.5))
    assert (brake.throttle_press_deg, brake.brake_press_deg) == (0.0, PEDAL_PRESS_LIMIT_DEG / 2)


@pytest.mark.parametrize("angle", [0.0, 0.4, -0.9, math.pi / 2, 2.5])
def test_head_yaw_follows_the_compass_bump(angle: float) -> None:
    pose = fly_pose(ControlVector.neutral(), _bump(angle), POPULATIONS)
    assert pose.head_yaw_deg == pytest.approx(HEAD_YAW_LIMIT_DEG * math.sin(angle), abs=0.05)


def test_no_bump_and_no_brain_leave_the_head_still() -> None:
    silent = np.full(POPULATIONS.total, -0.5)
    assert fly_pose(ControlVector.neutral(), silent, POPULATIONS).head_yaw_deg == 0.0
    assert fly_pose(ControlVector.neutral()) == FlyPose()
    with pytest.raises(ValueError):
        fly_pose(ControlVector.neutral(), silent)
    with pytest.raises(ValueError):
        fly_pose(ControlVector.neutral(), silent[:-1], POPULATIONS)


def test_wings_and_antennae_follow_the_descending_neurons() -> None:
    rates = np.zeros(POPULATIONS.total)
    rates[-32:-16] = 0.5  # left half active, right half silent
    pose = fly_pose(ControlVector.neutral(), rates, POPULATIONS)
    assert pose.wing_left_deg > 0 and pose.wing_right_deg == 0
    assert pose.antenna_left_deg > 0 and pose.antenna_right_deg == 0


def test_smoothing_moves_part_way_and_starts_unsmoothed() -> None:
    target = FlyPose(wheel_deg=40.0, head_yaw_deg=-10.0)
    assert smooth_pose(None, target) == target
    assert smooth_pose(FlyPose(), target, alpha=0.25) == FlyPose(wheel_deg=10.0, head_yaw_deg=-2.5)
