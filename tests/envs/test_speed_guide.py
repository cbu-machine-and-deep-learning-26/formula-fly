"""The target speed behind the braking reward (GH-104)."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.braking_line import speed_profile
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.envs.speed_guide import SpeedGuide, SpeedTarget
from fly_driver.interface import ControlVector

GUIDE = SpeedGuide(start_factor=0.6, end_factor=1.0, ramp_steps=1000)
BASE_REWARD = {"type": "lap_time", "crash_penalty_m": 18.0}
FLAT_OUT = ControlVector(steer=0.0, throttle=1.0, brake=0.0)


@pytest.mark.parametrize(
    ("steps", "factor"),
    [(0, 0.6), (250, 0.7), (500, 0.8), (1000, 1.0), (5000, 1.0), (-10, 0.6)],
)
def test_the_factor_on_hand_computed_steps(steps: int, factor: float) -> None:
    assert GUIDE.factor(steps) == pytest.approx(factor)


def test_a_zero_length_ramp_uses_the_end_factor_at_once() -> None:
    assert SpeedGuide(0.6, 0.9, 0).factor(0) == 0.9


@pytest.mark.parametrize(
    ("start", "end", "ramp"), [(0.0, 1.0, 10), (0.6, -1.0, 10), (0.6, 1.0, -1)]
)
def test_bad_settings_are_refused(start: float, end: float, ramp: int) -> None:
    with pytest.raises(ValueError):
        SpeedGuide(start, end, ramp)


@pytest.fixture(scope="module")
def centerline() -> Centerline:
    return Centerline.load()


def test_the_target_is_the_factor_times_the_profile(centerline: Centerline) -> None:
    profile = speed_profile(centerline)
    target = SpeedTarget(GUIDE, centerline)
    arclength = float(profile.arclength_m[100])
    assert target.speed_mps(arclength) == pytest.approx(0.6 * float(profile.speed_mps[100]))
    target.set_training_steps(1000)
    assert target.speed_mps(arclength) == pytest.approx(float(profile.speed_mps[100]))


def test_the_target_is_low_for_a_corner_and_high_on_a_straight(centerline: Centerline) -> None:
    target = SpeedTarget(SpeedGuide(1.0, 1.0, 0), centerline)
    # Vale's apex against the long straight before Stowe.
    assert target.speed_mps(5510.0) < 0.5 * target.speed_mps(4600.0)


def test_the_target_wraps_across_the_start_line(centerline: Centerline) -> None:
    target = SpeedTarget(SpeedGuide(1.0, 1.0, 0), centerline)
    assert target.speed_mps(1.0) == pytest.approx(target.speed_mps(centerline.length + 1.0))
    near_end = target.speed_mps(centerline.length - 0.5)
    near_start = target.speed_mps(0.5)
    assert abs(near_end - near_start) < 2.0


def test_over_speed_is_how_far_above_the_target(centerline: Centerline) -> None:
    target = SpeedTarget(SpeedGuide(1.0, 1.0, 0), centerline)
    limit = target.speed_mps(1000.0)
    assert target.over_speed_mps(1000.0, limit + 5.0) == pytest.approx(5.0)
    assert target.over_speed_mps(1000.0, limit - 5.0) == 0.0


def test_no_target_on_the_circuit_is_too_slow_to_avoid_the_stall_rule(
    centerline: Centerline,
) -> None:
    """At its lowest factor, the target must stay above the 30 km/h stall limit everywhere, or
    obeying it would end the drive."""
    target = SpeedTarget(SpeedGuide(0.6, 0.6, 0), centerline)
    lowest = min(target.speed_mps(float(s)) for s in np.arange(0.0, centerline.length, 4.0))
    assert lowest > 8.3


def _track(**kwargs) -> PracticeTrack:
    env = PracticeTrack(terminate_off_track=False, reward=dict(BASE_REWARD), **kwargs)
    try:
        env.reset(seed=0)
    except Exception as exc:  # pragma: no cover - depends on the machine
        env.close()
        pytest.skip(f"no offscreen GL context: {exc}")
    return env


def test_without_a_guide_the_env_reports_nothing() -> None:
    env = _track()
    try:
        assert "over_speed_mps" not in env.step(FLAT_OUT)[4]
    finally:
        env.close()


def test_a_guide_can_come_from_a_config_mapping() -> None:
    env = _track(speed_guide={"start_factor": 0.6, "end_factor": 1.0, "ramp_steps": 10})
    try:
        assert env.step(FLAT_OUT)[4]["over_speed_mps"] >= 0.0
    finally:
        env.close()


def test_the_report_follows_training_progress() -> None:
    env = _track(speed_guide=SpeedGuide(0.01, 1.0, 1000))
    try:
        for _ in range(100):
            info = env.step(FLAT_OUT)[4]
        low_factor = info["over_speed_mps"]
        assert low_factor > 0.0
        env.set_training_steps(1000)
        assert env.step(FLAT_OUT)[4]["over_speed_mps"] < low_factor
    finally:
        env.close()


def test_the_cost_reaches_the_reward_terms_and_they_add_up() -> None:
    env = PracticeTrack(
        terminate_off_track=False,
        reward={**BASE_REWARD, "over_speed_m": 1.0},
        speed_guide=SpeedGuide(0.01, 0.01, 0),
    )
    try:
        try:
            env.reset(seed=0)
        except Exception as exc:  # pragma: no cover - depends on the machine
            pytest.skip(f"no offscreen GL context: {exc}")
        for _ in range(100):
            _, reward, _, _, info = env.step(FLAT_OUT)
        assert info["reward_terms"]["over_speed"] < 0.0  # still gaining speed over the target
        assert sum(info["reward_terms"].values()) == pytest.approx(reward)
    finally:
        env.close()


def test_a_guide_does_not_change_the_physics() -> None:
    positions = []
    for guide in (None, GUIDE):
        env = _track(speed_guide=guide)
        try:
            for step in range(50):
                env.step(ControlVector(steer=0.5 if step % 2 else -0.5, throttle=1.0, brake=0.0))
            positions.append(env.data.qpos.copy())
        finally:
            env.close()
    assert np.array_equal(positions[0], positions[1])


def test_a_training_only_guide_is_dropped_from_evaluation_envs() -> None:
    pytest.importorskip("torch")
    from fly_driver.training import EnvConfig
    from fly_driver.training.train import build_env

    config = EnvConfig(
        type="practice_track",
        max_steps=100,
        params={
            "reward": {**BASE_REWARD, "over_speed_m": 1.0},
            "speed_guide": {"start_factor": 0.6, "end_factor": 1.0, "ramp_steps": 10},
        },
    )
    training, evaluation = build_env(config, training=True), build_env(config)
    try:
        assert training._speed_target is not None
        assert evaluation._speed_target is None
    finally:
        training.close()
        evaluation.close()
