"""Reward terms are logged; standing still cannot farm return."""

from __future__ import annotations

from fly_driver.envs.dummy import DummyRaceEnv
from fly_driver.interface import ControlVector
from fly_driver.training.reward import RewardTerms, farms_without_progress, shaped_reward


def test_idle_bonus_cannot_exceed_idle_cap() -> None:
    idle = RewardTerms(progress=0.0, on_track=0.01, alive=0.0, collision=0.0)
    assert not farms_without_progress(idle)
    greedy = RewardTerms(progress=0.0, on_track=0.0, alive=1.0, collision=0.0)
    assert farms_without_progress(greedy)
    assert shaped_reward(idle) == 0.01


def test_dummy_env_zero_throttle_does_not_accumulate_progress() -> None:
    env = DummyRaceEnv(seed=0, max_steps=20)
    env.reset(seed=0)
    total = 0.0
    progressed = 0.0
    idle = ControlVector(steer=0.0, throttle=0.0, brake=0.0)
    for _ in range(20):
        step = env.step(idle)
        total += step.reward
        progressed += step.terms.progress
        assert not farms_without_progress(step.terms)
    assert progressed == 0.0
    assert total < 0.5


def test_dummy_env_throttle_increases_progress() -> None:
    env = DummyRaceEnv(seed=0, max_steps=20)
    env.reset(seed=0)
    go = ControlVector(steer=0.0, throttle=1.0, brake=0.0)
    progressed = 0.0
    for _ in range(10):
        step = env.step(go)
        progressed += step.terms.progress
        for key, value in step.terms.as_dict().items():
            assert key.startswith("reward/")
            assert isinstance(value, float)
    assert progressed > 0.0
