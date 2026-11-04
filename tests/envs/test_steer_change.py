"""The env's report of how far the steering moved, and the optional cost on it (GH-104)."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.interface import ControlVector

STEER_COST_M = 0.25
BASE_REWARD = {"type": "lap_time", "crash_penalty_m": 18.0}


def _control(steer: float) -> ControlVector:
    return ControlVector(steer=steer, throttle=1.0, brake=0.0)


def _reset(env: PracticeTrack) -> None:
    """``env.reset()``, or skip the test if this machine cannot render offscreen."""
    try:
        env.reset(seed=0)
    except Exception as exc:  # pragma: no cover - depends on the machine
        env.close()
        pytest.skip(f"no offscreen GL context: {exc}")


@pytest.fixture
def env():
    track = PracticeTrack(terminate_off_track=False, reward=dict(BASE_REWARD))
    _reset(track)
    yield track
    track.close()


def test_steer_change_is_the_distance_moved_since_the_last_frame(env: PracticeTrack) -> None:
    changes = [env.step(_control(steer))[4]["steer_change"] for steer in (1.0, -1.0, -0.5, -0.5)]
    # The car starts centred: 1 from centre, 2 lock to lock, 0.5, then held.
    assert changes == pytest.approx([1.0, 2.0, 0.5, 0.0])


def test_reset_recentres_the_steering_it_measures_from(env: PracticeTrack) -> None:
    env.step(_control(1.0))
    env.reset(seed=0)
    assert env.step(_control(0.25))[4]["steer_change"] == pytest.approx(0.25)


def test_with_the_cost_off_there_is_no_steering_term(env: PracticeTrack) -> None:
    assert "steering" not in env.step(_control(1.0))[4]["reward_terms"]


def test_the_cost_reaches_the_terms_and_they_still_add_up() -> None:
    track = PracticeTrack(
        terminate_off_track=False, reward={**BASE_REWARD, "steer_change_m": STEER_COST_M}
    )
    _reset(track)
    try:
        for _ in range(50):  # a run-up, so this small change stays under the cap on the cost
            track.step(_control(0.0))
        _, reward, _, _, info = track.step(_control(0.1))
        terms = info["reward_terms"]
        assert terms["steering"] == pytest.approx(-STEER_COST_M * 0.1)
        assert sum(terms.values()) == pytest.approx(reward)
    finally:
        track.close()


def test_the_cost_does_not_change_the_physics() -> None:
    """Same actions with and without the cost: same car, only the reward differs."""
    positions = []
    for reward in (dict(BASE_REWARD), {**BASE_REWARD, "steer_change_m": STEER_COST_M}):
        track = PracticeTrack(terminate_off_track=False, reward=reward)
        _reset(track)
        try:
            for step in range(20):
                track.step(_control(1.0 if step % 2 else -1.0))
            positions.append(track.data.qpos.copy())
        finally:
            track.close()
    assert np.array_equal(positions[0], positions[1])
