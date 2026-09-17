"""Contract and dynamics tests for the numpy-only dummy track."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs import DEFAULT_DUMMY_FRAME_SHAPE, DummyTrackEnv


def _steer_to_centre(env: DummyTrackEnv, gain: float = 2.0) -> np.ndarray:
    lateral = env._lateral
    return np.array([np.clip(-gain * lateral, -1.0, 1.0), 1.0, 0.0], dtype=np.float32)


def test_reset_and_step_follow_the_frame_contract() -> None:
    """Observations are uint8 RGB frames of the declared shape."""
    env = DummyTrackEnv()
    frame, info = env.reset(seed=0)

    assert frame.shape == DEFAULT_DUMMY_FRAME_SHAPE
    assert frame.dtype == np.uint8
    assert info["lap_complete"] is False and info["lap_time"] is None
    next_frame, reward, terminated, truncated, info = env.step(
        np.zeros(3, dtype=np.float32)
    )
    assert (
        next_frame.shape == DEFAULT_DUMMY_FRAME_SHAPE and next_frame.dtype == np.uint8
    )
    assert isinstance(reward, float)
    assert (terminated, truncated) == (False, False)
    assert env.render().shape == DEFAULT_DUMMY_FRAME_SHAPE


def test_same_seed_and_actions_give_identical_episodes() -> None:
    """The wind is the only randomness and it is seeded."""
    rng = np.random.default_rng(3)
    actions = [
        np.array([rng.uniform(-1, 1), rng.uniform(0, 1), 0.0], dtype=np.float32)
        for _ in range(50)
    ]

    def rollout(seed: int) -> list[tuple[bytes, float]]:
        env = DummyTrackEnv()
        env.reset(seed=seed)
        out = []
        for action in actions:
            frame, reward, terminated, truncated, _ = env.step(action)
            out.append((frame.tobytes(), reward))
            if terminated or truncated:
                break
        return out

    assert rollout(7) == rollout(7)
    assert rollout(7) != rollout(8)


@pytest.mark.parametrize(
    "action",
    [
        np.array([1.5, 0.0, 0.0]),
        np.array([0.0, -0.1, 0.0]),
        np.array([0.0, 0.0, 2.0]),
        np.array([np.nan, 0.0, 0.0]),
        np.zeros(2),
    ],
)
def test_out_of_range_or_malformed_actions_raise(action: np.ndarray) -> None:
    """No silent clipping or reshaping at the env boundary."""
    env = DummyTrackEnv()
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(action)


def test_accepts_objects_with_to_array() -> None:
    """A ControlVector-style object is accepted through ``to_array``."""

    class Control:
        def to_array(self) -> np.ndarray:
            return np.array([0.0, 0.5, 0.0], dtype=np.float32)

    env = DummyTrackEnv()
    env.reset(seed=0)
    _, _, _, _, info = env.step(Control())
    assert info["speed"] > 0


def test_proportional_steering_completes_a_lap() -> None:
    """The task is solvable: steer toward the centreline and floor it."""
    env = DummyTrackEnv()
    env.reset(seed=0)
    terminated = truncated = False
    info: dict[str, object] = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(_steer_to_centre(env))

    assert info["lap_complete"] is True
    assert info["off_track"] is False
    assert (
        isinstance(info["lap_time"], float)
        and 0 < info["lap_time"] < env.max_steps * env.dt
    )


def test_passive_steering_leaves_the_track() -> None:
    """The bend is strong enough that not steering ends the episode off track."""
    env = DummyTrackEnv()
    env.reset(seed=0)
    terminated = truncated = False
    info: dict[str, object] = {}
    while not (terminated or truncated):
        _, _, terminated, truncated, info = env.step(np.array([0.0, 1.0, 0.0]))

    assert info["off_track"] is True and info["lap_complete"] is False


def test_standing_still_never_earns_reward() -> None:
    """Reward cannot be farmed without progressing (AGENTS.md §11)."""
    env = DummyTrackEnv(max_steps=20)
    env.reset(seed=0)
    rewards = []
    for _ in range(20):
        _, reward, _, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
        rewards.append(reward)
    assert truncated is True
    assert all(reward <= 0.0 for reward in rewards)


def test_truncates_at_max_steps_and_refuses_steps_after_done() -> None:
    """Episode bookkeeping matches the Gymnasium contract."""
    env = DummyTrackEnv(max_steps=5)
    env.reset(seed=0)
    for _ in range(4):
        _, _, terminated, truncated, _ = env.step(np.zeros(3))
        assert not terminated and not truncated
    _, _, terminated, truncated, _ = env.step(np.zeros(3))
    assert truncated and not terminated
    with pytest.raises(RuntimeError):
        env.step(np.zeros(3))


def test_frames_change_with_progress_and_lateral_offset() -> None:
    """The observation carries state: motion stripes and road position."""
    env = DummyTrackEnv()
    first, _ = env.reset(seed=0)
    for _ in range(100):
        moving, *_ = env.step(np.array([0.0, 1.0, 0.0]))
    assert not np.array_equal(first, moving)
    env._lateral = 2.0
    shifted = env._render_state()
    assert not np.array_equal(moving, shifted)
