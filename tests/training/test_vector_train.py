"""The vectorised trainer (GH-27): same advantages, same files, many tracks at once.

Runs on the numpy dummy track with the pixel eye, like ``test_train.py``, so it needs torch
but not flyvis or a GL context.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fly_driver.training import (  # noqa: E402
    BrainConfig,
    EnvConfig,
    EyeConfig,
    PolicyConfig,
    PPOConfig,
    TrainConfig,
)
from fly_driver.training.env_factory import build_env  # noqa: E402
from fly_driver.training.ppo import RolloutBuffer  # noqa: E402
from fly_driver.training.train import train_seed  # noqa: E402
from fly_driver.training.vector_env import SubprocVectorEnv, SyncVectorEnv  # noqa: E402
from fly_driver.training.vector_train import VectorRollout  # noqa: E402


def _config(tmp_path: Path, **overrides) -> TrainConfig:
    base = TrainConfig(
        name="vector",
        output_dir=str(tmp_path / "runs"),
        seeds=(0,),
        total_env_steps=256,
        device="cpu",
        eye=EyeConfig(type="pixels", frozen=True, params={"downsample": 8}),
        policy=PolicyConfig(hidden_sizes=(8,)),
        env=EnvConfig(type="dummy", max_steps=40),
        ppo=PPOConfig(rollout_steps=64, minibatches=4, epochs=2),
        num_envs=4,
    )
    return replace(base, **overrides)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_rollout_steps_must_divide_by_num_envs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="divide by num_envs"):
        _config(tmp_path, num_envs=3)


def test_each_tracks_advantages_match_the_single_track_recursion() -> None:
    """GAE down each column is exactly what RolloutBuffer computes for that track alone."""
    rng = np.random.default_rng(0)
    steps, num_envs = 17, 3
    rollout = VectorRollout(steps, num_envs, feature_dim=2)
    rollout.rewards[:] = rng.normal(size=(steps, num_envs))
    rollout.values[:] = rng.normal(size=(steps, num_envs))
    rollout.episode_ends[:] = rng.random((steps, num_envs)) < 0.2
    last_values = rng.normal(size=num_envs)
    last_values[1] = 0.0 if rollout.episode_ends[-1, 1] else last_values[1]
    rollout.compute_advantages(last_values, gamma=0.97, gae_lambda=0.9)

    for track in range(num_envs):
        single = RolloutBuffer(steps, 2)
        for step in range(steps):
            single.add(
                features=np.zeros(2, np.float32),
                raw_action=np.zeros(3, np.float32),
                log_prob=0.0,
                reward=float(rollout.rewards[step, track]),
                episode_end=bool(rollout.episode_ends[step, track]),
                value=float(rollout.values[step, track]),
            )
        single.compute_advantages(float(last_values[track]), gamma=0.97, gae_lambda=0.9)
        np.testing.assert_allclose(rollout.advantages[:, track], single.advantages, rtol=1e-5)
        np.testing.assert_allclose(rollout.returns[:, track], single.returns, rtol=1e-5)


def test_flatten_keeps_each_tracks_steps_together() -> None:
    rollout = VectorRollout(3, 2, feature_dim=1)
    rollout.rewards[:] = np.array([[0, 10], [1, 11], [2, 12]], dtype=np.float32)
    buffer = rollout.flatten()
    np.testing.assert_array_equal(buffer.rewards, [0, 1, 2, 10, 11, 12])
    assert buffer.filled == buffer.steps == 6


def test_a_track_that_hits_the_step_cap_resets_and_keeps_its_last_frame() -> None:
    env_config = EnvConfig(type="dummy", max_steps=5)
    vector = SyncVectorEnv(lambda: build_env(env_config, training=True), 2)
    try:
        vector.reset([0, 1])
        actions = np.tile([0.0, 0.3, 0.0], (2, 1))
        for _ in range(4):
            result = vector.step(actions, [7, 8])
            assert not result.truncated.any()
        result = vector.step(actions, [7, 8])
        assert result.truncated.all()
        assert sorted(result.final_frames) == [0, 1]
        assert all(info is not None for info in result.reset_infos)
        # The observation is already the next episode's first frame: one more step is fine.
        assert not vector.step(actions, [9, 10]).truncated.any()
    finally:
        vector.close()


def test_worker_processes_step_exactly_like_one_process() -> None:
    env_config = EnvConfig(type="dummy", max_steps=6)
    sync = SyncVectorEnv(lambda: build_env(env_config, training=True), 2)
    workers = SubprocVectorEnv(env_config, 2)
    try:
        assert workers.frame_shape == sync.frame_shape
        sync.reset([3, 4])
        workers.reset([3, 4])
        rng = np.random.default_rng(1)
        for _ in range(9):
            actions = np.column_stack(
                [rng.uniform(-1, 1, 2), rng.uniform(0, 1, 2), rng.uniform(0, 0.2, 2)]
            )
            left, right = sync.step(actions, [5, 6]), workers.step(actions, [5, 6])
            np.testing.assert_array_equal(left.rewards, right.rewards)
            np.testing.assert_array_equal(left.truncated, right.truncated)
            np.testing.assert_array_equal(sync.frames, workers.frames)
    finally:
        sync.close()
        workers.close()


def test_a_vectorised_run_writes_the_same_files(tmp_path: Path) -> None:
    config = _config(
        tmp_path, logging=replace(_config(tmp_path).logging, checkpoint_every_updates=2)
    )
    result = train_seed(config, 0)
    run_dir = result.run_dir
    assert result.env_steps == 256 and result.updates == 4
    episodes = _rows(run_dir / "episodes.csv")
    updates = _rows(run_dir / "updates.csv")
    assert len(updates) == 4
    assert [int(row["episode"]) for row in episodes] == list(range(len(episodes)))
    # Four tracks capped at 40 steps over 64 steps each: every track finishes at least once.
    assert len(episodes) >= 4
    for row in episodes:
        terms = sum(float(v) for k, v in row.items() if k.startswith("reward_"))
        assert terms == pytest.approx(float(row["total_return"]), abs=1e-6)
    assert (run_dir / "policy_final.pt").exists()
    assert (run_dir / "policy_update_00002.pt").exists()


def test_a_vectorised_run_with_a_brain_trains_and_saves_it(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        brain=BrainConfig(type="central_complex", params={"neuron_count": 64, "output_dim": 8}),
    )
    result = train_seed(config, 0)
    assert (result.run_dir / "brain_final.pt").exists()
    assert result.last_update is not None
    assert np.isfinite(result.last_update.policy_loss)
