"""Eval seed determinism (CLAUDE.md §11)."""

from __future__ import annotations

from pathlib import Path

from fly_driver.eyes import RandomProjectionEye
from fly_driver.interface import DirectDriveAgent
from fly_driver.policies import LinearPolicy
from fly_driver.training.eval import evaluate


def _agent(seed: int) -> DirectDriveAgent:
    eye = RandomProjectionEye(output_size=32, seed=seed)
    policy = LinearPolicy(feature_size=32, seed=seed)
    return DirectDriveAgent(eye, policy)


def test_eval_is_deterministic_given_seed() -> None:
    a = evaluate(_agent(0), seed=0, n_episodes=2, max_steps=16)
    b = evaluate(_agent(0), seed=0, n_episodes=2, max_steps=16)
    assert a.mean_return == b.mean_return
    assert a.episodes[0].actions == b.episodes[0].actions
    assert a.episodes[1].length == b.episodes[1].length


def test_different_seeds_change_rollout() -> None:
    a = evaluate(_agent(0), seed=0, n_episodes=1, max_steps=16)
    b = evaluate(_agent(0), seed=1, n_episodes=1, max_steps=16)
    assert a.episodes[0].actions != b.episodes[0].actions


def test_eval_video_fallback_npz(tmp_path: Path) -> None:
    report = evaluate(
        _agent(0),
        seed=0,
        n_episodes=1,
        max_steps=4,
        record_video=True,
        video_dir=tmp_path,
    )
    path = report.episodes[0].video_path
    assert path is not None
    assert path.exists()
    assert path.suffix in {".npz", ".mp4"}
