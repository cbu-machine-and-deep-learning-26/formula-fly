"""Deterministic evaluation protocol (seeded dummy env by default)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fly_driver.envs.dummy import DummyRaceEnv
from fly_driver.interface import DirectDriveAgent
from fly_driver.training.video import write_episode_video


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    return_: float
    length: int
    actions: tuple[tuple[float, float, float], ...]
    video_path: Path | None


@dataclass(frozen=True, slots=True)
class EvalReport:
    seed: int
    episodes: tuple[EpisodeResult, ...]

    @property
    def mean_return(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([ep.return_ for ep in self.episodes]))


def evaluate(
    agent: DirectDriveAgent,
    *,
    seed: int,
    n_episodes: int = 3,
    max_steps: int = 64,
    record_video: bool = False,
    video_dir: str | Path | None = None,
) -> EvalReport:
    """Run n seeded episodes. Episode i uses seed + i for the env; the agent is reset each time."""
    results: list[EpisodeResult] = []
    for i in range(n_episodes):
        env = DummyRaceEnv(seed=seed + i, max_steps=max_steps)
        agent.reset()
        frame = env.reset(seed=seed + i)
        total = 0.0
        actions: list[tuple[float, float, float]] = []
        frames = [frame.copy()]
        length = 0
        for _ in range(max_steps):
            control = agent.act(frame)
            clipped = control.clipped()
            actions.append((clipped.steer, clipped.throttle, clipped.brake))
            step = env.step(control)
            total += step.reward
            length += 1
            frame = step.frame
            frames.append(frame.copy())
            if step.terminated or step.truncated:
                break
        video_path = None
        if record_video:
            directory = Path(video_dir or "outputs/videos")
            video_path = write_episode_video(directory / f"eval_seed{seed}_ep{i}", frames)
        results.append(
            EpisodeResult(
                return_=float(total),
                length=length,
                actions=tuple(actions),
                video_path=video_path,
            )
        )
    return EvalReport(seed=seed, episodes=tuple(results))
