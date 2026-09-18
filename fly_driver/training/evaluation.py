"""The one evaluation path: same seed, same actions, same numbers.

:func:`evaluate` runs an agent through a fixed episode protocol and reports lap
time, return, and robustness under configured perturbations. Every episode
``i`` uses seed ``config.seed + i`` for the env, the agent (if it has
``reset(seed=...)``), and any perturbation, so two runs with the same config
produce byte-identical action sequences; :attr:`EvalReport.actions_digest`
makes that cheap to assert.

Sample efficiency is a property of a learning curve, not of one evaluation, so
:class:`EvalConfig.env_steps` records how many training steps the evaluated
policy had seen and :mod:`fly_driver.analysis.learning_curves` turns a series
of reports into steps-to-threshold.

Video is optional: with ``record_video`` the first ``video_episodes`` episodes of
each condition are written as mp4 when a backend exists and skipped with a
warning otherwise (:mod:`fly_driver.training.video`).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from fly_driver.training.perturbations import PerturbationSpec, apply_perturbation
from fly_driver.training.video import VideoWriter, open_video

__all__ = [
    "CLEAN_CONDITION",
    "Agent",
    "ConditionSummary",
    "Env",
    "EpisodeRecord",
    "EvalConfig",
    "EvalReport",
    "evaluate",
    "seed_everything",
]

CLEAN_CONDITION = "clean"

Frame = npt.NDArray[np.uint8]
Action = npt.NDArray[np.float32]


class Agent(Protocol):
    """Frame in, ``(steer, throttle, brake)`` out.

    ``act`` may return an array-like or an object with ``to_array()``. An
    optional ``reset(seed=...)`` is called at each episode start.
    """

    def act(self, observation: Any) -> Any:
        """Choose an action for the current observation."""


class Env(Protocol):
    """Gymnasium-style environment surface used by the harness."""

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        """Start an episode."""

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Advance one frame."""

    def render(self) -> Any:
        """Return an RGB frame or ``None``."""


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation protocol settings.

    Args:
        episodes: Episodes per condition.
        seed: Base seed; episode ``i`` uses ``seed + i``.
        max_steps: Per-episode step cap applied by the harness, or ``None`` to
            rely on the env alone.
        frame_rate_hz: Env frame rate; lap time falls back to ``steps / rate``
            when ``info`` has no ``lap_time``, and it is the video frame rate.
        record_video: Write mp4s of the first ``video_episodes`` per condition.
        video_episodes: How many episodes per condition to record.
        video_backend: ``"auto"``, ``"imageio"``, or ``"cv2"``.
        output_dir: Where ``episodes.csv``, ``summary.json``, and videos go;
            ``None`` keeps everything in memory.
        perturbations: Robustness conditions, each run with the same seeds as
            the clean condition.
        env_steps: Training steps the evaluated policy has seen (for curves).
    """

    episodes: int = 3
    seed: int = 0
    max_steps: int | None = None
    frame_rate_hz: float = 50.0
    record_video: bool = False
    video_episodes: int = 1
    video_backend: str = "auto"
    output_dir: str | None = None
    perturbations: tuple[PerturbationSpec, ...] = field(default_factory=tuple)
    env_steps: int = 0

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be at least 1 or None")
        if self.frame_rate_hz <= 0:
            raise ValueError("frame_rate_hz must be positive")
        if self.video_episodes < 0 or self.video_episodes > self.episodes:
            raise ValueError("video_episodes must be between 0 and episodes")
        if self.env_steps < 0:
            raise ValueError("env_steps must be non-negative")
        specs = tuple(
            spec if isinstance(spec, PerturbationSpec) else PerturbationSpec.from_dict(spec)
            for spec in self.perturbations
        )
        labels = [spec.label for spec in specs]
        if len(set(labels)) != len(labels) or CLEAN_CONDITION in labels:
            raise ValueError(f"perturbation labels must be unique and not 'clean': {labels}")
        object.__setattr__(self, "perturbations", specs)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalConfig:
        """Build from a mapping; unknown keys are an error, not a typo to ignore."""
        known = {item.name for item in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown eval config keys {unknown}; known: {sorted(known)}")
        values = dict(data)
        if "perturbations" in values:
            values["perturbations"] = tuple(values["perturbations"] or ())
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> EvalConfig:
        """Load a YAML mapping (requires PyYAML)."""
        try:
            import yaml
        except ImportError as error:
            raise ImportError("EvalConfig.from_yaml needs PyYAML: pip install pyyaml") from error
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise TypeError(f"{path} must contain a mapping at the top level")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe mapping; inverse of :meth:`from_dict`."""
        data = asdict(self)
        data["perturbations"] = [spec.to_dict() for spec in self.perturbations]
        return data

    @property
    def condition_labels(self) -> tuple[str, ...]:
        """``("clean", <perturbation labels>...)``."""
        return (CLEAN_CONDITION, *(spec.label for spec in self.perturbations))


@dataclass(frozen=True)
class EpisodeRecord:
    """One evaluated episode."""

    condition: str
    episode: int
    seed: int
    steps: int
    total_return: float
    lap_complete: bool
    lap_time_s: float | None
    terminated: bool
    truncated: bool
    actions: Action
    video_path: str | None

    def to_row(self) -> dict[str, Any]:
        """CSV row without the action array."""
        row = asdict(self)
        del row["actions"]
        return row


@dataclass(frozen=True)
class ConditionSummary:
    """Aggregate over the episodes of one condition."""

    condition: str
    episodes: int
    mean_return: float
    std_return: float
    lap_completion_rate: float
    mean_lap_time_s: float | None
    mean_steps: float
    return_ratio_to_clean: float | None


@dataclass(frozen=True)
class EvalReport:
    """Everything :func:`evaluate` produced."""

    config: EvalConfig
    episodes: tuple[EpisodeRecord, ...]
    conditions: tuple[ConditionSummary, ...]
    actions_digest: str
    video_backend: str | None

    def summary(self, condition: str = CLEAN_CONDITION) -> ConditionSummary:
        """Return the summary for one condition label."""
        for item in self.conditions:
            if item.condition == condition:
                return item
        raise KeyError(f"no condition {condition!r}; have {[c.condition for c in self.conditions]}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe summary (episode rows without actions)."""
        return {
            "config": self.config.to_dict(),
            "env_steps": self.config.env_steps,
            "actions_digest": self.actions_digest,
            "video_backend": self.video_backend,
            "conditions": [asdict(item) for item in self.conditions],
            "episodes": [item.to_row() for item in self.episodes],
        }

    def write(self, output_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
        """Write ``episodes.csv`` and ``summary.json`` into ``output_dir``."""
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        csv_path = target / "episodes.csv"
        rows = [item.to_row() for item in self.episodes]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        json_path = target / "summary.json"
        json_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return csv_path, json_path


def seed_everything(seed: int) -> None:
    """Seed ``random``, ``numpy.random``, and torch if it is already imported.

    torch is not imported here: an agent that uses it will have imported it,
    and a numpy-only run should not pay for it.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  load-bearing for determinism protocol
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)


def _as_action(action: object) -> Action:
    raw = action.to_array() if hasattr(action, "to_array") else action
    values = np.asarray(raw, dtype=np.float32)
    if values.shape != (3,):
        raise ValueError(f"agent must return a (3,) action, got shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"agent returned a non-finite action: {values.tolist()}")
    return values


def _capture_frame(env: Any, observation: object) -> Frame | None:
    frame = env.render()
    if frame is None and isinstance(observation, np.ndarray):
        frame = observation
    if isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.ndim == 3:
        return frame
    return None


def _run_episode(
    env: Any,
    agent: Agent,
    config: EvalConfig,
    *,
    condition: str,
    episode: int,
    writer: VideoWriter | None,
) -> EpisodeRecord:
    seed = config.seed + episode
    observation, _ = env.reset(seed=seed)
    reset = getattr(agent, "reset", None)
    if reset is not None:
        reset(seed=seed)
    if writer is not None:
        first = _capture_frame(env, observation)
        if first is not None:
            writer.append(first)

    actions: list[Action] = []
    total_return = 0.0
    terminated = truncated = False
    lap_complete = False
    lap_time: float | None = None
    steps = 0
    while not (terminated or truncated):
        action = _as_action(agent.act(observation))
        observation, reward, terminated, truncated, info = env.step(action)
        terminated, truncated = bool(terminated), bool(truncated)
        actions.append(action)
        total_return += float(reward)
        steps += 1
        if writer is not None:
            frame = _capture_frame(env, observation)
            if frame is not None:
                writer.append(frame)
        if info.get("lap_complete"):
            lap_complete = True
            reported = info.get("lap_time")
            lap_time = float(reported) if reported is not None else steps / config.frame_rate_hz
        if config.max_steps is not None and steps >= config.max_steps and not terminated:
            truncated = True

    stacked = np.stack(actions) if actions else np.zeros((0, 3), dtype=np.float32)
    return EpisodeRecord(
        condition=condition,
        episode=episode,
        seed=seed,
        steps=steps,
        total_return=total_return,
        lap_complete=lap_complete,
        lap_time_s=lap_time,
        terminated=terminated,
        truncated=truncated,
        actions=stacked,
        video_path=None if writer is None else str(writer.path),
    )


def _summarise(
    condition: str, records: Sequence[EpisodeRecord], clean_mean: float | None
) -> ConditionSummary:
    returns = np.array([item.total_return for item in records], dtype=np.float64)
    lap_times = [item.lap_time_s for item in records if item.lap_time_s is not None]
    mean_return = float(returns.mean())
    ratio: float | None = None
    if clean_mean is not None and clean_mean != 0.0:
        ratio = mean_return / clean_mean
    return ConditionSummary(
        condition=condition,
        episodes=len(records),
        mean_return=mean_return,
        std_return=float(returns.std()),
        lap_completion_rate=float(np.mean([item.lap_complete for item in records])),
        mean_lap_time_s=float(np.mean(lap_times)) if lap_times else None,
        mean_steps=float(np.mean([item.steps for item in records])),
        return_ratio_to_clean=ratio,
    )


def _digest(records: Sequence[EpisodeRecord]) -> str:
    hasher = hashlib.sha256()
    for item in records:
        hasher.update(item.condition.encode())
        hasher.update(np.int64(item.steps).tobytes())
        hasher.update(np.ascontiguousarray(item.actions, dtype=np.float32).tobytes())
    return hasher.hexdigest()


def evaluate(make_env: Callable[[], Env], agent: Agent, config: EvalConfig) -> EvalReport:
    """Run the evaluation protocol and optionally write its outputs.

    Args:
        make_env: Factory called once per condition; the harness closes each env.
        agent: Policy under test.
        config: Protocol settings.

    Returns:
        The report. With ``config.output_dir`` set it is also written to disk.
    """
    seed_everything(config.seed)
    output_dir = Path(config.output_dir) if config.output_dir is not None else None
    records: list[EpisodeRecord] = []
    summaries: list[ConditionSummary] = []
    video_backend: str | None = None
    video_wanted = config.record_video and config.video_episodes > 0
    video_dir = Path("videos") if output_dir is None else output_dir / "videos"

    for index, label in enumerate(config.condition_labels):
        env: Any = make_env()
        if index > 0:
            env = apply_perturbation(env, config.perturbations[index - 1])
        condition_records: list[EpisodeRecord] = []
        try:
            for episode in range(config.episodes):
                writer: VideoWriter | None = None
                if video_wanted and episode < config.video_episodes:
                    path = video_dir / f"{_slug(label)}_ep{episode:03d}.mp4"
                    writer = open_video(path, config.frame_rate_hz, config.video_backend)
                    if writer is None:
                        video_wanted = False
                    else:
                        video_backend = writer.backend
                try:
                    record = _run_episode(
                        env,
                        agent,
                        config,
                        condition=label,
                        episode=episode,
                        writer=writer,
                    )
                finally:
                    if writer is not None:
                        writer.close()
                condition_records.append(record)
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        clean_mean = summaries[0].mean_return if summaries else None
        summaries.append(_summarise(label, condition_records, clean_mean))
        records.extend(condition_records)

    report = EvalReport(
        config=config,
        episodes=tuple(records),
        conditions=tuple(summaries),
        actions_digest=_digest(records),
        video_backend=video_backend,
    )
    if output_dir is not None:
        report.write(output_dir)
    elif config.record_video and video_backend is not None:
        warnings.warn(
            f"record_video without output_dir wrote videos under {video_dir}",
            stacklevel=2,
        )
    return report


def _slug(label: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in label).strip("_")
