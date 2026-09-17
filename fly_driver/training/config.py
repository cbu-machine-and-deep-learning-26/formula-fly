"""Typed experiment config: condition = {eye_type, brain, frozen/finetuned, seed}."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

EyeType = Literal["flyvis", "cnn", "random_projection", "shuffled"]
BrainType = Literal["identity", "lif"]
EyeMode = Literal["frozen", "finetuned"]


@dataclass(frozen=True, slots=True)
class Condition:
    eye_type: EyeType = "random_projection"
    brain: BrainType = "identity"
    eye_mode: EyeMode = "frozen"
    seed: int = 0


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    csv_path: str = "outputs/metrics.csv"
    wandb: bool = False
    wandb_project: str = "formula-fly"
    wandb_mode: str = "offline"


@dataclass(frozen=True, slots=True)
class EvalConfig:
    deterministic: bool = True
    n_episodes: int = 3
    record_video: bool = False
    video_dir: str = "outputs/videos"
    max_steps: int = 64


@dataclass(frozen=True, slots=True)
class EnvConfig:
    name: str = "dummy"
    frame_shape: tuple[int, int, int] = (96, 96, 3)


@dataclass(frozen=True, slots=True)
class EyeConfig:
    output_size: int = 256


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    condition: Condition = field(default_factory=Condition)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    eye: EyeConfig = field(default_factory=EyeConfig)

    @property
    def frozen(self) -> bool:
        return self.condition.eye_mode == "frozen"


def _tuple3(value: Any) -> tuple[int, int, int]:
    seq = tuple(int(v) for v in value)
    if len(seq) != 3:
        raise ValueError(f"frame_shape must be 3 ints, got {value!r}")
    return seq[0], seq[1], seq[2]


def load_experiment(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    cond = raw.get("condition") or {}
    log = raw.get("logging") or {}
    ev = raw.get("eval") or {}
    env = raw.get("env") or {}
    eye = raw.get("eye") or {}
    frame = env.get("frame_shape", (96, 96, 3))
    return ExperimentConfig(
        condition=Condition(
            eye_type=cond.get("eye_type", "random_projection"),
            brain=cond.get("brain", "identity"),
            eye_mode=cond.get("eye_mode", "frozen"),
            seed=int(cond.get("seed", 0)),
        ),
        logging=LoggingConfig(
            csv_path=str(log.get("csv_path", "outputs/metrics.csv")),
            wandb=bool(log.get("wandb", False)),
            wandb_project=str(log.get("wandb_project", "formula-fly")),
            wandb_mode=str(log.get("wandb_mode", "offline")),
        ),
        eval=EvalConfig(
            deterministic=bool(ev.get("deterministic", True)),
            n_episodes=int(ev.get("n_episodes", 3)),
            record_video=bool(ev.get("record_video", False)),
            video_dir=str(ev.get("video_dir", "outputs/videos")),
            max_steps=int(ev.get("max_steps", 64)),
        ),
        env=EnvConfig(
            name=str(env.get("name", "dummy")),
            frame_shape=_tuple3(frame),
        ),
        eye=EyeConfig(output_size=int(eye.get("output_size", 256))),
    )


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
