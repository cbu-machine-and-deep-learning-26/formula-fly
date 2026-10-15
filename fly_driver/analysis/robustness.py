"""Robustness evaluation of a finished training run's final policy (GH-27).

Runs the evaluation harness (:func:`fly_driver.training.evaluate`) on a seed's
``policy_final.pt`` with the env and eye rebuilt from the run's own ``config.yaml``, so a
seeded control eye or shuffle comes out identical to the one it trained with. The harness
does not report track progress, so each episode's summed ``info["progress_m"]`` is recorded
alongside it in ``progress.csv``.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Any

from fly_driver.drivers import DirectDriveAgent
from fly_driver.policies.mlp_policy import MlpPolicy
from fly_driver.training.config import TrainConfig
from fly_driver.training.evaluation import EvalConfig, EvalReport, evaluate
from fly_driver.training.train import FINAL_CHECKPOINT, build_env, build_eye, resolve_device

__all__ = ["PROGRESS_FILENAME", "evaluate_checkpoint"]

PROGRESS_FILENAME = "progress.csv"


class _ProgressRecorder:
    """Env pass-through that appends one summed ``progress_m`` per episode to ``sink``."""

    def __init__(self, env: Any, sink: list[float]) -> None:
        self._env = env
        self._sink = sink

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._sink.append(0.0)
        return self._env.reset(**kwargs)

    def step(self, action: object) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        result = self._env.step(action)
        self._sink[-1] += float(result[4].get("progress_m", 0.0))
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)


def evaluate_checkpoint(run_dir: str | Path, eval_config: EvalConfig) -> EvalReport:
    """Evaluate a seed's final policy; write the report and ``progress.csv`` to its output_dir.

    Args:
        run_dir: A seed's run directory, ``<output_dir>/<name>/seed_<k>``.
        eval_config: The protocol (episodes, seed, perturbations, output_dir).
            ``frame_rate_hz`` and ``env_steps`` are taken from the run.

    Raises:
        ValueError: If the run fine-tuned its eye (its weights are not reloaded here) or
            ``eval_config.output_dir`` is unset.
    """
    run_dir = Path(run_dir)
    if eval_config.output_dir is None:
        raise ValueError("eval_config.output_dir must be set: progress.csv is written there")
    config = TrainConfig.from_yaml(run_dir / "config.yaml")
    if not config.eye.frozen:
        raise ValueError(f"{run_dir} fine-tuned its eye; only frozen-eye runs are supported")
    policy = MlpPolicy.load(run_dir / FINAL_CHECKPOINT, device=resolve_device(config.device))
    policy.eval()
    env = build_env(config.env)
    try:
        eye = build_eye(config.eye, env)
        frame_rate_hz = float(env.frame_rate_hz)
    finally:
        env.close()
    eval_config = replace(
        eval_config,
        frame_rate_hz=frame_rate_hz,
        env_steps=int(policy.extra.get("env_steps", 0)),
    )
    progress: list[float] = []
    report = evaluate(
        lambda: _ProgressRecorder(build_env(config.env), progress),
        DirectDriveAgent(eye, policy),
        eval_config,
    )
    if len(progress) != len(report.episodes):
        raise RuntimeError(
            f"recorded {len(progress)} episodes, the report has {len(report.episodes)}"
        )
    with (Path(eval_config.output_dir) / PROGRESS_FILENAME).open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["condition", "episode", "progress_m"])
        for record, metres in zip(report.episodes, progress, strict=True):
            writer.writerow([record.condition, record.episode, metres])
    return report
