"""Robustness evaluation of a finished training run's final policy (GH-27).

Runs the evaluation harness (:func:`fly_driver.training.evaluate`) on a seed's
``policy_final.pt``, with the driver rebuilt by :func:`fly_driver.training.train.load_driver`
from the run's own ``config.yaml``: a seeded control eye or shuffle comes out identical to
the one it trained with, and a brain run gets its ``brain_final.pt``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from fly_driver.training.config import TrainConfig
from fly_driver.training.evaluation import EvalConfig, EvalReport, evaluate
from fly_driver.training.train import FINAL_CHECKPOINT, build_env, load_driver, resolve_device

__all__ = ["evaluate_checkpoint"]


def evaluate_checkpoint(run_dir: str | Path, eval_config: EvalConfig) -> EvalReport:
    """Evaluate a seed's final driver; the report is written to ``eval_config.output_dir``.

    Args:
        run_dir: A seed's run directory, ``<output_dir>/<name>/seed_<k>``.
        eval_config: The protocol (episodes, seed, perturbations, output_dir).
            ``frame_rate_hz`` and ``env_steps`` are taken from the run.
    """
    run_dir = Path(run_dir)
    config = TrainConfig.from_yaml(run_dir / "config.yaml")
    env = build_env(config.env)
    try:
        driver = load_driver(
            config, run_dir / FINAL_CHECKPOINT, env, device=resolve_device(config.device)
        )
        frame_rate_hz = float(env.frame_rate_hz)
    finally:
        env.close()
    eval_config = replace(
        eval_config,
        frame_rate_hz=frame_rate_hz,
        env_steps=int(getattr(driver.policy, "extra", {}).get("env_steps", 0)),
    )
    return evaluate(lambda: build_env(config.env), driver, eval_config)
