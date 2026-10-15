"""Robustness evaluation of a finished run's checkpoint (GH-27), on the dummy track."""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from fly_driver.analysis.robustness import PROGRESS_FILENAME, evaluate_checkpoint  # noqa: E402
from fly_driver.training import (  # noqa: E402
    EnvConfig,
    EvalConfig,
    EyeConfig,
    PolicyConfig,
    PPOConfig,
    TrainConfig,
)
from fly_driver.training.perturbations import PerturbationSpec  # noqa: E402
from fly_driver.training.train import train_seed  # noqa: E402

pytestmark = pytest.mark.usefixtures("cpu_default_device")


def test_the_final_policy_replays_deterministically_with_progress_per_episode(tmp_path: Path):
    config = TrainConfig(
        name="test",
        output_dir=str(tmp_path / "runs"),
        seeds=(0,),
        total_env_steps=256,
        device="cpu",
        eye=EyeConfig(type="pixels", frozen=True, params={"downsample": 8}),
        policy=PolicyConfig(hidden_sizes=(8,)),
        env=EnvConfig(type="dummy", max_steps=40),
        ppo=PPOConfig(rollout_steps=64, minibatches=4, epochs=2),
    )
    run_dir = train_seed(config, 0).run_dir
    protocol = EvalConfig(
        episodes=2,
        perturbations=(PerturbationSpec.from_dict({"name": "brightness", "scale": 0.5}),),
    )
    first = evaluate_checkpoint(run_dir, replace(protocol, output_dir=str(tmp_path / "a")))
    second = evaluate_checkpoint(run_dir, replace(protocol, output_dir=str(tmp_path / "b")))

    assert first.config.env_steps == 256
    assert first.actions_digest == second.actions_digest
    with (tmp_path / "a" / PROGRESS_FILENAME).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["condition"], row["episode"]) for row in rows] == [
        (record.condition, str(record.episode)) for record in first.episodes
    ]
    assert len(rows) == 4
