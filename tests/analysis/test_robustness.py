"""Robustness evaluation of a finished run's checkpoint (GH-27), on the dummy track."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("torch")

from fly_driver.analysis.robustness import evaluate_checkpoint  # noqa: E402
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


def test_the_final_policy_replays_deterministically(tmp_path: Path):
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
    assert [item.condition for item in first.conditions] == list(protocol.condition_labels)
    assert (tmp_path / "a" / "summary.json").exists()
