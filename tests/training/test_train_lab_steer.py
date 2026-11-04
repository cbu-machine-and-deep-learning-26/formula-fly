"""The steering-cost trial's recipe (GH-104): train_lab.yaml plus one change and a shorter run."""

from __future__ import annotations

from pathlib import Path

import yaml

from fly_driver.training import TrainConfig

CONFIGS = Path(__file__).resolve().parents[2] / "fly_driver" / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_the_trial_is_train_lab_plus_its_changes_and_nothing_else() -> None:
    lab, trial = _load("train_lab.yaml"), _load("train_lab_steer.yaml")
    assert trial["env"]["params"]["reward"].pop("steer_change_m") == 0.25
    assert trial.pop("name") == "train_lab_steer"
    assert trial.pop("total_env_steps") == 1_500_000
    del lab["name"], lab["total_env_steps"]
    assert trial == lab


def test_the_trial_loads_as_a_train_config() -> None:
    config = TrainConfig.from_yaml(CONFIGS / "train_lab_steer.yaml")
    assert config.env.params["reward"]["steer_change_m"] == 0.25
    assert config.init_from == "weights/best/policy_best.pt"


def test_the_lab_recipe_has_no_steering_cost() -> None:
    assert "steer_change_m" not in _load("train_lab.yaml")["env"]["params"]["reward"]
