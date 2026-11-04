"""The variable-controls recipe (GH-104): train_lab.yaml plus its two changes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fly_driver.training import TrainConfig

CONFIGS = Path(__file__).resolve().parents[2] / "fly_driver" / "configs"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))


def test_the_recipe_is_train_lab_plus_its_changes_and_nothing_else() -> None:
    lab, recipe = _load("train_lab.yaml"), _load("train_lab_variable.yaml")
    assert recipe["env"]["params"].pop("car") == {
        "steer_gain": 192000.0,
        "steer_damping_nms": 800.0,
    }
    assert recipe.pop("init_from") == "weights/variable_start/policy_best.pt"
    assert recipe.pop("name") == "train_lab_variable"
    del lab["init_from"], lab["name"]
    assert recipe == lab


def test_the_recipe_loads_as_a_train_config() -> None:
    config = TrainConfig.from_yaml(CONFIGS / "train_lab_variable.yaml")
    assert config.init_from == "weights/variable_start/policy_best.pt"


def test_the_range_penalty_is_off_because_it_stalled_launches() -> None:
    config = TrainConfig.from_yaml(CONFIGS / "train_lab_variable.yaml")
    assert config.ppo.output_range_coef == 0.0


def test_the_lab_recipe_has_none_of_the_changes() -> None:
    lab = _load("train_lab.yaml")
    assert "car" not in lab["env"]["params"]
    assert lab["init_from"] == "weights/best/policy_best.pt"


def test_the_recipe_builds_a_car_with_the_stiff_steering() -> None:
    pytest.importorskip("torch")
    from fly_driver.training.train import build_env

    config = TrainConfig.from_yaml(CONFIGS / "train_lab_variable.yaml")
    env = build_env(config.env)
    try:
        assert env.car.steer_gain == 192000.0
        assert env.car.steer_damping_nms == 800.0
        actuator = env.model.actuator("steer_fl").id
        assert env.model.actuator_gainprm[actuator][0] == 192000.0
        assert env.model.dof_damping[env.model.joint("steer_fl").dofadr[0]] == 800.0
    finally:
        env.close()


def test_the_lab_recipe_builds_the_default_car() -> None:
    pytest.importorskip("torch")
    from fly_driver.training.train import build_env

    config = TrainConfig.from_yaml(CONFIGS / "train_lab.yaml")
    env = build_env(config.env)
    try:
        assert env.car.steer_gain == 12000.0
    finally:
        env.close()
