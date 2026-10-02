"""The practice track the training loop builds is the one the env tests check (GH-16, GH-17).

``tests/envs/test_practice_track.py`` checks the checked-in training config against the real
track, and it has to restate ``_build_practice_track`` to do it: ``train.py`` imports torch,
CI installs none, and a test that imports it is skipped on CI -- which is how a change to the
env's off-track default once broke training with nothing failing. This file runs wherever
torch *is* installed and checks that restatement against the real builder, so the copy cannot
drift from the original without a test noticing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("yaml")

from fly_driver.training import train as train_module  # noqa: E402
from fly_driver.training.config import TrainConfig  # noqa: E402
from tests.envs.test_practice_track import _built_like_gh17  # noqa: E402

DEFAULT_YAML = Path(__file__).parents[2] / "fly_driver" / "configs" / "train_default.yaml"


def _settings(env) -> dict:
    """What decides how an episode ends and what the camera sees."""
    return {
        "terminate_off_track": env._terminate_off_track,
        "track_limit": env._track_limit,
        "max_steps": env._max_steps,
        "racing_line": env.scene.racing_line,
        "grass_friction_scale": env.scene.grass_friction_scale,
    }


def test_the_env_tests_build_the_track_the_way_training_does():
    config = TrainConfig.from_yaml(DEFAULT_YAML).env
    real = train_module.build_env(config)
    copy = _built_like_gh17(config)
    try:
        assert _settings(copy) == _settings(real)
    finally:
        real.close()
        copy.close()


def test_the_training_config_ends_the_episode_off_the_circuit():
    """Stated against the real builder too, in case the copy is the thing that is wrong."""
    env = train_module.build_env(TrainConfig.from_yaml(DEFAULT_YAML).env)
    try:
        assert env._terminate_off_track is True
        assert env._track_limit > 0.0
    finally:
        env.close()
