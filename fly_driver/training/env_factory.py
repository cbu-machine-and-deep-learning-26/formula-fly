"""Build the env a config names, without importing torch or any eye (GH-27).

:mod:`fly_driver.training.train` re-exports :func:`build_env` and :data:`ENV_BUILDERS`. They
live here so the vectorised trainer's worker processes
(:mod:`fly_driver.training.vector_env`) can build a track without paying for torch, flyvis
or a CUDA context in every worker.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from fly_driver.envs.assetto_corsa import AssettoCorsaTrack
from fly_driver.envs.dummy_track import DummyTrackEnv
from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.envs.scene import SceneConfig
from fly_driver.training.config import KNOWN_ENV_TYPES, EnvConfig

__all__ = ["ENV_BUILDERS", "TRAINING_ONLY_ENV_PARAMS", "build_env"]


def _build_practice_track(config: EnvConfig) -> PracticeTrack:
    params = dict(config.params)
    scene_kwargs = dict(params.pop("scene", {}) or {})
    if "racing_line" in params:
        scene_kwargs["racing_line"] = bool(params.pop("racing_line"))
    scene = SceneConfig(**scene_kwargs) if scene_kwargs else None
    return PracticeTrack(scene=scene, max_steps=config.max_steps, **params)


def _build_dummy_track(config: EnvConfig) -> DummyTrackEnv:
    return DummyTrackEnv(max_steps=config.max_steps, **config.params)


def _build_assetto_corsa(config: EnvConfig) -> AssettoCorsaTrack:
    return AssettoCorsaTrack.connect(max_steps=config.max_steps, **config.params)


#: Env type name → builder. Keyed exactly like :data:`~fly_driver.training.config.KNOWN_ENV_TYPES`.
ENV_BUILDERS: dict[str, Callable[[EnvConfig], Any]] = {
    "practice_track": _build_practice_track,
    "dummy": _build_dummy_track,
    "assetto_corsa": _build_assetto_corsa,
}

assert set(ENV_BUILDERS) == set(KNOWN_ENV_TYPES)

TRAINING_ONLY_ENV_PARAMS = frozenset({"random_start", "start_focus"})
#: Episode options the trainer reads and no env sees: ``teacher_drive`` (``{"share": s}``) has
#: the trainer execute the driving teacher's controls for a share ``s`` of training episodes
#: (:func:`~fly_driver.training.vector_train.run_vectorized`); ``teacher_weights``
#: (``{"brake": w}``) counts the teacher's pedal error ``w`` times on steps where it brakes.
TRAINER_ONLY_ENV_PARAMS = frozenset({"teacher_drive", "teacher_weights"})


def env_params(config: EnvConfig, *, training: bool) -> dict[str, Any]:
    """The params an env of ``config`` is built with: never the trainer's own, and the
    training-only ones only for a training env."""
    dropped = TRAINER_ONLY_ENV_PARAMS | (frozenset() if training else TRAINING_ONLY_ENV_PARAMS)
    return {key: value for key, value in config.params.items() if key not in dropped}


def build_env(config: EnvConfig, *, training: bool = False) -> Any:
    """Construct the env a config names.

    Args:
        config: The env section of a run config.
        training: Build the training env. Only then do ``params.random_start`` and
            ``params.start_focus`` reach the env; every evaluation env (periodic,
            ``scripts/evaluate.py``, robustness) drops them and starts on the grid, so lap
            times stay comparable across runs.
    """
    params = env_params(config, training=training)
    if params != dict(config.params):
        config = replace(config, params=params)
    return ENV_BUILDERS[config.type](config)
