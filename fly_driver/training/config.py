"""Training configuration: one experiment condition, the loop that runs it, and its logs.

`AGENTS.md` §10 fixes what a condition is -- ``{eye_type, brain, frozen/finetuned, seed}`` --
and §6 fixes that every condition runs at least three seeds. Both live here as typed fields,
so a sweep is a YAML file per condition and never a script with the seed edited by hand.
:data:`DEFAULT_SEEDS` is ``(0, 1, 2)``; ``scripts/train.py --seed N`` narrows a run to one of
them so the three can be spread across machines.

Everything is validated at construction. Unknown keys are an error at every level of the
mapping, because a misspelt ``learning_rate`` that silently keeps the default is the
plausible-looking failure `AGENTS.md` §11 warns about, and a sweep that ran the wrong
condition for a week is the worst version of it.

The names an experiment can ask for -- :data:`KNOWN_EYE_TYPES`, :data:`KNOWN_BRAINS`,
:data:`KNOWN_ENV_TYPES` -- are listed here so a config can be checked without importing
torch; the builders that turn a name into an object live in :mod:`fly_driver.training.train`
and a test keeps the two lists in step. torch-free on purpose: the evaluation harness and
the base install import this package.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

__all__ = [
    "DEFAULT_SEEDS",
    "KNOWN_ACTIVATIONS",
    "KNOWN_BRAINS",
    "KNOWN_ENV_TYPES",
    "KNOWN_EYE_TYPES",
    "EnvConfig",
    "EyeConfig",
    "LoggingConfig",
    "PPOConfig",
    "PeriodicEvalConfig",
    "PolicyConfig",
    "TrainConfig",
    "WandbConfig",
]

#: The seeds a condition runs unless told otherwise. `AGENTS.md` §6: minimum three.
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)

#: Eyes the loop knows how to build. ``flyvis`` is the default (`AGENTS.md` §6); ``pixels``
#: is block-averaged grey pixels, the smoke-test eye that runs anywhere. The RQ1 controls --
#: CNN, random projection, degree-matched shuffled connectome -- are GH-15's and register
#: here when they land. Builders: ``fly_driver.training.train.EYE_BUILDERS``.
KNOWN_EYE_TYPES: tuple[str, ...] = ("flyvis", "pixels")

#: Brains between the eye and the policy. ``none`` reads the eye directly (the RQ1 rung);
#: the central-complex / whole-brain model (GH-23) registers here.
KNOWN_BRAINS: tuple[str, ...] = ("none",)

#: Environments. ``practice_track`` is the MuJoCo circuit; ``dummy`` is the numpy stand-in
#: the harness and the tests use.
KNOWN_ENV_TYPES: tuple[str, ...] = ("practice_track", "dummy")

KNOWN_ACTIVATIONS: tuple[str, ...] = ("tanh", "relu")

_WANDB_MODES = ("online", "offline", "disabled")


def _check_int(name: str, value: object, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return int(value)


def _check_finite(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return float(value)


def _check_unit_interval(name: str, value: object) -> float:
    number = _check_finite(name, value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {number}")
    return number


def _check_params(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys, got {value!r}")
    return dict(value)


@dataclass(frozen=True)
class EyeConfig:
    """Which eye, whether it learns, and how to build it.

    Args:
        type: One of :data:`KNOWN_EYE_TYPES`.
        frozen: ``True`` trains only the policy on top of a fixed eye -- the project default
            (`AGENTS.md` §6). ``False`` puts the eye's own parameters into the PPO optimiser,
            which needs an eye that exposes a differentiable batch encoder; the loop refuses
            eyes that do not rather than quietly training the head alone.
        params: Keyword arguments for the eye's constructor, e.g. ``readouts`` or
            ``checkpoint`` for flyvis, ``downsample`` for pixels. ``frame_shape`` and
            ``frame_rate_hz`` are supplied from the env and must not be repeated here.
    """

    type: str = "flyvis"
    frozen: bool = True
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in KNOWN_EYE_TYPES:
            raise ValueError(f"unknown eye type {self.type!r}; choose from {list(KNOWN_EYE_TYPES)}")
        if not isinstance(self.frozen, bool):
            raise ValueError(f"eye.frozen must be true or false, got {self.frozen!r}")
        params = _check_params("eye.params", self.params)
        reserved = sorted({"frame_shape", "frame_rate_hz"} & set(params))
        if reserved:
            raise ValueError(
                f"eye.params must not set {reserved}; the loop takes them from the env so the "
                "eye and the track cannot disagree (see fly_driver/interface.py)"
            )
        object.__setattr__(self, "params", params)


@dataclass(frozen=True)
class PolicyConfig:
    """The small policy head that reads the eye (or the brain).

    Args:
        hidden_sizes: Widths of the hidden layers of both the actor and the critic. Empty
            means a linear readout.
        activation: ``tanh`` or ``relu``.
        normalize_features: Standardise features with running statistics before the first
            layer. Optic-lobe activity spans roughly -3 to 6 a.u. with very different scales
            per cell type; pixels sit in ``[0, 1]``. The statistics are part of the checkpoint.
        init_log_std: Initial log standard deviation of the Gaussian over the raw action.
    """

    hidden_sizes: tuple[int, ...] = (64, 64)
    activation: str = "tanh"
    normalize_features: bool = True
    init_log_std: float = 0.0

    def __post_init__(self) -> None:
        sizes = tuple(self.hidden_sizes)
        for size in sizes:
            _check_int("policy.hidden_sizes entries", size, minimum=1)
        object.__setattr__(self, "hidden_sizes", sizes)
        if self.activation not in KNOWN_ACTIVATIONS:
            raise ValueError(
                f"unknown policy.activation {self.activation!r}; choose from {KNOWN_ACTIVATIONS}"
            )
        object.__setattr__(
            self, "init_log_std", _check_finite("policy.init_log_std", self.init_log_std)
        )


@dataclass(frozen=True)
class EnvConfig:
    """Which track and how long an episode may run.

    Args:
        type: One of :data:`KNOWN_ENV_TYPES`.
        max_steps: Episode truncation, in env steps (one step is one frame). A slow but
            clean lap of the practice track is five minutes of simulated time; the default
            allows it.
        params: Extra constructor keyword arguments. For ``practice_track`` the key
            ``racing_line`` (bool) selects the painted line -- leave it on for hand driving,
            turn it off for RQ1 comparisons (see the env's docstring) -- and everything else
            (``track_limit``, ``frame_shape``...) goes to ``PracticeTrack`` as given.
    """

    type: str = "practice_track"
    max_steps: int = 15_000
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in KNOWN_ENV_TYPES:
            raise ValueError(f"unknown env type {self.type!r}; choose from {list(KNOWN_ENV_TYPES)}")
        object.__setattr__(
            self, "max_steps", _check_int("env.max_steps", self.max_steps, minimum=1)
        )
        params = _check_params("env.params", self.params)
        if "max_steps" in params:
            raise ValueError("set env.max_steps, not env.params.max_steps")
        object.__setattr__(self, "params", params)


@dataclass(frozen=True)
class PPOConfig:
    """Proximal policy optimisation hyperparameters, CleanRL's continuous-control defaults.

    Args:
        rollout_steps: Env steps collected per update.
        minibatches: Minibatches per epoch; must divide ``rollout_steps``.
        epochs: Passes over each rollout.
        learning_rate: Adam step size.
        anneal_learning_rate: Decay the step size linearly to zero over the run.
        gamma: Discount. At 50 Hz, 0.99 is a two-second horizon.
        gae_lambda: Generalised advantage estimation trace decay.
        clip_coef: Policy ratio clip.
        clip_value_loss: Clip the value update the same way.
        entropy_coef: Weight of the entropy bonus.
        value_coef: Weight of the value loss.
        max_grad_norm: Global gradient clip.
        normalize_advantages: Standardise advantages per minibatch.
        target_kl: Stop an update early past this approximate KL; ``None`` never stops.
    """

    rollout_steps: int = 2048
    minibatches: int = 32
    epochs: int = 10
    learning_rate: float = 3e-4
    anneal_learning_rate: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    clip_value_loss: bool = True
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    target_kl: float | None = None

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "rollout_steps", _check_int("ppo.rollout_steps", self.rollout_steps, minimum=1))
        set_(self, "minibatches", _check_int("ppo.minibatches", self.minibatches, minimum=1))
        set_(self, "epochs", _check_int("ppo.epochs", self.epochs, minimum=1))
        if self.rollout_steps % self.minibatches != 0:
            raise ValueError(
                f"ppo.minibatches={self.minibatches} must divide "
                f"ppo.rollout_steps={self.rollout_steps}"
            )
        if _check_finite("ppo.learning_rate", self.learning_rate) <= 0:
            raise ValueError(f"ppo.learning_rate must be positive, got {self.learning_rate}")
        set_(self, "gamma", _check_unit_interval("ppo.gamma", self.gamma))
        set_(self, "gae_lambda", _check_unit_interval("ppo.gae_lambda", self.gae_lambda))
        if _check_finite("ppo.clip_coef", self.clip_coef) <= 0:
            raise ValueError(f"ppo.clip_coef must be positive, got {self.clip_coef}")
        for name in ("entropy_coef", "value_coef"):
            if _check_finite(f"ppo.{name}", getattr(self, name)) < 0:
                raise ValueError(f"ppo.{name} must be non-negative, got {getattr(self, name)}")
        if _check_finite("ppo.max_grad_norm", self.max_grad_norm) <= 0:
            raise ValueError(f"ppo.max_grad_norm must be positive, got {self.max_grad_norm}")
        if self.target_kl is not None and _check_finite("ppo.target_kl", self.target_kl) <= 0:
            raise ValueError(f"ppo.target_kl must be positive or null, got {self.target_kl}")

    @property
    def minibatch_size(self) -> int:
        """Samples per minibatch."""
        return self.rollout_steps // self.minibatches


@dataclass(frozen=True)
class PeriodicEvalConfig:
    """Run the deterministic evaluation protocol every so often during training.

    Each evaluation is a full :func:`fly_driver.training.evaluate` call with ``env_steps``
    set, written under ``<run_dir>/eval/step_<n>/``, so
    :func:`fly_driver.analysis.learning_curve` reads a run's learning curve straight off the
    disk. Lap completion -- the later success signal -- is measured here, on the
    deterministic policy, not on the noisy training episodes.

    Args:
        every_updates: Evaluate after every this many PPO updates; ``0`` disables it.
        episodes: Episodes per evaluation.
        max_steps: Harness-side cap per episode; ``None`` leaves it to the env.
        record_video: Record the first episode of each evaluation when a backend exists.
    """

    every_updates: int = 0
    episodes: int = 1
    max_steps: int | None = None
    record_video: bool = False

    def __post_init__(self) -> None:
        set_ = object.__setattr__
        set_(self, "every_updates", _check_int("eval.every_updates", self.every_updates, minimum=0))
        set_(self, "episodes", _check_int("eval.episodes", self.episodes, minimum=1))
        if self.max_steps is not None:
            set_(self, "max_steps", _check_int("eval.max_steps", self.max_steps, minimum=1))


@dataclass(frozen=True)
class WandbConfig:
    """Weights & Biases, off unless asked for. CSV logging does not depend on it.

    Args:
        enabled: Log to W&B as well as CSV. The ``wandb`` package is imported only then.
        project: W&B project.
        entity: W&B entity (team or user); ``None`` uses the account default.
        mode: ``online``, ``offline`` or ``disabled``.
        tags: Free-form run tags.
    """

    enabled: bool = False
    project: str = "formula-fly"
    entity: str | None = None
    mode: str = "online"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError(f"logging.wandb.enabled must be true or false, got {self.enabled!r}")
        if self.mode not in _WANDB_MODES:
            raise ValueError(f"logging.wandb.mode must be one of {_WANDB_MODES}, got {self.mode!r}")
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))


@dataclass(frozen=True)
class LoggingConfig:
    """Where the numbers go. CSV is always written; there is no switch for it.

    Args:
        wandb: Optional Weights & Biases mirror of the same rows.
        checkpoint_every_updates: Also save the policy every this many updates. ``0`` saves
            only ``policy_final.pt``.
    """

    wandb: WandbConfig = field(default_factory=WandbConfig)
    checkpoint_every_updates: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "checkpoint_every_updates",
            _check_int(
                "logging.checkpoint_every_updates", self.checkpoint_every_updates, minimum=0
            ),
        )


#: Nested dataclass fields, per owner, for :meth:`TrainConfig.from_dict`.
_NESTED: dict[type, dict[str, type]] = {
    LoggingConfig: {"wandb": WandbConfig},
}


def _from_mapping(cls: type, data: Mapping[str, Any], *, path: str) -> Any:
    """Build ``cls`` from a mapping, recursing into nested configs; unknown keys raise."""
    if not isinstance(data, Mapping):
        raise TypeError(f"{path} must be a mapping, got {type(data).__name__}")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown {path} keys {unknown}; known: {sorted(known)}")
    nested = _NESTED.get(cls, {})
    values: dict[str, Any] = {}
    for key, value in data.items():
        if key in nested:
            values[key] = _from_mapping(nested[key], value, path=f"{path}.{key}")
        elif isinstance(value, list):
            values[key] = tuple(value)
        else:
            values[key] = value
    return cls(**values)


def _yaml_safe(value: Any) -> Any:
    """Tuples become lists so PyYAML writes plain sequences rather than ``!!python/tuple``."""
    if isinstance(value, Mapping):
        return {key: _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(item) for item in value]
    return value


@dataclass(frozen=True)
class TrainConfig:
    """One training condition and the loop that runs it.

    Args:
        name: Run name; runs land in ``<output_dir>/<name>/seed_<k>/``.
        output_dir: Root for run directories.
        seeds: The seeds this condition runs, each a complete training run with its own
            env, eye, policy and logs. :data:`DEFAULT_SEEDS` unless narrowed.
        total_env_steps: Env steps per seed. Rounded down to whole rollouts.
        device: ``auto`` (CUDA when available, else CPU), ``cpu``, or a torch device string.
        eye: The visual frontend.
        brain: One of :data:`KNOWN_BRAINS`.
        policy: The readout on top.
        env: The track.
        ppo: The optimiser.
        eval: Periodic deterministic evaluation.
        logging: CSV always; W&B and checkpoints optional.
    """

    name: str = "flyvis_frozen"
    output_dir: str = "runs"
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    total_env_steps: int = 1_000_000
    device: str = "auto"
    eye: EyeConfig = field(default_factory=EyeConfig)
    brain: str = "none"
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    eval: PeriodicEvalConfig = field(default_factory=PeriodicEvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def __post_init__(self) -> None:
        if not self.name or os.sep in self.name or "/" in self.name or self.name in (".", ".."):
            raise ValueError(f"name must be a single path component, got {self.name!r}")
        seeds = (self.seeds,) if isinstance(self.seeds, int) else tuple(self.seeds)
        if not seeds:
            raise ValueError("seeds must name at least one seed (AGENTS.md §6 asks for three)")
        for seed in seeds:
            _check_int("seeds entries", seed, minimum=0)
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"seeds must be distinct, got {list(seeds)}")
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in seeds))
        object.__setattr__(
            self, "total_env_steps", _check_int("total_env_steps", self.total_env_steps, minimum=1)
        )
        if self.total_env_steps < self.ppo.rollout_steps:
            raise ValueError(
                f"total_env_steps={self.total_env_steps} is less than one rollout of "
                f"ppo.rollout_steps={self.ppo.rollout_steps}; nothing would be learned"
            )
        if self.brain not in KNOWN_BRAINS:
            raise ValueError(f"unknown brain {self.brain!r}; choose from {list(KNOWN_BRAINS)}")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError(f"device must be a non-empty string, got {self.device!r}")

    # -- derived ------------------------------------------------------------------------

    @property
    def num_updates(self) -> int:
        """PPO updates per seed: whole rollouts that fit in ``total_env_steps``."""
        return self.total_env_steps // self.ppo.rollout_steps

    @property
    def run_root(self) -> Path:
        """``<output_dir>/<name>``, shared by every seed of the condition."""
        return Path(self.output_dir) / self.name

    def run_dir(self, seed: int) -> Path:
        """``<output_dir>/<name>/seed_<seed>``: one seed's logs and checkpoints."""
        return self.run_root / f"seed_{int(seed)}"

    def with_seeds(self, seeds: tuple[int, ...] | list[int]) -> TrainConfig:
        """A copy that runs only ``seeds``."""
        return replace(self, seeds=tuple(seeds))

    # -- serialisation --------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TrainConfig:
        """Build from a nested mapping; unknown keys at any level are an error."""
        return _from_mapping(cls, data, path="train config")

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> TrainConfig:
        """Load a YAML mapping (requires PyYAML)."""
        try:
            import yaml
        except ImportError as error:
            raise ImportError("TrainConfig.from_yaml needs PyYAML: pip install pyyaml") from error
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise TypeError(f"{path} must contain a mapping at the top level")
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """JSON- and YAML-safe nested mapping; inverse of :meth:`from_dict`."""
        return _yaml_safe(asdict(self))

    def to_yaml(self, path: str | os.PathLike[str]) -> Path:
        """Write the resolved config, so a run directory says exactly what ran."""
        try:
            import yaml
        except ImportError as error:
            raise ImportError("TrainConfig.to_yaml needs PyYAML: pip install pyyaml") from error
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        return target


# Register the nested fields after every class exists.
_NESTED[TrainConfig] = {
    "eye": EyeConfig,
    "policy": PolicyConfig,
    "env": EnvConfig,
    "ppo": PPOConfig,
    "eval": PeriodicEvalConfig,
    "logging": LoggingConfig,
}
