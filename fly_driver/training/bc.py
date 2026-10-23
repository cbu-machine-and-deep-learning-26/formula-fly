"""Behaviour cloning of the fly driver on human laps, warm from a MuJoCo checkpoint (GH-29).

The eye stays frozen and is never run here: :func:`~fly_driver.training.bc_data.cache_eye_features`
ran it once per drive. The brain and the policy's actor load from a ``policy_*.pt`` and the
``brain_*.pt`` beside it (:func:`~fly_driver.training.train.warm_start`, as ``init_from``
does) and learn to reproduce the human's ``(steer, throttle, brake)``.

**The brain is recurrent**, and is trained the way PPO trains it: one step of gradient
through :meth:`~fly_driver.brains.central_complex.CentralComplexBrain.transition` from a
stored membrane. Each epoch first runs the brain through every segment in order, without
gradients, to get the membrane before each frame under the current weights; minibatches
then step it once from those membranes, with gradients. The membranes go stale within an
epoch, exactly as PPO's do within an update.

**The loss** is the squared error of the action the env would receive, the clipped mean,
against the human's, but without the clip's flat gradient: past a bound, only the error
on the target's side of it counts. Brake at 0 with a mean of -0.6 costs nothing; a mean of
1.4 for a throttle of 0.5 still pulls back. ``log_std`` is not trained, so PPO starts with
the base checkpoint's exploration rather than a BC-shrunk one.

**The critic** is left as the base checkpoint had it. Human laps carry no rewards, so there
is nothing to fit it to; PPO refits it in its first updates.

**The normaliser** in front of the policy is refitted to the brain's outputs on the training
laps, once, with the base brain. The MuJoCo statistics describe another world's features.

Writes ``<output_dir>/<name>/``: ``policy_final.pt`` and ``brain_final.pt`` (the epoch with
the lowest held-out error), ``config.yaml`` (the base config with ``init_from`` cleared, so
:func:`~fly_driver.training.train.load_driver` rebuilds the driver) and ``bc_log.csv``.
Point a PPO config's ``init_from`` at ``policy_final.pt``.
"""

from __future__ import annotations

import csv
import dataclasses
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from fly_driver.brains.central_complex import CentralComplexBrain
from fly_driver.interface import FRAME_RATE_HZ, FRAME_SHAPE
from fly_driver.policies.mlp_policy import MlpPolicy
from fly_driver.training.bc_data import HumanLaps, cache_eye_features, eye_cache_key
from fly_driver.training.config import TrainConfig
from fly_driver.training.evaluation import seed_everything
from fly_driver.training.train import (
    FINAL_CHECKPOINT,
    build_brain,
    build_eye,
    build_policy,
    resolve_device,
    warm_start,
)

__all__ = ["BcConfig", "BcData", "BcEpoch", "action_loss", "load_bc_data", "train_bc"]

LOG_FILENAME = "bc_log.csv"
_ACTION_LOW = (-1.0, 0.0, 0.0)
_ACTION_HIGH = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class BcConfig:
    """One behaviour-cloning run; every YAML key maps to a field, unknown keys are an error.

    Attributes:
        name: Run name; the run lands in ``<output_dir>/<name>/``.
        base_config: A PPO run config (``TrainConfig`` YAML) that builds the eye, brain and
            policy. Use the config the cloned driver will fine-tune with, e.g. the AC one.
        init_from: The ``policy_*.pt`` to start from; its ``brain_*.pt`` sits beside it.
        datasets: :class:`~fly_driver.training.bc_data.HumanLaps` ``.npz`` files.
        holdout_laps: The last this-many laps, in dataset order, are held out.
        epochs: Passes over the training frames.
        batch_size: Frames per gradient step.
        learning_rate: Adam's step size, for the brain and the actor.
        max_grad_norm: Gradient-norm clip per step.
        train_brain: Train the brain's synapses with the actor; off trains the actor only.
        output_dir: Root for run directories.
        device: ``auto``, ``cpu``, ``cuda``, ...
        seed: Fixes initial shuffles and torch.
    """

    name: str
    base_config: str
    init_from: str
    datasets: tuple[str, ...]
    holdout_laps: int = 1
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 1e-4
    max_grad_norm: float = 0.5
    train_brain: bool = True
    output_dir: str = "runs"
    device: str = "auto"
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.datasets:
            raise ValueError("datasets must name at least one HumanLaps .npz")
        if self.holdout_laps < 1:
            raise ValueError("holdout_laps must be at least 1: the held-out error picks the epoch")
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size and learning_rate must be positive")

    @property
    def run_dir(self) -> Path:
        """``<output_dir>/<name>``."""
        return Path(self.output_dir) / self.name

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BcConfig:
        """Build from a mapping; unknown keys are an error."""
        known = {item.name for item in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown BC config keys {sorted(unknown)}; known: {sorted(known)}")
        values = dict(data)
        values["datasets"] = tuple(str(path) for path in values.get("datasets", ()))
        return cls(**values)

    @classmethod
    def from_yaml(cls, path: str | os.PathLike[str]) -> BcConfig:
        """Load a YAML mapping (requires PyYAML)."""
        import yaml

        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})


@dataclass(frozen=True)
class BcData:
    """Every frame's eye features and human action, in segment order, with the split.

    Attributes:
        eye_features: ``(n, eye_dim)`` float32.
        actions: ``(n, 3)`` float32 ``(steer, throttle, brake)``.
        segment_starts: ``(n,)`` bool, true where the brain resets.
        holdout: ``(n,)`` bool, true for held-out frames.
    """

    eye_features: np.ndarray
    actions: np.ndarray
    segment_starts: np.ndarray
    holdout: np.ndarray


def load_bc_data(
    datasets: Sequence[str | os.PathLike[str]],
    eye: Any,
    eye_config: Mapping[str, Any],
    holdout_laps: int,
) -> BcData:
    """Load drives, cache their eye features beside them, and hold out the last laps.

    Raises:
        ValueError: If fewer than ``holdout_laps + 1`` laps are given, so nothing would train.
    """
    key = eye_cache_key(eye_config)
    features, actions, starts, lap_ids = [], [], [], []
    for index, path in enumerate(datasets):
        laps = HumanLaps.load(path)
        features.append(cache_eye_features(laps, eye, Path(path).with_suffix(f".eye_{key}.npz")))
        actions.append(laps.controls)
        segment_start = np.zeros(len(laps), bool)
        segment_start[[start for start, _ in laps.segment_bounds()]] = True
        starts.append(segment_start)
        lap_ids.extend((index, int(lap)) for lap in laps.laps)
    ordered = list(dict.fromkeys(lap_ids))
    if len(ordered) <= holdout_laps:
        raise ValueError(
            f"{len(ordered)} laps cannot hold out {holdout_laps} and still train on one"
        )
    held = set(ordered[-holdout_laps:])
    return BcData(
        eye_features=np.concatenate(features),
        actions=np.concatenate(actions),
        segment_starts=np.concatenate(starts),
        holdout=np.array([lap in held for lap in lap_ids]),
    )


def action_loss(mean: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error of the clipped action, keeping gradients past the bounds.

    Past a bound, an error that only pushes the mean further past it is zero (the env would
    clip it to the target anyway); any other error counts in full.
    """
    low = target.new_tensor(_ACTION_LOW)
    high = target.new_tensor(_ACTION_HIGH)
    error = mean - target
    clipped_away = ((target <= low) & (error < 0)) | ((target >= high) & (error > 0))
    return torch.where(clipped_away, torch.zeros_like(error), error).square().mean()


@dataclass(frozen=True)
class BcEpoch:
    """One epoch's numbers; ``*_mae`` is per ``(steer, throttle, brake)`` of the clipped mean."""

    epoch: int
    train_loss: float
    train_mae: tuple[float, float, float]
    holdout_mae: tuple[float, float, float]
    holdout_loss: float
    is_best: bool = field(default=False)


@torch.no_grad()
def _unroll(
    brain: CentralComplexBrain, eye_features: torch.Tensor, segment_starts: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    """Membrane before each frame and the brain's output after it, segment by segment."""
    states = eye_features.new_zeros((len(eye_features), brain.neuron_count))
    outputs = eye_features.new_zeros((len(eye_features), brain.output_dim))
    voltage = eye_features.new_zeros(brain.neuron_count)
    for index in range(len(eye_features)):
        if segment_starts[index]:
            voltage = voltage.new_zeros(brain.neuron_count)
        states[index] = voltage
        voltage, outputs[index] = brain.transition(voltage, eye_features[index])
    return states, outputs


def _refit_normalizer(policy: MlpPolicy, features: torch.Tensor) -> None:
    if policy.normalizer is None:
        return
    with torch.no_grad():
        policy.normalizer.mean.zero_()
        policy.normalizer.var.fill_(1.0)
        policy.normalizer.count.fill_(1e-4)
    policy.normalizer.update(features)


@torch.no_grad()
def _score(
    policy: MlpPolicy, features: torch.Tensor, actions: torch.Tensor
) -> tuple[float, tuple[float, float, float]]:
    mean, _ = policy(features)
    loss = float(action_loss(mean, actions))
    clipped = torch.maximum(
        torch.minimum(mean, mean.new_tensor(_ACTION_HIGH)), mean.new_tensor(_ACTION_LOW)
    )
    mae = (clipped - actions).abs().mean(0).tolist()
    return loss, (mae[0], mae[1], mae[2])


def train_bc(config: BcConfig, *, overwrite: bool = False, log: Any = print) -> list[BcEpoch]:
    """Clone the human laps into the warm-started brain and actor; see the module docstring.

    Raises:
        FileExistsError: If the run directory exists and ``overwrite`` is off.
    """
    run_dir = config.run_dir
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} exists; pass overwrite=True or choose another name")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    seed_everything(config.seed)
    device = resolve_device(config.device)

    base = TrainConfig.from_yaml(config.base_config)
    dataclasses.replace(base, init_from=None).to_yaml(run_dir / "config.yaml")
    frames = SimpleNamespace(frame_shape=FRAME_SHAPE, frame_rate_hz=FRAME_RATE_HZ)
    eye = build_eye(base.eye, frames)
    brain = build_brain(base.brain, eye, frames)
    if brain is None:
        raise ValueError("BC here trains a brain and a policy; the base config has no brain")
    policy = build_policy(base.policy, brain.output_dim)
    warm_start(Path(config.init_from), policy, brain)
    brain.to(device)
    policy.to(device)
    brain.requires_grad_(config.train_brain)

    data = load_bc_data(config.datasets, eye, dataclasses.asdict(base.eye), config.holdout_laps)
    eye_features = torch.as_tensor(data.eye_features, device=device)
    actions = torch.as_tensor(data.actions, device=device)
    train_index = torch.as_tensor(np.flatnonzero(~data.holdout), device=device)
    held_index = torch.as_tensor(np.flatnonzero(data.holdout), device=device)
    log(f"{len(train_index)} training frames, {len(held_index)} held out, on {device}")

    parameters = list(policy.actor.parameters())
    if config.train_brain:
        parameters += list(brain.parameters())
    optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    states, outputs = _unroll(brain, eye_features, data.segment_starts)
    _refit_normalizer(policy, outputs[train_index])
    held_loss, held_mae = _score(policy, outputs[held_index], actions[held_index])
    log(f"base checkpoint: held-out loss {held_loss:.4f}, MAE {np.round(held_mae, 3).tolist()}")
    best = held_loss
    _save(policy, brain, run_dir)

    history: list[BcEpoch] = []
    with (run_dir / LOG_FILENAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["epoch", "train_loss", "holdout_loss"]
            + [
                f"{split}_mae_{name}"
                for split in ("train", "holdout")
                for name in ("steer", "throttle", "brake")
            ]
            + ["is_best"]
        )
        writer.writerow([0, "", held_loss, "", "", "", *held_mae, True])
        for epoch in range(1, config.epochs + 1):
            order = train_index[torch.randperm(len(train_index), generator=generator).to(device)]
            losses = []
            for batch in order.split(config.batch_size):
                if config.train_brain:
                    _, features = brain.transition(states[batch], eye_features[batch])
                else:
                    features = outputs[batch]
                mean, _ = policy(features)
                loss = action_loss(mean, actions[batch])
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
                optimizer.step()
                losses.append(float(loss.detach()))
            states, outputs = _unroll(brain, eye_features, data.segment_starts)
            _, train_mae = _score(policy, outputs[train_index], actions[train_index])
            held_loss, held_mae = _score(policy, outputs[held_index], actions[held_index])
            is_best = held_loss < best
            if is_best:
                best = held_loss
                _save(policy, brain, run_dir)
            record = BcEpoch(epoch, float(np.mean(losses)), train_mae, held_mae, held_loss, is_best)
            history.append(record)
            writer.writerow([epoch, record.train_loss, held_loss, *train_mae, *held_mae, is_best])
            handle.flush()
            log(
                f"epoch {epoch}: train loss {record.train_loss:.4f}, held-out loss "
                f"{held_loss:.4f}, held-out MAE {np.round(held_mae, 3).tolist()}"
                + (" (best)" if is_best else "")
            )
    return history


def _save(policy: MlpPolicy, brain: CentralComplexBrain, run_dir: Path) -> None:
    policy.save(run_dir / FINAL_CHECKPOINT, extra={"trained_by": "behaviour cloning (GH-29)"})
    torch.save(
        {key: value.detach().cpu() for key, value in brain.state_dict().items()},
        run_dir / FINAL_CHECKPOINT.replace("policy_", "brain_", 1),
    )
