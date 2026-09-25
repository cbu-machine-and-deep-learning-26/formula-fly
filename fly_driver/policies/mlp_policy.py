"""The small policy head: eye features in, a Gaussian over the control out (GH-17).

This is the trainable readout `AGENTS.md` §2 and §6 talk about -- with a frozen eye it is
the *only* thing that learns. Two MLPs of the same shape, an actor for the action mean and
a critic for the value, plus a state-independent log standard deviation, as in CleanRL's
continuous-control PPO.

The action is Gaussian over an unbounded *raw* control and the env receives
:meth:`~fly_driver.interface.ControlVector.clipped` of a sample. The clip is the interface's
explicit squash, and the log-probability is of the raw sample, which is the standard
clipped-Gaussian treatment. :meth:`MlpPolicy.act` -- the :class:`~fly_driver.interface.Policy`
protocol, what the evaluation harness and a :class:`~fly_driver.drivers.DirectDriveAgent`
call -- returns the clipped *mean*, so evaluation is deterministic while training samples.

Feature normalisation lives inside the module (:class:`RunningNormalizer`) so its statistics
travel with the checkpoint: a policy evaluated with different statistics from the ones it
trained under would see scrambled inputs and never say so.

Imports torch at module scope, so ``fly_driver.policies`` does not import this file; take
it by its full path.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from fly_driver.interface import ControlVector, Features, validate_features

__all__ = ["ACTION_DIM", "ACTIVATIONS", "MlpPolicy", "RunningNormalizer"]

#: ``(steer, throttle, brake)``.
ACTION_DIM = 3

ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU}


class RunningNormalizer(nn.Module):
    """Running mean and variance of the features, applied as ``(x - mean) / std``.

    Args:
        dim: Feature length.
        epsilon: Pseudo-count the statistics start from, so the first batch does not divide
            by a variance of zero.
        clip: Normalised values are clipped to ``[-clip, clip]``.
    """

    mean: torch.Tensor
    var: torch.Tensor
    count: torch.Tensor

    def __init__(self, dim: int, *, epsilon: float = 1e-4, clip: float = 10.0) -> None:
        super().__init__()
        self.clip = float(clip)
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float32))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float32))
        self.register_buffer("count", torch.tensor(float(epsilon), dtype=torch.float64))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Standardise ``(batch, dim)`` features with the current statistics."""
        scaled = (features - self.mean) / torch.sqrt(self.var + 1e-8)
        return torch.clamp(scaled, -self.clip, self.clip)

    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """Fold a ``(batch, dim)`` batch into the statistics (parallel-variance merge)."""
        if batch.ndim != 2 or batch.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"normalizer expects (batch, {self.mean.shape[0]}) features, got "
                f"{tuple(batch.shape)}"
            )
        batch = batch.to(self.mean.device, dtype=torch.float32)
        batch_count = float(batch.shape[0])
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, unbiased=False)
        total = float(self.count) + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total)
        m2 = (
            self.var * float(self.count)
            + batch_var * batch_count
            + delta.square() * (float(self.count) * batch_count / total)
        )
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.fill_(total)


def _orthogonal(layer: nn.Linear, std: float, bias: float = 0.0) -> nn.Linear:
    """CleanRL's initialisation: orthogonal weights, constant bias."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


def _mlp(
    input_dim: int, hidden_sizes: Sequence[int], output_dim: int, activation: str, last_std: float
) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = input_dim
    for size in hidden_sizes:
        layers.append(_orthogonal(nn.Linear(width, size), std=float(np.sqrt(2))))
        layers.append(ACTIVATIONS[activation]())
        width = size
    layers.append(_orthogonal(nn.Linear(width, output_dim), std=last_std))
    return nn.Sequential(*layers)


class MlpPolicy(nn.Module):
    """Actor-critic MLP over a feature vector; implements the ``Policy`` protocol.

    Args:
        feature_dim: Length of the feature vector the eye (or brain) emits.
        hidden_sizes: Hidden widths shared by actor and critic. Empty means linear.
        activation: ``"tanh"`` or ``"relu"``.
        normalize_features: Standardise inputs with a :class:`RunningNormalizer`.
        init_log_std: Initial log standard deviation of the raw action Gaussian.

    Raises:
        ValueError: On a non-positive ``feature_dim`` or an unknown activation.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: str = "tanh",
        normalize_features: bool = True,
        init_log_std: float = 0.0,
    ) -> None:
        super().__init__()
        if int(feature_dim) < 1:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; choose from {list(ACTIVATIONS)}")
        self._feature_dim = int(feature_dim)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.activation = activation
        self.init_log_std = float(init_log_std)
        self.normalizer = RunningNormalizer(self._feature_dim) if normalize_features else None
        self.actor = _mlp(self._feature_dim, self.hidden_sizes, ACTION_DIM, activation, 0.01)
        self.critic = _mlp(self._feature_dim, self.hidden_sizes, 1, activation, 1.0)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), self.init_log_std))

    # -- the Policy protocol ------------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        """Length of the feature vector :meth:`act` expects."""
        return self._feature_dim

    @property
    def device(self) -> torch.device:
        """Where the parameters live."""
        return self.log_std.device

    def reset(self, seed: int | None = None) -> None:
        """Nothing to reset: :meth:`act` is deterministic, so the seed changes nothing."""
        del seed

    def act(self, features: Features) -> ControlVector:
        """The clipped mean action for one feature vector: deterministic, for evaluation."""
        validate_features(features, self._feature_dim)
        with torch.no_grad():
            batch = torch.as_tensor(features, device=self.device)[None]
            mean, _ = self(batch)
        steer, throttle, brake = mean[0].tolist()
        return ControlVector.clipped(steer, throttle, brake)

    # -- the training surface -----------------------------------------------------------

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action mean (batch, 3), value (batch,))`` for ``(batch, dim)`` features."""
        if self.normalizer is not None:
            features = self.normalizer(features)
        return self.actor(features), self.critic(features).squeeze(-1)

    def distribution(self, features: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        """Return the raw-action Gaussian and the value for a batch of features."""
        mean, value = self(features)
        return Normal(mean, self.log_std.exp().expand_as(mean)), value

    @torch.no_grad()
    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample raw actions: ``(raw (batch, 3), log_prob (batch,), value (batch,))``.

        Uses torch's global generator, so ``torch.manual_seed`` fixes the whole rollout.
        """
        dist, value = self.distribution(features)
        raw = dist.sample()
        return raw, dist.log_prob(raw).sum(-1), value

    def evaluate_actions(
        self, features: torch.Tensor, raw_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_prob, entropy, value)`` of stored raw actions, with gradients."""
        dist, value = self.distribution(features)
        return dist.log_prob(raw_actions).sum(-1), dist.entropy().sum(-1), value

    @torch.no_grad()
    def value(self, features: torch.Tensor) -> torch.Tensor:
        """The critic's estimate for a batch of features."""
        return self(features)[1]

    def update_normalizer(self, features: torch.Tensor) -> None:
        """Fold a batch of raw features into the running statistics, if normalising."""
        if self.normalizer is not None:
            self.normalizer.update(features)

    # -- checkpoints -------------------------------------------------------------------

    def save(self, path: str | os.PathLike[str], *, extra: Mapping[str, Any] | None = None) -> Path:
        """Write weights, shape and normaliser statistics; ``extra`` rides along."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "fly_driver.MlpPolicy/1",
            "feature_dim": self._feature_dim,
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.activation,
            "normalize_features": self.normalizer is not None,
            "init_log_std": self.init_log_std,
            "state_dict": {key: value.detach().cpu() for key, value in self.state_dict().items()},
            "extra": dict(extra or {}),
        }
        torch.save(payload, target)
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, device: str | torch.device = "cpu") -> MlpPolicy:
        """Rebuild a policy saved by :meth:`save`; ``policy.extra`` holds what rode along."""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != "fly_driver.MlpPolicy/1":
            raise ValueError(f"{path} is not an MlpPolicy checkpoint")
        policy = cls(
            feature_dim=int(payload["feature_dim"]),
            hidden_sizes=tuple(payload["hidden_sizes"]),
            activation=str(payload["activation"]),
            normalize_features=bool(payload["normalize_features"]),
            init_log_std=float(payload["init_log_std"]),
        )
        policy.load_state_dict(payload["state_dict"])
        policy.extra: dict[str, Any] = dict(payload.get("extra", {}))
        return policy.to(device)
