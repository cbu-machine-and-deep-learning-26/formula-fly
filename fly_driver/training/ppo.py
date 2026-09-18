"""Proximal policy optimisation, CleanRL-style, on stored rollouts (GH-17).

Three pieces. :class:`RolloutBuffer` holds one rollout -- features, raw actions, their
log-probabilities, rewards, episode ends and value estimates -- and turns them into
advantages with generalised advantage estimation. :class:`PPOLearner` owns the optimiser and
runs the clipped-surrogate update over minibatches. :class:`DifferentiableEye` is the
contract an eye must meet to be *fine-tuned* rather than frozen: a batch of frames in, a
batch of features out, with gradients.

Why the fine-tuned path needs its own contract: the :class:`~fly_driver.interface.Eye` the
loop rolls out with is numpy in, numpy out, one frame at a time, and may keep state between
frames (the flyvis optic lobe does). Training through it would mean backpropagating through
the whole episode. A feed-forward eye can instead re-encode the rollout's frames inside each
minibatch, which is what :meth:`PPOLearner.update` does when it is given one. Eyes that keep
state are refused for ``frozen: false`` rather than silently trained as if they were
feed-forward; see :mod:`fly_driver.training.train`.

Numerical guards, per `AGENTS.md` §11: the update raises on a non-finite loss instead of
stepping on it, and the tests pin the advantage recursion to a hand-computed case.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn

from fly_driver.policies.mlp_policy import ACTION_DIM, MlpPolicy
from fly_driver.training.config import PPOConfig

__all__ = ["DifferentiableEye", "PPOLearner", "RolloutBuffer", "UpdateStats"]


@runtime_checkable
class DifferentiableEye(Protocol):
    """An eye whose encoding is a stateless, differentiable torch function of frames.

    The rollout still uses ``encode`` (numpy, one frame) and the update uses
    :meth:`encode_batch`; the two must agree on the same frame, or the first minibatch's
    probability ratio is not 1 and the clip has nothing to protect.
    """

    def parameters(self) -> Iterator[nn.Parameter]:
        """The parameters the optimiser should own."""

    def encode_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """``(batch, height, width, 3)`` uint8 frames to ``(batch, feature_dim)`` float32."""


class RolloutBuffer:
    """One rollout of fixed length, filled step by step.

    Args:
        steps: Rollout length.
        feature_dim: Feature length stored per step.
        frame_shape: When given, frames are stored too (uint8), for a fine-tuned eye.

    ``episode_ends[t]`` is true when the episode ended *after* step ``t`` -- the natural
    reading -- so the advantage recursion never looks at the next row's flag.
    """

    def __init__(
        self, steps: int, feature_dim: int, *, frame_shape: tuple[int, int, int] | None = None
    ) -> None:
        if steps < 1 or feature_dim < 1:
            raise ValueError("steps and feature_dim must be positive")
        self.steps = int(steps)
        self.feature_dim = int(feature_dim)
        self.features = np.zeros((self.steps, self.feature_dim), dtype=np.float32)
        self.frames = (
            None if frame_shape is None else np.zeros((self.steps, *frame_shape), dtype=np.uint8)
        )
        self.raw_actions = np.zeros((self.steps, ACTION_DIM), dtype=np.float32)
        self.log_probs = np.zeros(self.steps, dtype=np.float32)
        self.rewards = np.zeros(self.steps, dtype=np.float32)
        self.episode_ends = np.zeros(self.steps, dtype=np.bool_)
        self.values = np.zeros(self.steps, dtype=np.float32)
        self.advantages = np.zeros(self.steps, dtype=np.float32)
        self.returns = np.zeros(self.steps, dtype=np.float32)
        self._filled = 0

    @property
    def filled(self) -> int:
        """Steps stored so far."""
        return self._filled

    def add(
        self,
        *,
        features: np.ndarray,
        raw_action: np.ndarray,
        log_prob: float,
        reward: float,
        episode_end: bool,
        value: float,
        frame: np.ndarray | None = None,
    ) -> None:
        """Store one step. Raises when the buffer is already full."""
        if self._filled >= self.steps:
            raise RuntimeError("rollout buffer is full; compute advantages and start a new one")
        index = self._filled
        self.features[index] = features
        self.raw_actions[index] = raw_action
        self.log_probs[index] = log_prob
        self.rewards[index] = reward
        self.episode_ends[index] = episode_end
        self.values[index] = value
        if self.frames is not None:
            if frame is None:
                raise ValueError("this buffer stores frames; add() needs one")
            self.frames[index] = frame
        self._filled += 1

    def reset(self) -> None:
        """Start filling from the beginning; the arrays are overwritten in place."""
        self._filled = 0

    def compute_advantages(self, last_value: float, *, gamma: float, gae_lambda: float) -> None:
        """Generalised advantage estimation over the filled steps.

        Args:
            last_value: The critic's estimate for the observation *after* the last step, used
                to bootstrap only if that step did not end an episode.
            gamma: Discount.
            gae_lambda: Trace decay.
        """
        if self._filled != self.steps:
            raise RuntimeError(f"buffer has {self._filled} of {self.steps} steps; fill it first")
        last_gae = 0.0
        for step in reversed(range(self.steps)):
            continuing = 0.0 if self.episode_ends[step] else 1.0
            next_value = last_value if step == self.steps - 1 else float(self.values[step + 1])
            delta = (
                float(self.rewards[step])
                + gamma * next_value * continuing
                - float(self.values[step])
            )
            last_gae = delta + gamma * gae_lambda * continuing * last_gae
            self.advantages[step] = last_gae
        self.returns[:] = self.advantages + self.values


@dataclass(frozen=True)
class UpdateStats:
    """What one :meth:`PPOLearner.update` did."""

    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    learning_rate: float
    epochs_run: int


class PPOLearner:
    """The clipped-surrogate update over a :class:`RolloutBuffer`.

    Args:
        policy: The actor-critic to train.
        config: Hyperparameters.
        eye: A :class:`DifferentiableEye` to fine-tune alongside the policy, or ``None`` to
            train the policy on the stored features alone (the frozen-eye default).
        device: Where the update runs.

    Raises:
        ValueError: If ``eye`` is given but has no parameters to train.
    """

    def __init__(
        self,
        policy: MlpPolicy,
        config: PPOConfig,
        *,
        eye: DifferentiableEye | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.policy = policy
        self.config = config
        self.eye = eye
        self.device = torch.device(device)
        parameters = list(policy.parameters())
        if eye is not None:
            eye_parameters = [p for p in eye.parameters() if p.requires_grad]
            if not eye_parameters:
                raise ValueError("eye has no trainable parameters; use frozen: true")
            parameters.extend(eye_parameters)
        self.optimizer = torch.optim.Adam(parameters, lr=config.learning_rate, eps=1e-5)

    def learning_rate_for(self, update_index: int, total_updates: int) -> float:
        """The step size for update ``update_index`` (from 1) of ``total_updates``."""
        if not self.config.anneal_learning_rate:
            return self.config.learning_rate
        if update_index < 1 or total_updates < 1 or update_index > total_updates:
            raise ValueError(f"update {update_index} of {total_updates} is out of range")
        fraction = 1.0 - (update_index - 1) / total_updates
        return fraction * self.config.learning_rate

    def update(
        self, buffer: RolloutBuffer, *, update_index: int, total_updates: int
    ) -> UpdateStats:
        """Run ``config.epochs`` passes of minibatch updates over the buffer.

        Raises:
            RuntimeError: If a loss is not finite; the parameters are left as they were.
        """
        config = self.config
        if buffer.filled != buffer.steps or buffer.steps != config.rollout_steps:
            raise ValueError(
                f"buffer holds {buffer.filled}/{buffer.steps} steps but the config expects "
                f"{config.rollout_steps} full steps per update"
            )
        if self.eye is not None and buffer.frames is None:
            raise ValueError("fine-tuning the eye needs a buffer that stores frames")

        learning_rate = self.learning_rate_for(update_index, total_updates)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

        features = torch.as_tensor(buffer.features, device=self.device)
        frames = (
            None if buffer.frames is None else torch.as_tensor(buffer.frames, device=self.device)
        )
        raw_actions = torch.as_tensor(buffer.raw_actions, device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs, device=self.device)
        advantages = torch.as_tensor(buffer.advantages, device=self.device)
        returns = torch.as_tensor(buffer.returns, device=self.device)
        old_values = torch.as_tensor(buffer.values, device=self.device)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        kls: list[float] = []
        clip_fractions: list[float] = []
        epochs_run = 0
        stop = False
        for _ in range(config.epochs):
            epochs_run += 1
            permutation = torch.randperm(buffer.steps, device=self.device)
            for start in range(0, buffer.steps, config.minibatch_size):
                index = permutation[start : start + config.minibatch_size]
                if self.eye is not None and frames is not None:
                    batch_features = self.eye.encode_batch(frames[index])
                else:
                    batch_features = features[index]
                log_prob, entropy, value = self.policy.evaluate_actions(
                    batch_features, raw_actions[index]
                )
                log_ratio = log_prob - old_log_probs[index]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = ((ratio - 1.0).abs() > config.clip_coef).float().mean()

                batch_advantages = advantages[index]
                if config.normalize_advantages and index.numel() > 1:
                    batch_advantages = (batch_advantages - batch_advantages.mean()) / (
                        batch_advantages.std() + 1e-8
                    )
                surrogate = -batch_advantages * ratio
                surrogate_clipped = -batch_advantages * torch.clamp(
                    ratio, 1.0 - config.clip_coef, 1.0 + config.clip_coef
                )
                policy_loss = torch.max(surrogate, surrogate_clipped).mean()

                if config.clip_value_loss:
                    value_clipped = old_values[index] + torch.clamp(
                        value - old_values[index], -config.clip_coef, config.clip_coef
                    )
                    value_loss = (
                        0.5
                        * torch.max(
                            (value - returns[index]).square(),
                            (value_clipped - returns[index]).square(),
                        ).mean()
                    )
                else:
                    value_loss = 0.5 * (value - returns[index]).square().mean()

                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    - config.entropy_coef * entropy_mean
                    + config.value_coef * value_loss
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"non-finite PPO loss at update {update_index}: policy "
                        f"{policy_loss.item()}, value {value_loss.item()}, entropy "
                        f"{entropy_mean.item()}; not stepping on it (AGENTS.md §11)"
                    )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for group in self.optimizer.param_groups for p in group["params"]],
                    config.max_grad_norm,
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_mean.item())
                kls.append(approx_kl.item())
                clip_fractions.append(clip_fraction.item())
            if config.target_kl is not None and kls and kls[-1] > config.target_kl:
                stop = True
            if stop:
                break

        explained = _explained_variance(buffer.values, buffer.returns)
        return UpdateStats(
            policy_loss=float(np.mean(policy_losses)),
            value_loss=float(np.mean(value_losses)),
            entropy=float(np.mean(entropies)),
            approx_kl=float(np.mean(kls)),
            clip_fraction=float(np.mean(clip_fractions)),
            explained_variance=explained,
            learning_rate=learning_rate,
            epochs_run=epochs_run,
        )


def _explained_variance(values: np.ndarray, returns: np.ndarray) -> float:
    """``1 - Var(returns - values) / Var(returns)``; NaN when the returns do not vary."""
    variance = float(np.var(returns))
    if variance == 0.0:
        return float("nan")
    return float(1.0 - np.var(returns - values) / variance)
