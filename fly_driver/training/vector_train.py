"""The training loop over many tracks at once (GH-27): ``num_envs > 1``.

Same condition, same PPO, same files as :mod:`fly_driver.training.train`; only the rollout
differs. Every step, ``num_envs`` worker processes (:mod:`fly_driver.training.vector_env`)
render and simulate in parallel, then the eye, the brain and the policy each run once on the
whole batch. The single-track loop runs the eye, brain and policy at batch size 1, which
leaves a 4090 or 5090 mostly idle; batching is what lets one trainer use the card.

What stays the same, so every tool that reads a run keeps working:

- ``episodes.csv``, ``updates.csv``, ``parameters.json``, ``policy_update_<n>.pt`` /
  ``brain_update_<n>.pt`` checkpoints and periodic evaluation, all written by the same code.
  The lab supervisor's 50-drive windows read ``episodes.csv`` in the order drives finished.
- ``ppo.rollout_steps`` is the steps per update, summed over the tracks: 2048 over 16 tracks
  is 128 steps each. Raise it with ``num_envs`` to keep each track's stretch long.
- Each track has its own advantage recursion; nothing leaks from one track into another.
- An episode cut off by the step cap bootstraps from the frame the cap cut off, as before.
- A reset puts that track's eye and brain back to rest; the others carry on.

Reproducibility: one generator seeded with ``seed`` draws every reset seed in the main
process, in track order. Physics and the eye are deterministic for a given action sequence.
Rendering is not quite (see :mod:`fly_driver.envs.practice_track`), and neither are CUDA
kernels, so compare returns, not bytes, across machines.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fly_driver.brains.central_complex import CentralComplexBrain
from fly_driver.eyes.batched import BatchEye, build_batch_eye
from fly_driver.interface import ControlVector
from fly_driver.policies.mlp_policy import MlpPolicy
from fly_driver.training.config import TrainConfig
from fly_driver.training.loggers import RunLogger, TrainingEpisode, UpdateRecord
from fly_driver.training.ppo import PPOLearner, RolloutBuffer
from fly_driver.training.teacher import OFF_TARGET
from fly_driver.training.vector_env import SubprocVectorEnv, SyncVectorEnv, VectorEnv

__all__ = ["VectorRollout", "build_vector_env", "run_vectorized"]


def build_vector_env(config: TrainConfig, *, in_process: bool = False) -> VectorEnv:
    """The training tracks for ``config``: worker processes, or one process for tests."""
    if in_process:
        from fly_driver.training.env_factory import build_env

        return SyncVectorEnv(lambda: build_env(config.env, training=True), config.num_envs)
    return SubprocVectorEnv(config.env, config.num_envs)


class VectorRollout:
    """``(steps, num_envs)`` storage that flattens into a :class:`RolloutBuffer`.

    Args:
        steps: Steps per track per rollout.
        num_envs: Tracks.
        feature_dim: What the policy reads.
        frame_shape: Store frames too, for a fine-tuned eye.
        brain_dims: ``(input_dim, neuron_count)`` to store the brain's inputs and membranes.
    """

    def __init__(
        self,
        steps: int,
        num_envs: int,
        feature_dim: int,
        *,
        frame_shape: tuple[int, int, int] | None = None,
        brain_dims: tuple[int, int] | None = None,
    ) -> None:
        self.steps = steps
        self.num_envs = num_envs
        shape = (steps, num_envs)
        self.features = np.zeros((*shape, feature_dim), dtype=np.float32)
        self.frames = None if frame_shape is None else np.zeros((*shape, *frame_shape), np.uint8)
        self.brain_inputs = (
            None if brain_dims is None else np.zeros((*shape, brain_dims[0]), np.float32)
        )
        self.brain_states = (
            None if brain_dims is None else np.zeros((*shape, brain_dims[1]), np.float32)
        )
        self.raw_actions = np.zeros((*shape, 3), dtype=np.float32)
        self.teacher_targets = np.zeros((*shape, 3), dtype=np.float32)
        self.policy_mask = np.ones(shape, dtype=np.bool_)
        self.teacher_weights = np.ones(shape, dtype=np.float32)
        self.log_probs = np.zeros(shape, dtype=np.float32)
        self.rewards = np.zeros(shape, dtype=np.float32)
        self.episode_ends = np.zeros(shape, dtype=np.bool_)
        self.values = np.zeros(shape, dtype=np.float32)
        self.advantages = np.zeros(shape, dtype=np.float32)
        self.returns = np.zeros(shape, dtype=np.float32)

    def compute_advantages(
        self, last_values: np.ndarray, *, gamma: float, gae_lambda: float
    ) -> None:
        """GAE down each track's column, exactly :meth:`RolloutBuffer.compute_advantages`.

        Args:
            last_values: ``(num_envs,)`` critic estimates after each track's last step, used
                only where that step did not end an episode.
            gamma: Discount.
            gae_lambda: Trace decay.
        """
        last_gae = np.zeros(self.num_envs, dtype=np.float64)
        next_values = np.asarray(last_values, dtype=np.float64)
        for step in reversed(range(self.steps)):
            continuing = 1.0 - self.episode_ends[step].astype(np.float64)
            values = self.values[step].astype(np.float64)
            delta = self.rewards[step] + gamma * next_values * continuing - values
            last_gae = delta + gamma * gae_lambda * continuing * last_gae
            self.advantages[step] = last_gae
            next_values = values
        self.returns[:] = self.advantages + self.values

    def flatten(self) -> RolloutBuffer:
        """A full :class:`RolloutBuffer` of ``steps * num_envs`` rows, advantages included.

        Rows are track-major (all of track 0, then track 1...), which keeps each track's
        steps contiguous; PPO shuffles minibatches anyway.
        """
        total = self.steps * self.num_envs
        frame_shape = None if self.frames is None else tuple(self.frames.shape[2:])
        brain_dims = (
            None
            if self.brain_inputs is None or self.brain_states is None
            else (self.brain_inputs.shape[2], self.brain_states.shape[2])
        )
        buffer = RolloutBuffer(
            total,
            self.features.shape[2],
            frame_shape=frame_shape,  # type: ignore[arg-type]
            brain_dims=brain_dims,
        )

        def flat(array: np.ndarray) -> np.ndarray:
            return np.ascontiguousarray(array.swapaxes(0, 1).reshape(total, *array.shape[2:]))

        buffer.features[:] = flat(self.features)
        buffer.raw_actions[:] = flat(self.raw_actions)
        buffer.teacher_targets[:] = flat(self.teacher_targets)
        buffer.policy_mask[:] = flat(self.policy_mask)
        buffer.teacher_weights[:] = flat(self.teacher_weights)
        buffer.log_probs[:] = flat(self.log_probs)
        buffer.rewards[:] = flat(self.rewards)
        buffer.episode_ends[:] = flat(self.episode_ends)
        buffer.values[:] = flat(self.values)
        buffer.advantages[:] = flat(self.advantages)
        buffer.returns[:] = flat(self.returns)
        if buffer.frames is not None and self.frames is not None:
            buffer.frames[:] = flat(self.frames)
        if buffer.brain_inputs is not None and self.brain_inputs is not None:
            buffer.brain_inputs[:] = flat(self.brain_inputs)
        if buffer.brain_states is not None and self.brain_states is not None:
            buffer.brain_states[:] = flat(self.brain_states)
        buffer._filled = total  # noqa: SLF001  filled in bulk rather than through add()
        return buffer


class _BatchBrain:
    """A membrane per track, stepped through the shared brain's batched transition."""

    def __init__(self, brain: CentralComplexBrain, num_envs: int, device: torch.device) -> None:
        self.brain = brain
        self.voltage = torch.zeros(num_envs, brain.neuron_count, device=device)

    def reset(self, indices: list[int]) -> None:
        if indices:
            self.voltage[indices] = 0.0

    def state(self) -> np.ndarray:
        return self.voltage.detach().cpu().numpy().astype(np.float32, copy=True)

    @torch.no_grad()
    def step(self, eye_features: torch.Tensor, indices: list[int] | None = None) -> torch.Tensor:
        voltage = self.voltage if indices is None else self.voltage[indices]
        updated, output = self.brain.transition(voltage, eye_features)
        if indices is None:
            self.voltage.copy_(updated)
        else:
            self.voltage[indices] = updated
        return output

    @torch.no_grad()
    def replay(self, state: np.ndarray, eye_features: torch.Tensor) -> torch.Tensor:
        """Step again from a stored membrane, with the brain's current weights."""
        voltage = torch.as_tensor(state, device=self.voltage.device)
        updated, output = self.brain.transition(voltage, eye_features)
        self.voltage.copy_(updated)
        return output


def run_vectorized(
    config: TrainConfig,
    seed: int,
    *,
    vector_env: VectorEnv,
    eye: Any,
    brain: CentralComplexBrain | None,
    policy: MlpPolicy,
    learner: PPOLearner,
    loggers: RunLogger,
    run_dir: Path,
    device: torch.device,
    progress: Any,
    batch_eye: BatchEye | None = None,
) -> Any:
    """Train one seed on ``vector_env``. Called by :func:`fly_driver.training.train.train_seed`.

    Args:
        eye: The run's eye, built for the vector env's frames. A flyvis eye lends its network
            to :class:`~fly_driver.eyes.batched.BatchedFlyvisEye`; a stateless eye serves
            every track, and is the one ``learner`` fine-tunes when the eye is not frozen.

    Returns:
        The :class:`~fly_driver.training.train.SeedResult`.
    """
    # Imported here: train.py imports this module lazily, and these are its helpers.
    from fly_driver.training.train import (
        FINAL_CHECKPOINT,
        SeedResult,
        _EpisodeTally,
        _evaluate,
        _reward_terms,
        _sync_eval_eye,
        build_eye,
    )

    ppo = config.ppo
    num_envs = vector_env.num_envs
    per_env_steps = ppo.rollout_steps // num_envs
    frame_shape = tuple(vector_env.frame_shape)
    frame_rate_hz = float(vector_env.frame_rate_hz)
    fine_tuned = learner.eye is not None
    if batch_eye is None:
        batch_eye = build_batch_eye(config.eye, eye, vector_env, num_envs)
    if int(batch_eye.feature_dim) != int(eye.feature_dim):
        raise ValueError(
            f"batched eye emits {batch_eye.feature_dim} features, eye {eye.feature_dim}"
        )
    feature_dim = int(policy.feature_dim)
    batch_brain = None if brain is None else _BatchBrain(brain, num_envs, device)
    rollout = VectorRollout(
        per_env_steps,
        num_envs,
        feature_dim,
        frame_shape=frame_shape if fine_tuned else None,  # type: ignore[arg-type]
        brain_dims=None if brain is None else (brain.input_dim, brain.neuron_count),
    )
    episode_seeds = np.random.default_rng(seed)
    # A share of training episodes is driven by the teacher itself, so its passes through
    # corners the policy never survives (T3, Nov 5) enter the data the teacher loss sees.
    drive_share = float((config.env.params.get("teacher_drive") or {}).get("share", 0.0))
    if not 0.0 <= drive_share < 1.0:
        raise ValueError(f"env.params.teacher_drive.share must be in [0, 1), got {drive_share}")
    drive_rng = np.random.default_rng(seed + 7919)
    # Braking is rare in a rollout, so its teacher error is averaged away against PPO's pull
    # toward speed (T3's last corner, Nov 5): weigh the steps where the teacher brakes.
    brake_weight = float((config.env.params.get("teacher_weights") or {}).get("brake", 1.0))
    if brake_weight <= 0:
        raise ValueError(f"env.params.teacher_weights.brake must be positive, got {brake_weight}")
    teacher_driven = np.zeros(num_envs, dtype=np.bool_)

    def draw_driver(index: int) -> None:
        teacher_driven[index] = drive_share > 0 and drive_rng.random() < drive_share
        tallies[index].teacher_driven = bool(teacher_driven[index])

    teacher = None
    if ppo.uses_teacher or drive_share > 0:
        from fly_driver.training.teacher import DrivingTeacher

        teacher = DrivingTeacher.for_centerline(
            fraction=ppo.teacher_speed_fraction, lookahead_s=ppo.teacher_lookahead_s
        )
    # Each track's car state before its next action: what the teacher labels.
    car_state = np.zeros((4, num_envs))  # arclength, speed, lateral, heading error

    def remember_car(index: int, info: dict[str, Any] | None) -> None:
        info = info or {}
        for row, key in enumerate(("arclength_m", "speed_mps", "lateral_m", "heading_error_rad")):
            car_state[row, index] = float(info.get(key) or 0.0)

    def draw_seed() -> int:
        return int(episode_seeds.integers(0, 2**31 - 1))

    def observe(
        frames: np.ndarray, indices: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray | None, torch.Tensor]:
        """Eye features, the membranes before this frame, and what the policy reads."""
        eye_features = batch_eye.encode(frames, indices)
        if not np.all(np.isfinite(eye_features)):
            raise ValueError("the eye produced non-finite features (AGENTS.md §11)")
        if batch_brain is None:
            return eye_features, None, torch.as_tensor(eye_features, device=device)
        state = batch_brain.state() if indices is None else batch_brain.state()[indices]
        output = batch_brain.step(torch.as_tensor(eye_features, device=device), indices)
        return eye_features, state, output

    started = time.perf_counter()
    env_steps = 0
    episode_index = 0
    eval_eye: Any = None
    eval_reports: list[Any] = []
    last_record: UpdateRecord | None = None

    vector_env.set_training_steps(0)
    reset_infos = vector_env.reset([draw_seed() for _ in range(num_envs)])
    pending_seeds = [draw_seed() for _ in range(num_envs)]
    batch_eye.reset(list(range(num_envs)))
    if batch_brain is not None:
        batch_brain.reset(list(range(num_envs)))
    tallies = [_new_tally(_EpisodeTally, frame_rate_hz, info) for info in reset_infos]
    for index, info in enumerate(reset_infos):
        remember_car(index, info)
        draw_driver(index)
    frames = vector_env.frames.copy()
    eye_features, brain_state, features = observe(frames)

    for update in range(1, config.num_updates + 1):
        update_started = time.perf_counter()
        vector_env.set_training_steps(env_steps)
        finished: list[TrainingEpisode] = []
        controls: list[np.ndarray] = []
        for step in range(per_env_steps):
            raw, log_prob, value = policy.sample(features)
            raw_actions = raw.cpu().numpy()
            targets = None if teacher is None else teacher.targets(*car_state)
            if targets is not None and teacher_driven.any():
                raw_actions = np.where(teacher_driven[:, None], targets, raw_actions)
            actions = np.stack(
                [ControlVector.clipped(*row.tolist()).to_array() for row in raw_actions]
            )
            controls.append(np.column_stack([actions[:, 1], actions[:, 2], np.abs(actions[:, 0])]))
            result = vector_env.step(actions, pending_seeds)
            env_steps += num_envs

            stored_rewards = result.rewards.astype(np.float64).copy()
            ended = result.terminated | result.truncated
            if result.final_frames:
                # Bootstrap a capped episode from the frame the cap cut off, through that
                # track's eye and brain before they go back to rest.
                capped = sorted(result.final_frames)
                capped_frames = np.stack([result.final_frames[index] for index in capped])
                capped_features = observe(capped_frames, capped)[2]
                capped_values = policy.value(capped_features).cpu().numpy()
                stored_rewards[capped] += ppo.gamma * capped_values

            rollout.features[step] = features.cpu().numpy()
            rollout.raw_actions[step] = raw_actions
            rollout.policy_mask[step] = ~teacher_driven
            if targets is not None:
                rollout.teacher_targets[step] = targets
                braking = targets[:, 2] > OFF_TARGET
                rollout.teacher_weights[step] = np.where(braking, brake_weight, 1.0)
            rollout.log_probs[step] = log_prob.cpu().numpy()
            rollout.rewards[step] = stored_rewards
            rollout.episode_ends[step] = ended
            rollout.values[step] = value.cpu().numpy()
            if rollout.frames is not None:
                rollout.frames[step] = frames
            if rollout.brain_inputs is not None and rollout.brain_states is not None:
                rollout.brain_inputs[step] = eye_features
                rollout.brain_states[step] = brain_state

            done_indices = [int(index) for index in np.flatnonzero(ended)]
            for index in range(num_envs):
                info = result.infos[index]
                reward = float(result.rewards[index])
                tallies[index].add(reward, _reward_terms(info, reward), info)
                remember_car(index, result.reset_infos[index] if ended[index] else info)
            for index in done_indices:
                record = tallies[index].finish(
                    seed=seed,
                    episode=episode_index,
                    update=update,
                    env_steps=env_steps,
                    terminated=bool(result.terminated[index]),
                    truncated=bool(result.truncated[index]),
                )
                loggers.log_episode(record)
                finished.append(record)
                episode_index += 1
                reset_info = result.reset_infos[index] or {}
                tallies[index] = _new_tally(_EpisodeTally, frame_rate_hz, reset_info)
                draw_driver(index)
                pending_seeds[index] = draw_seed()
            if done_indices:
                batch_eye.reset(done_indices)
                if batch_brain is not None:
                    batch_brain.reset(done_indices)
            frames = vector_env.frames.copy()
            eye_features, brain_state, features = observe(frames)

        last_values = policy.value(features).cpu().numpy()
        last_values = np.where(rollout.episode_ends[-1], 0.0, last_values)
        rollout.compute_advantages(last_values, gamma=ppo.gamma, gae_lambda=ppo.gae_lambda)
        buffer = rollout.flatten()
        stats = learner.update(buffer, update_index=update, total_updates=config.num_updates)
        policy.update_normalizer(torch.as_tensor(buffer.features, device=device))
        if batch_brain is not None and brain_state is not None:
            # The next rollout's first frames went through the brain before this update.
            features = batch_brain.replay(brain_state, torch.as_tensor(eye_features, device=device))

        now = time.perf_counter()
        mean_throttle, mean_brake, mean_abs_steer = np.concatenate(controls).mean(axis=0).tolist()
        last_record = UpdateRecord(
            seed=seed,
            update=update,
            env_steps=env_steps,
            episodes_completed=len(finished),
            mean_episode_return=(
                float(np.mean([item.total_return for item in finished])) if finished else None
            ),
            mean_episode_steps=(
                float(np.mean([item.steps for item in finished])) if finished else None
            ),
            policy_loss=stats.policy_loss,
            value_loss=stats.value_loss,
            entropy=stats.entropy,
            approx_kl=stats.approx_kl,
            clip_fraction=stats.clip_fraction,
            explained_variance=stats.explained_variance,
            learning_rate=stats.learning_rate,
            steps_per_second=ppo.rollout_steps / max(now - update_started, 1e-9),
            elapsed_s=now - started,
            mean_throttle=mean_throttle,
            mean_brake=mean_brake,
            mean_abs_steer=mean_abs_steer,
            braking_line_fade=None,
        )
        loggers.log_update(last_record)
        if progress is not None:
            progress(last_record)

        every_checkpoint = config.logging.checkpoint_every_updates
        if every_checkpoint and update % every_checkpoint == 0:
            policy.save(
                run_dir / f"policy_update_{update:05d}.pt",
                extra={"seed": seed, "env_steps": env_steps, "update": update},
            )
            if brain is not None:
                torch.save(brain.state_dict(), run_dir / f"brain_update_{update:05d}.pt")
        every_eval = config.eval.every_updates
        if every_eval and update % every_eval == 0:
            if eval_eye is None:
                eval_eye = build_eye(config.eye, vector_env)
            if learner.eye is not None:
                _sync_eval_eye(learner.eye, eval_eye, fine_tuned=True)
            eval_reports.append(
                _evaluate(
                    config,
                    seed,
                    env_steps=env_steps,
                    eye=eval_eye,
                    brain=brain,
                    policy=policy,
                    run_dir=run_dir,
                    frame_rate_hz=frame_rate_hz,
                )
            )

    checkpoint = policy.save(
        run_dir / FINAL_CHECKPOINT,
        extra={
            "seed": seed,
            "env_steps": env_steps,
            "update": config.num_updates,
            "config": config.to_dict(),
        },
    )
    if fine_tuned and hasattr(learner.eye, "state_dict"):
        torch.save(learner.eye.state_dict(), run_dir / "eye_final.pt")  # type: ignore[union-attr]
    if brain is not None:
        torch.save(brain.state_dict(), run_dir / "brain_final.pt")
    return SeedResult(
        seed=seed,
        run_dir=run_dir,
        env_steps=env_steps,
        updates=config.num_updates,
        episodes=episode_index,
        checkpoint=checkpoint,
        last_update=last_record,
        eval_reports=tuple(eval_reports),
    )


def _new_tally(tally_class: Any, frame_rate_hz: float, info: dict[str, Any]) -> Any:
    start_m = info.get("arclength_m")
    return tally_class(
        frame_rate_hz,
        None if start_m is None else float(start_m),
        bool(info.get("start_focused", False)),
    )
