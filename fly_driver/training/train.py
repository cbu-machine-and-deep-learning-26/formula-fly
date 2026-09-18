"""The training loop: config → env → eye → brain → policy → PPO, one seed at a time (GH-17).

This is `AGENTS.md` §10 step 10 -- "PPO on the MuJoCo practice track with the frozen flyvis
eye + small policy head" -- as a function of a :class:`~fly_driver.training.TrainConfig`.
:func:`train` runs every seed in the config, :func:`train_seed` runs one, and each seed is a
complete, independent run: its own env, its own eye, its own policy, its own directory of
logs. Nothing is shared between seeds, so three seeds in one process and three seeds on
three Sparks produce the same files.

What a seed's run directory holds::

    runs/<name>/config.yaml              the condition, as run (written by train())
    runs/<name>/seed_0/config.yaml       the same, narrowed to this seed
    runs/<name>/seed_0/episodes.csv      one row per training episode, reward split by term
    runs/<name>/seed_0/updates.csv       one row per PPO update
    runs/<name>/seed_0/policy_final.pt   the policy head (+ normaliser statistics)
    runs/<name>/seed_0/eval/step_<n>/    the deterministic evaluation protocol's outputs

The loop rolls out through the same seams :class:`~fly_driver.drivers.DirectDriveAgent`
uses -- the frame is validated, the features are validated, the action is a
:class:`~fly_driver.interface.ControlVector` -- and the periodic evaluation *is* a
``DirectDriveAgent`` driven by :func:`fly_driver.training.evaluate`, so the numbers a
training run reports are the harness's numbers, not a second definition of them.

Seeds and determinism: ``seed`` seeds Python, numpy and torch once, every episode's env
reset takes a seed drawn from a generator seeded with it, and the periodic evaluation
restores every generator it touched, so a run's training rows are the same with evaluation
on or off. On CPU two runs with the same config are byte-identical; the tests assert it.
"""

from __future__ import annotations

import random
import shutil
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fly_driver.drivers import DirectDriveAgent
from fly_driver.envs.dummy_track import DummyTrackEnv
from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.envs.scene import SceneConfig
from fly_driver.eyes.flyvis_eye import FlyvisEye
from fly_driver.eyes.pixel_eye import PixelEye
from fly_driver.interface import ControlVector, Eye, validate_features, validate_frame
from fly_driver.policies.mlp_policy import MlpPolicy
from fly_driver.training.config import (
    KNOWN_BRAINS,
    KNOWN_ENV_TYPES,
    KNOWN_EYE_TYPES,
    EnvConfig,
    EyeConfig,
    PolicyConfig,
    TrainConfig,
)
from fly_driver.training.evaluation import EvalConfig, EvalReport, evaluate, seed_everything
from fly_driver.training.loggers import (
    EPISODES_FILENAME,
    CsvLogger,
    RunLogger,
    TrainingEpisode,
    TrainingLogger,
    UpdateRecord,
    WandbLogger,
)
from fly_driver.training.ppo import DifferentiableEye, PPOLearner, RolloutBuffer

__all__ = [
    "BRAIN_BUILDERS",
    "ENV_BUILDERS",
    "EYE_BUILDERS",
    "FINAL_CHECKPOINT",
    "ProgressCallback",
    "SeedResult",
    "build_env",
    "build_eye",
    "build_policy",
    "resolve_device",
    "train",
    "train_seed",
]

FINAL_CHECKPOINT = "policy_final.pt"

#: Called after every PPO update with its record; ``scripts/train.py`` prints it.
ProgressCallback = Callable[[UpdateRecord], None]


# -- building the stages from their config ------------------------------------------------


def _build_practice_track(config: EnvConfig) -> PracticeTrack:
    params = dict(config.params)
    scene_kwargs = dict(params.pop("scene", {}) or {})
    if "racing_line" in params:
        scene_kwargs["racing_line"] = bool(params.pop("racing_line"))
    scene = SceneConfig(**scene_kwargs) if scene_kwargs else None
    return PracticeTrack(scene=scene, max_steps=config.max_steps, **params)


def _build_dummy_track(config: EnvConfig) -> DummyTrackEnv:
    return DummyTrackEnv(max_steps=config.max_steps, **config.params)


def _build_flyvis_eye(config: EyeConfig, env: Any) -> Eye:
    return FlyvisEye(
        frame_shape=tuple(env.frame_shape),
        frame_rate_hz=float(env.frame_rate_hz),
        **config.params,
    )


def _build_pixel_eye(config: EyeConfig, env: Any) -> Eye:
    return PixelEye(frame_shape=tuple(env.frame_shape), **config.params)


#: Env type name → builder. Keyed exactly like :data:`~fly_driver.training.config.KNOWN_ENV_TYPES`.
ENV_BUILDERS: dict[str, Callable[[EnvConfig], Any]] = {
    "practice_track": _build_practice_track,
    "dummy": _build_dummy_track,
}

#: Eye type name → builder taking the env, which supplies ``frame_shape`` and
#: ``frame_rate_hz`` so the eye and the track cannot disagree. GH-15 registers the control
#: eyes here (and in ``KNOWN_EYE_TYPES``).
EYE_BUILDERS: dict[str, Callable[[EyeConfig, Any], Eye]] = {
    "flyvis": _build_flyvis_eye,
    "pixels": _build_pixel_eye,
}

#: Brain name → builder, or ``None`` for no brain (the policy reads the eye directly).
#: GH-23 registers the whole-brain model here (and in ``KNOWN_BRAINS``).
BRAIN_BUILDERS: dict[str, Callable[..., Any] | None] = {"none": None}

assert set(ENV_BUILDERS) == set(KNOWN_ENV_TYPES)
assert set(EYE_BUILDERS) == set(KNOWN_EYE_TYPES)
assert set(BRAIN_BUILDERS) == set(KNOWN_BRAINS)


def build_env(config: EnvConfig) -> Any:
    """Construct the env a config names."""
    return ENV_BUILDERS[config.type](config)


def build_eye(config: EyeConfig, env: Any) -> Eye:
    """Construct the eye a config names, sized to ``env``'s frames."""
    return EYE_BUILDERS[config.type](config, env)


def build_policy(config: PolicyConfig, feature_dim: int) -> MlpPolicy:
    """Construct the policy head for ``feature_dim`` features."""
    return MlpPolicy(
        feature_dim=feature_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
        normalize_features=config.normalize_features,
        init_log_std=config.init_log_std,
    )


def resolve_device(name: str) -> torch.device:
    """``auto`` is CUDA when available, else CPU; anything else is taken as a torch device."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"device {name!r} requested but CUDA is not available here")
    return device


# -- the run ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedResult:
    """What one seed's run produced.

    Args:
        seed: The seed.
        run_dir: Its directory.
        env_steps: Env steps taken.
        updates: PPO updates run.
        episodes: Training episodes finished.
        checkpoint: The final policy checkpoint.
        last_update: The last update's record, or ``None`` if none ran.
        eval_reports: Every periodic evaluation, in order.
    """

    seed: int
    run_dir: Path
    env_steps: int
    updates: int
    episodes: int
    checkpoint: Path
    last_update: UpdateRecord | None
    eval_reports: tuple[EvalReport, ...]


def train(
    config: TrainConfig, *, progress: ProgressCallback | None = None, overwrite: bool = False
) -> list[SeedResult]:
    """Run every seed in ``config``, one after another, and write the condition's config.

    Args:
        config: The condition.
        progress: Called after every update of every seed.
        overwrite: Delete an existing seed directory instead of refusing to touch it.

    Returns:
        One :class:`SeedResult` per seed, in config order.
    """
    config.run_root.mkdir(parents=True, exist_ok=True)
    config.to_yaml(config.run_root / "config.yaml")
    return [
        train_seed(config, seed, progress=progress, overwrite=overwrite) for seed in config.seeds
    ]


def train_seed(
    config: TrainConfig,
    seed: int,
    *,
    progress: ProgressCallback | None = None,
    overwrite: bool = False,
) -> SeedResult:
    """Run one seed of a condition: build everything fresh, train, log, save.

    Args:
        config: The condition; ``seed`` must be one of its ``seeds``.
        seed: Which seed.
        progress: Called after every PPO update.
        overwrite: Delete an existing run directory instead of refusing.

    Raises:
        ValueError: If ``seed`` is not in the config, or the eye cannot be fine-tuned as
            asked, or the env does not publish per-term rewards.
        FileExistsError: If the run directory already holds a run and ``overwrite`` is off.
    """
    if seed not in config.seeds:
        raise ValueError(f"seed {seed} is not in the config's seeds {list(config.seeds)}")
    run_dir = config.run_dir(seed)
    if (run_dir / EPISODES_FILENAME).exists():
        if not overwrite:
            raise FileExistsError(
                f"{run_dir} already holds a run; pass overwrite=True (scripts/train.py "
                "--overwrite) or choose another name"
            )
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config.with_seeds([seed]).to_yaml(run_dir / "config.yaml")

    seed_everything(seed)
    device = resolve_device(config.device)
    env = build_env(config.env)
    try:
        eye = build_eye(config.eye, env)
        trainable_eye = _trainable_eye(config.eye, eye)
        if BRAIN_BUILDERS[config.brain] is not None:  # pragma: no cover - none registered yet
            raise NotImplementedError(f"brain {config.brain!r} has a builder but no wiring")
        feature_dim = int(eye.feature_dim)
        policy = build_policy(config.policy, feature_dim).to(device)
        learner = PPOLearner(policy, config.ppo, eye=trainable_eye, device=device)
        loggers = _open_loggers(config, seed, run_dir)
        try:
            return _run(
                config,
                seed,
                env=env,
                eye=eye,
                policy=policy,
                learner=learner,
                loggers=loggers,
                run_dir=run_dir,
                device=device,
                progress=progress,
            )
        finally:
            loggers.close()
    finally:
        env.close()


def _trainable_eye(config: EyeConfig, eye: Eye) -> DifferentiableEye | None:
    """The eye to hand the optimiser, or ``None`` when it stays frozen."""
    if config.frozen:
        freeze = getattr(eye, "requires_grad_", None)
        if callable(freeze):
            freeze(False)
        return None
    if not isinstance(eye, DifferentiableEye):
        raise ValueError(
            f"eye type {config.type!r} cannot be fine-tuned by this loop: it has no "
            "differentiable batch encoder (encode_batch). The flyvis optic lobe keeps state "
            "between frames, so training through it is backpropagation through time -- a "
            "separate condition (AGENTS.md §6) this loop does not build -- and pixels has no "
            "parameters. Set eye.frozen: true."
        )
    if not any(parameter.requires_grad for parameter in eye.parameters()):
        raise ValueError(f"eye type {config.type!r} has no trainable parameters; set frozen: true")
    return eye


def _open_loggers(config: TrainConfig, seed: int, run_dir: Path) -> RunLogger:
    loggers: list[TrainingLogger] = [CsvLogger(run_dir)]
    if config.logging.wandb.enabled:
        loggers.append(
            WandbLogger(
                config.logging.wandb,
                run_name=f"{config.name}/seed_{seed}",
                group=config.name,
                run_config={**config.to_dict(), "seed": seed},
                run_dir=run_dir,
            )
        )
    return RunLogger(loggers)


class _EpisodeTally:
    """Running totals of the episode in progress."""

    def __init__(self, frame_rate_hz: float) -> None:
        self.frame_rate_hz = frame_rate_hz
        self.steps = 0
        self.total_return = 0.0
        self.terms: dict[str, float] = {}
        self.lap_complete = False
        self.lap_time: float | None = None

    def add(self, reward: float, terms: Mapping[str, float], info: Mapping[str, Any]) -> None:
        if self.terms and set(terms) != set(self.terms):
            raise ValueError(
                f"reward terms changed within an episode: {sorted(self.terms)} then "
                f"{sorted(terms)}; a reward function must publish the same terms every step"
            )
        self.steps += 1
        self.total_return += reward
        for name, value in terms.items():
            self.terms[name] = self.terms.get(name, 0.0) + value
        if info.get("lap_complete"):
            self.lap_complete = True
            reported = info.get("lap_time")
            # The harness's rule: the env's completed-lap time, else steps over the rate.
            self.lap_time = (
                float(reported) if reported is not None else self.steps / self.frame_rate_hz
            )

    def finish(
        self,
        *,
        seed: int,
        episode: int,
        update: int,
        env_steps: int,
        terminated: bool,
        truncated: bool,
    ) -> TrainingEpisode:
        return TrainingEpisode(
            seed=seed,
            episode=episode,
            update=update,
            env_steps=env_steps,
            steps=self.steps,
            total_return=self.total_return,
            lap_complete=self.lap_complete,
            lap_time_s=self.lap_time,
            terminated=terminated,
            truncated=truncated,
            reward_terms=dict(self.terms),
        )


def _reward_terms(info: Mapping[str, Any], reward: float) -> dict[str, float]:
    """The step's per-term rewards, checked against the scalar the env returned."""
    terms = info.get("reward_terms")
    if not isinstance(terms, Mapping) or not terms:
        raise ValueError(
            "env info has no 'reward_terms' mapping for this step; the loop logs per-term "
            "rewards (AGENTS.md §11) and needs the env to publish them the way PracticeTrack "
            "and DummyTrackEnv do"
        )
    values = {str(name): float(value) for name, value in terms.items()}
    total = sum(values.values())
    if abs(total - float(reward)) > 1e-6 * max(1.0, abs(float(reward))):
        raise ValueError(
            f"reward terms sum to {total!r} but the env returned reward {float(reward)!r}; "
            "a RewardFunction's terms must add up to its reward"
        )
    return values


@contextmanager
def _preserved_rng() -> Iterator[None]:
    """Leave every generator exactly as it was, whatever ran inside."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()  # noqa: NPY002  the harness seeds the legacy global
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)  # noqa: NPY002  and it must be put back
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def _sync_eval_eye(train_eye: Eye, eval_eye: Eye, *, fine_tuned: bool) -> None:
    """Give the evaluation eye the training eye's current weights when they can change."""
    if fine_tuned and hasattr(train_eye, "state_dict") and hasattr(eval_eye, "load_state_dict"):
        eval_eye.load_state_dict(train_eye.state_dict())


def _evaluate(
    config: TrainConfig,
    seed: int,
    *,
    env_steps: int,
    eye: Eye,
    policy: MlpPolicy,
    run_dir: Path,
    frame_rate_hz: float,
) -> EvalReport:
    """One deterministic evaluation, written under ``run_dir/eval/step_<env_steps>``."""
    eval_config = EvalConfig(
        episodes=config.eval.episodes,
        seed=seed,
        max_steps=config.eval.max_steps,
        frame_rate_hz=frame_rate_hz,
        record_video=config.eval.record_video,
        output_dir=str(run_dir / "eval" / f"step_{env_steps:09d}"),
        env_steps=env_steps,
    )
    driver = DirectDriveAgent(eye, policy)
    was_training = policy.training
    policy.eval()
    try:
        with _preserved_rng():
            return evaluate(lambda: build_env(config.env), driver, eval_config)
    finally:
        policy.train(was_training)


def _run(
    config: TrainConfig,
    seed: int,
    *,
    env: Any,
    eye: Eye,
    policy: MlpPolicy,
    learner: PPOLearner,
    loggers: RunLogger,
    run_dir: Path,
    device: torch.device,
    progress: ProgressCallback | None,
) -> SeedResult:
    ppo = config.ppo
    frame_shape = tuple(int(size) for size in env.frame_shape)
    frame_rate_hz = float(env.frame_rate_hz)
    feature_dim = int(policy.feature_dim)
    fine_tuned = learner.eye is not None
    buffer = RolloutBuffer(
        ppo.rollout_steps, feature_dim, frame_shape=frame_shape if fine_tuned else None
    )
    episode_seeds = np.random.default_rng(seed)
    num_updates = config.num_updates
    started = time.perf_counter()
    env_steps = 0
    episode_index = 0
    eval_eye: Eye | None = None
    eval_reports: list[EvalReport] = []
    last_record: UpdateRecord | None = None

    def to_device(features: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(features, device=device)[None]

    def begin_episode() -> tuple[np.ndarray, np.ndarray]:
        frame, _ = env.reset(seed=int(episode_seeds.integers(0, 2**31 - 1)))
        eye.reset()
        frame = validate_frame(frame, frame_shape)
        return frame, validate_features(eye.encode(frame), feature_dim)

    frame, features = begin_episode()
    tally = _EpisodeTally(frame_rate_hz)

    for update in range(1, num_updates + 1):
        update_started = time.perf_counter()
        buffer.reset()
        finished: list[TrainingEpisode] = []
        for _ in range(ppo.rollout_steps):
            raw, log_prob, value = policy.sample(to_device(features))
            raw_action = raw[0].cpu().numpy()
            # The interface's explicit squash: a non-finite sample raises here, loudly.
            control = ControlVector.clipped(*raw_action.tolist())
            next_frame, reward, terminated, truncated, info = env.step(control)
            terminated, truncated = bool(terminated), bool(truncated)
            terms = _reward_terms(info, reward)
            tally.add(float(reward), terms, info)
            env_steps += 1

            stored_reward = float(reward)
            if truncated and not terminated:
                # A time limit is not a terminal state: bootstrap from the observation the
                # cap cut off, so the value function does not learn that time runs out.
                capped = validate_frame(next_frame, frame_shape)
                capped_features = validate_features(eye.encode(capped), feature_dim)
                stored_reward += ppo.gamma * float(policy.value(to_device(capped_features))[0])
            done = terminated or truncated
            buffer.add(
                features=features,
                raw_action=raw_action,
                log_prob=float(log_prob[0]),
                reward=stored_reward,
                episode_end=done,
                value=float(value[0]),
                frame=frame if fine_tuned else None,
            )

            if done:
                record = tally.finish(
                    seed=seed,
                    episode=episode_index,
                    update=update,
                    env_steps=env_steps,
                    terminated=terminated,
                    truncated=truncated,
                )
                loggers.log_episode(record)
                finished.append(record)
                episode_index += 1
                tally = _EpisodeTally(frame_rate_hz)
                frame, features = begin_episode()
            else:
                frame = validate_frame(next_frame, frame_shape)
                features = validate_features(eye.encode(frame), feature_dim)

        last_value = 0.0 if buffer.episode_ends[-1] else float(policy.value(to_device(features))[0])
        buffer.compute_advantages(last_value, gamma=ppo.gamma, gae_lambda=ppo.gae_lambda)
        stats = learner.update(buffer, update_index=update, total_updates=num_updates)
        # After the update, never before: the rollout's log-probabilities were computed
        # with the statistics in force during collection, and the update must see the same.
        policy.update_normalizer(torch.as_tensor(buffer.features, device=device))

        now = time.perf_counter()
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
        every_eval = config.eval.every_updates
        if every_eval and update % every_eval == 0:
            if eval_eye is None:
                eval_eye = build_eye(config.eye, env)
            _sync_eval_eye(eye, eval_eye, fine_tuned=fine_tuned)
            eval_reports.append(
                _evaluate(
                    config,
                    seed,
                    env_steps=env_steps,
                    eye=eval_eye,
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
            "update": num_updates,
            "config": config.to_dict(),
        },
    )
    if fine_tuned and hasattr(eye, "state_dict"):
        torch.save(eye.state_dict(), run_dir / "eye_final.pt")
    return SeedResult(
        seed=seed,
        run_dir=run_dir,
        env_steps=env_steps,
        updates=num_updates,
        episodes=episode_index,
        checkpoint=checkpoint,
        last_update=last_record,
        eval_reports=tuple(eval_reports),
    )
