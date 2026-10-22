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
    runs/<name>/seed_0/parameters.json   trainable parameter counts (RQ2 reads these)
    runs/<name>/seed_0/policy_final.pt   the policy head (+ normaliser statistics)
    runs/<name>/seed_0/brain_final.pt    the brain's state dict, when there is a brain
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

import copy
import json
import random
import shutil
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fly_driver.brains.blank import blank_trainable_parameter_count
from fly_driver.brains.central_complex import CentralComplexBrain
from fly_driver.drivers import DirectDriveAgent
from fly_driver.envs.assetto_corsa import AssettoCorsaTrack
from fly_driver.envs.dummy_track import DummyTrackEnv
from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.envs.scene import SceneConfig
from fly_driver.eyes.cnn_eye import CnnEye
from fly_driver.eyes.flyvis_eye import FlyvisEye
from fly_driver.eyes.pixel_eye import PixelEye
from fly_driver.eyes.random_projection_eye import RandomProjectionEye
from fly_driver.eyes.shuffled_connectome_eye import ShuffledConnectomeEye
from fly_driver.interface import ControlVector, Eye, validate_features, validate_frame
from fly_driver.policies.mlp_policy import MlpPolicy
from fly_driver.training.config import (
    KNOWN_BRAINS,
    KNOWN_ENV_TYPES,
    KNOWN_EYE_TYPES,
    BrainConfig,
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
    "PARAMETERS_FILENAME",
    "ProgressCallback",
    "SeedResult",
    "build_brain",
    "build_env",
    "build_eye",
    "build_policy",
    "load_driver",
    "resolve_device",
    "train",
    "train_seed",
]

FINAL_CHECKPOINT = "policy_final.pt"
PARAMETERS_FILENAME = "parameters.json"

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


def _build_assetto_corsa(config: EnvConfig) -> AssettoCorsaTrack:
    return AssettoCorsaTrack.connect(max_steps=config.max_steps, **config.params)


def _build_flyvis_eye(config: EyeConfig, env: Any) -> Eye:
    return FlyvisEye(
        frame_shape=tuple(env.frame_shape),
        frame_rate_hz=float(env.frame_rate_hz),
        **config.params,
    )


def _build_cnn_eye(config: EyeConfig, env: Any) -> Eye:
    return CnnEye(frame_shape=tuple(env.frame_shape), **config.params)


def _build_random_projection_eye(config: EyeConfig, env: Any) -> Eye:
    return RandomProjectionEye(frame_shape=tuple(env.frame_shape), **config.params)


def _build_shuffled_eye(config: EyeConfig, env: Any) -> Eye:
    return ShuffledConnectomeEye(
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
    "assetto_corsa": _build_assetto_corsa,
}

#: Eye type name → builder taking the env, which supplies ``frame_shape`` and
#: ``frame_rate_hz`` so the eye and the track cannot disagree.
EYE_BUILDERS: dict[str, Callable[[EyeConfig, Any], Eye]] = {
    "flyvis": _build_flyvis_eye,
    "cnn": _build_cnn_eye,
    "random_projection": _build_random_projection_eye,
    "shuffled": _build_shuffled_eye,
    "pixels": _build_pixel_eye,
}


def _build_central_complex(
    config: BrainConfig, input_dim: int, frame_rate_hz: float
) -> CentralComplexBrain:
    return CentralComplexBrain(input_dim, frame_rate_hz=frame_rate_hz, **config.params)


#: Brain name → builder taking the eye's feature count and the env's frame rate, or ``None``
#: for no brain (the policy reads the eye directly).
BRAIN_BUILDERS: dict[str, Callable[[BrainConfig, int, float], CentralComplexBrain] | None] = {
    "none": None,
    "central_complex": _build_central_complex,
}

assert set(ENV_BUILDERS) == set(KNOWN_ENV_TYPES)
assert set(EYE_BUILDERS) == set(KNOWN_EYE_TYPES)
assert set(BRAIN_BUILDERS) == set(KNOWN_BRAINS)


def build_env(config: EnvConfig, *, training: bool = False) -> Any:
    """Construct the env a config names.

    Args:
        config: The env section of a run config.
        training: Build the training env. Only then does ``params.random_start`` reach the
            env; every evaluation env (periodic, ``scripts/evaluate.py``, robustness) drops
            it and starts on the grid, so lap times stay comparable across runs.
    """
    if not training and "random_start" in config.params:
        params = {key: value for key, value in config.params.items() if key != "random_start"}
        config = replace(config, params=params)
    return ENV_BUILDERS[config.type](config)


def build_eye(config: EyeConfig, env: Any) -> Eye:
    """Construct the eye a config names, sized to ``env``'s frames."""
    return EYE_BUILDERS[config.type](config, env)


def build_brain(config: BrainConfig, eye: Eye, env: Any) -> CentralComplexBrain | None:
    """Construct the brain a config names, reading ``eye``'s features; ``None`` for no brain."""
    builder = BRAIN_BUILDERS[config.type]
    if builder is None:
        return None
    return builder(config, int(eye.feature_dim), float(env.frame_rate_hz))


def build_policy(config: PolicyConfig, feature_dim: int) -> MlpPolicy:
    """Construct the policy head for ``feature_dim`` features."""
    return MlpPolicy(
        feature_dim=feature_dim,
        hidden_sizes=config.hidden_sizes,
        activation=config.activation,
        normalize_features=config.normalize_features,
        init_log_std=config.init_log_std,
        init_action_mean=config.init_action_mean,
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
    env = build_env(config.env, training=True)
    try:
        eye = build_eye(config.eye, env)
        trainable_eye = _trainable_eye(config.eye, eye, device)
        brain = build_brain(config.brain, eye, env)
        if brain is not None:
            brain.to(device)
        feature_dim = int(eye.feature_dim) if brain is None else brain.output_dim
        policy = build_policy(config.policy, feature_dim).to(device)
        learner = PPOLearner(policy, config.ppo, eye=trainable_eye, brain=brain, device=device)
        parameters = _parameter_counts(config, trainable_eye, brain, policy)
        (run_dir / PARAMETERS_FILENAME).write_text(
            json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
        )
        loggers = _open_loggers(config, seed, run_dir, parameters)
        try:
            return _run(
                config,
                seed,
                env=env,
                eye=eye,
                brain=brain,
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


def _trainable_eye(config: EyeConfig, eye: Eye, device: torch.device) -> DifferentiableEye | None:
    """The eye to hand the optimiser, on the training device, or ``None`` when it is frozen."""
    if config.frozen:
        freeze = getattr(eye, "requires_grad_", None)
        if callable(freeze):
            freeze(False)
        return None
    if not isinstance(eye, DifferentiableEye):
        raise ValueError(
            f"eye type {config.type!r} cannot be fine-tuned by this loop: it has no "
            "differentiable batch encoder (encode_batch). The flyvis optic lobe and the "
            "shuffled connectome keep state between frames, so training through them is "
            "backpropagation through time -- a separate condition (AGENTS.md §6) this loop "
            "does not build -- and pixels and the random projection have no parameters. "
            "The small CNN does expose encode_batch. Set eye.frozen: true."
        )
    if not any(parameter.requires_grad for parameter in eye.parameters()):
        raise ValueError(f"eye type {config.type!r} has no trainable parameters; set frozen: true")
    move = getattr(eye, "to", None)
    if callable(move):
        move(device)
    return eye


def _parameter_counts(
    config: TrainConfig,
    eye: DifferentiableEye | None,
    brain: CentralComplexBrain | None,
    policy: MlpPolicy,
) -> dict[str, Any]:
    """Trainable scalars per stage, and a dense blank net of the brain's width (RQ2)."""

    def trainable(module: Any) -> int:
        if module is None:
            return 0
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    counts: dict[str, Any] = {
        "brain": config.brain.type,
        "eye_trainable": trainable(eye),
        "brain_trainable": trainable(brain),
        "head_trainable": trainable(policy),
        "neuron_count": None,
        "blank_trainable": None,
    }
    if brain is not None:
        counts["neuron_count"] = brain.neuron_count
        counts["blank_trainable"] = blank_trainable_parameter_count(
            brain.input_dim, brain.neuron_count
        )
    return counts


def _open_loggers(
    config: TrainConfig, seed: int, run_dir: Path, parameters: Mapping[str, Any]
) -> RunLogger:
    loggers: list[TrainingLogger] = [CsvLogger(run_dir)]
    if config.logging.wandb.enabled:
        loggers.append(
            WandbLogger(
                config.logging.wandb,
                run_name=f"{config.name}/seed_{seed}",
                group=config.name,
                run_config={**config.to_dict(), "seed": seed, "parameters": dict(parameters)},
                run_dir=run_dir,
            )
        )
    return RunLogger(loggers)


class _EpisodeTally:
    """Running totals of the episode in progress."""

    def __init__(self, frame_rate_hz: float, start_m: float | None) -> None:
        self.frame_rate_hz = frame_rate_hz
        self.start_m = start_m
        self.steps = 0
        self.total_return = 0.0
        self.terms: dict[str, float] = {}
        self.lap_complete = False
        self.lap_time: float | None = None
        self.distance_m: float | None = None

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
        if "progress_m" in info:
            self.distance_m = (self.distance_m or 0.0) + float(info["progress_m"])
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
            start_m=self.start_m,
            distance_m=self.distance_m,
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
    brain: CentralComplexBrain | None,
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
    # A copy, so evaluation's resets do not clear the training rollout's membrane.
    eval_brain = None if brain is None else copy.deepcopy(brain)
    driver = DirectDriveAgent(eye, policy, brain=eval_brain)
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
    brain: CentralComplexBrain | None,
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
    eye_dim = int(eye.feature_dim)
    feature_dim = int(policy.feature_dim)
    fine_tuned = learner.eye is not None
    buffer = RolloutBuffer(
        ppo.rollout_steps,
        feature_dim,
        frame_shape=frame_shape if fine_tuned else None,
        brain_dims=None if brain is None else (brain.input_dim, brain.neuron_count),
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

    def observe(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Eye features, the brain's membrane before this frame, and what the policy reads.

        The brain steps once per observed frame, here and nowhere else.
        """
        eye_features = validate_features(eye.encode(frame), eye_dim)
        if brain is None:
            return eye_features, None, eye_features
        state = brain.activity()
        return eye_features, state, validate_features(brain.step(eye_features), feature_dim)

    def begin_episode() -> tuple[
        np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, _EpisodeTally
    ]:
        """Reset env, eye and brain; the fresh tally records where the episode started."""
        frame, info = env.reset(seed=int(episode_seeds.integers(0, 2**31 - 1)))
        eye.reset()
        if brain is not None:
            brain.reset()
        frame = validate_frame(frame, frame_shape)
        start_m = info.get("arclength_m")
        tally = _EpisodeTally(frame_rate_hz, None if start_m is None else float(start_m))
        return frame, *observe(frame), tally

    frame, eye_features, brain_state, features, tally = begin_episode()

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
                capped_features = observe(validate_frame(next_frame, frame_shape))[2]
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
                brain_input=eye_features,
                brain_state=brain_state,
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
                frame, eye_features, brain_state, features, tally = begin_episode()
            else:
                frame = validate_frame(next_frame, frame_shape)
                eye_features, brain_state, features = observe(frame)

        last_value = 0.0 if buffer.episode_ends[-1] else float(policy.value(to_device(features))[0])
        buffer.compute_advantages(last_value, gamma=ppo.gamma, gae_lambda=ppo.gae_lambda)
        stats = learner.update(buffer, update_index=update, total_updates=num_updates)
        # After the update, never before: the rollout's log-probabilities were computed
        # with the statistics in force during collection, and the update must see the same.
        policy.update_normalizer(torch.as_tensor(buffer.features, device=device))
        if brain is not None:
            # The next rollout's first frame went through the brain before this update; step
            # it again from the same membrane so the policy reads what the update replays.
            with torch.no_grad():
                brain.voltage.copy_(torch.as_tensor(brain_state))
            features = validate_features(brain.step(eye_features), feature_dim)

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
            if brain is not None:
                torch.save(brain.state_dict(), run_dir / f"brain_update_{update:05d}.pt")
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
            "update": num_updates,
            "config": config.to_dict(),
        },
    )
    if fine_tuned and hasattr(eye, "state_dict"):
        torch.save(eye.state_dict(), run_dir / "eye_final.pt")
    if brain is not None:
        torch.save(brain.state_dict(), run_dir / "brain_final.pt")
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


def load_driver(
    config: TrainConfig,
    checkpoint: str | Path,
    env: Any,
    *,
    device: str | torch.device = "cpu",
) -> DirectDriveAgent:
    """Rebuild a trained eye → brain → policy driver from a policy checkpoint.

    The brain's and a fine-tuned eye's weights are read from the files saved beside the
    checkpoint: ``policy_final.pt`` pairs with ``brain_final.pt`` and ``eye_final.pt``,
    ``policy_update_00010.pt`` with ``brain_update_00010.pt``.

    Args:
        config: The run's config, e.g. ``TrainConfig.from_yaml(<seed dir>/config.yaml)``.
        checkpoint: A ``policy_*.pt`` file written by training.
        env: An env of the run's type; the eye takes its frame shape and rate from it.
        device: Where the policy and the brain run.

    Raises:
        FileNotFoundError: If the run had a brain or a fine-tuned eye and its weights are
            not beside the checkpoint. Periodic checkpoints do not save a fine-tuned eye.
    """
    checkpoint = Path(checkpoint)

    def sibling(stage: str) -> Path:
        path = checkpoint.with_name(checkpoint.name.replace("policy_", f"{stage}_", 1))
        if path == checkpoint or not path.exists():
            raise FileNotFoundError(f"{path} not found; it holds the {stage} for {checkpoint}")
        return path

    eye = build_eye(config.eye, env)
    if not config.eye.frozen:
        weights = torch.load(sibling("eye"), map_location="cpu", weights_only=True)
        eye.load_state_dict(weights)  # type: ignore[attr-defined]
    brain = build_brain(config.brain, eye, env)
    if brain is not None:
        brain.load_state_dict(torch.load(sibling("brain"), map_location="cpu", weights_only=True))
        brain.to(device)
    policy = MlpPolicy.load(checkpoint, device=device).eval()
    return DirectDriveAgent(eye, policy, brain=brain)
