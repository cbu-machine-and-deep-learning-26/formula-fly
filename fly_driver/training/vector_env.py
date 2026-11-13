"""Many tracks at once: one env per worker process, frames in shared memory (GH-27).

A single trainer used to drive one track at batch size 1: render, eye, brain and policy
in turn, about 55 steps/s on a lab PC, with the GPU mostly idle. :class:`SubprocVectorEnv`
steps ``num_envs`` tracks in parallel worker processes, each with its own MuJoCo model and
offscreen renderer, and the trainer feeds all of their frames through the eye in one batch.

Workers never import torch or an eye: they build their track with
:func:`fly_driver.training.env_factory.build_env` and nothing else, so a worker costs a
MuJoCo model and a GL context, not a CUDA context.

**Auto-reset.** A worker resets its env as soon as an episode ends, with the reset seed the
trainer sent alongside the action, so the trainer's episode seeds are drawn in one process
in a fixed order and a run stays reproducible. The step result for that env then carries the
*reset* frame as the next observation, the ended episode's last info, and the reset info.
When the episode ended on the step cap, the frame the cap cut off is kept too, in
:attr:`StepResult.final_frames`, because the trainer bootstraps the value from it.

**Frames.** Each worker writes its frame into its row of one shared ``(num_envs, H, W, 3)``
uint8 array. Pipes carry only the small per-step info dict.

:class:`SyncVectorEnv` is the same interface in one process, for tests and for envs that
cannot leave the main process.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from multiprocessing import shared_memory
from multiprocessing.connection import Connection
from typing import Any, Protocol

import numpy as np

from fly_driver.interface import ControlVector
from fly_driver.training.config import EnvConfig

__all__ = [
    "INFO_KEYS",
    "StepResult",
    "SubprocVectorEnv",
    "SyncVectorEnv",
    "VectorEnv",
    "slim_info",
]

#: The info keys the trainer and its loggers read. Workers send only these, to keep the
#: per-step message small.
INFO_KEYS = (
    "reward_terms",
    "progress_m",
    "lap_complete",
    "lap_time",
    "arclength_m",
    "start_focused",
    "crashed",
    "stalled",
    "diverged",
    "speed_mps",
    "lateral_m",
    "heading_error_rad",
)


def slim_info(info: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a step's info the trainer reads, with plain Python values."""
    slim: dict[str, Any] = {}
    for key in INFO_KEYS:
        if key not in info:
            continue
        value = info[key]
        if key == "reward_terms":
            value = {str(name): float(term) for name, term in value.items()}
        elif isinstance(value, np.generic):
            value = value.item()
        slim[key] = value
    return slim


@dataclass
class StepResult:
    """One step of every env.

    Args:
        rewards: ``(num_envs,)`` float64.
        terminated: ``(num_envs,)`` bool.
        truncated: ``(num_envs,)`` bool.
        infos: The step's info, per env. For an env whose episode ended, the ended episode's
            last step.
        reset_infos: The reset's info for an env whose episode ended this step, else ``None``.
        final_frames: ``{env index: frame}`` the step cap cut off, for every truncated env
            that was not also terminated. The observation in :attr:`VectorEnv.frames` is
            already the next episode's first frame.
    """

    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    infos: list[dict[str, Any]]
    reset_infos: list[dict[str, Any] | None]
    final_frames: dict[int, np.ndarray]


class VectorEnv(Protocol):
    """What the vectorised trainer needs from a set of tracks."""

    num_envs: int
    frame_shape: tuple[int, int, int]
    frame_rate_hz: float

    @property
    def frames(self) -> np.ndarray:
        """``(num_envs, H, W, 3)`` uint8: every env's current observation."""

    def reset(self, seeds: Sequence[int]) -> list[dict[str, Any]]:
        """Reset every env with its seed; returns each reset's info."""

    def step(self, actions: np.ndarray, reset_seeds: Sequence[int]) -> StepResult:
        """Step every env with its ``(steer, throttle, brake)`` row."""

    def set_training_steps(self, steps: int) -> None:
        """Tell every env how far training has got."""

    def close(self) -> None:
        """Release every env."""


def _step_one(
    env: Any, action: np.ndarray, reset_seed: int
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any], dict[str, Any] | None, np.ndarray | None]:
    control = ControlVector.clipped(*np.asarray(action, dtype=np.float64).tolist())
    frame, reward, terminated, truncated, info = env.step(control)
    terminated, truncated = bool(terminated), bool(truncated)
    reset_info = None
    final_frame = None
    if terminated or truncated:
        if truncated and not terminated:
            final_frame = np.asarray(frame, dtype=np.uint8).copy()
        frame, raw_reset_info = env.reset(seed=int(reset_seed))
        reset_info = slim_info(raw_reset_info)
    return frame, float(reward), terminated, truncated, slim_info(info), reset_info, final_frame


def _set_steps(env: Any, steps: int) -> None:
    setter = getattr(env, "set_training_steps", None)
    if callable(setter):
        setter(int(steps))


class SyncVectorEnv:
    """``num_envs`` envs stepped one after another in this process.

    Args:
        make_env: Builds one env; called ``num_envs`` times.
        num_envs: How many.
    """

    def __init__(self, make_env: Callable[[], Any], num_envs: int) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.envs = [make_env() for _ in range(num_envs)]
        self.num_envs = num_envs
        first = self.envs[0]
        self.frame_shape = tuple(int(size) for size in first.frame_shape)
        self.frame_rate_hz = float(first.frame_rate_hz)
        self._frames = np.zeros((num_envs, *self.frame_shape), dtype=np.uint8)

    @property
    def frames(self) -> np.ndarray:
        """Every env's current observation."""
        return self._frames

    def reset(self, seeds: Sequence[int]) -> list[dict[str, Any]]:
        """Reset every env."""
        infos = []
        for index, (env, seed) in enumerate(zip(self.envs, seeds, strict=True)):
            frame, info = env.reset(seed=int(seed))
            self._frames[index] = frame
            infos.append(slim_info(info))
        return infos

    def step(self, actions: np.ndarray, reset_seeds: Sequence[int]) -> StepResult:
        """Step every env, resetting any whose episode ended."""
        rewards = np.zeros(self.num_envs)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []
        reset_infos: list[dict[str, Any] | None] = []
        final_frames: dict[int, np.ndarray] = {}
        for index, env in enumerate(self.envs):
            frame, reward, ended, capped, info, reset_info, final = _step_one(
                env, actions[index], reset_seeds[index]
            )
            self._frames[index] = frame
            rewards[index], terminated[index], truncated[index] = reward, ended, capped
            infos.append(info)
            reset_infos.append(reset_info)
            if final is not None:
                final_frames[index] = final
        return StepResult(rewards, terminated, truncated, infos, reset_infos, final_frames)

    def set_training_steps(self, steps: int) -> None:
        """Tell every env how far training has got."""
        for env in self.envs:
            _set_steps(env, steps)

    def close(self) -> None:
        """Close every env."""
        for env in self.envs:
            env.close()


def _worker(
    connection: Connection,
    env_config: EnvConfig,
    index: int,
    shared_name: str,
    shape: tuple[int, ...],
) -> None:
    """One env in its own process, driven by commands on ``connection``."""
    # Imported here so the parent's import of this module stays light.
    from fly_driver.training.env_factory import build_env

    shared = shared_memory.SharedMemory(name=shared_name)
    frames = np.ndarray(shape, dtype=np.uint8, buffer=shared.buf)
    env = None
    try:
        env = build_env(env_config, training=True)
        connection.send(("ready", (tuple(env.frame_shape), float(env.frame_rate_hz))))
        while True:
            command, payload = connection.recv()
            if command == "reset":
                frame, info = env.reset(seed=int(payload))
                frames[index] = frame
                connection.send(("ok", slim_info(info)))
            elif command == "step":
                action, reset_seed = payload
                frame, *rest = _step_one(env, action, reset_seed)
                frames[index] = frame
                connection.send(("ok", tuple(rest)))
            elif command == "steps":
                _set_steps(env, payload)
                connection.send(("ok", None))
            elif command == "close":
                connection.send(("ok", None))
                break
            else:
                raise ValueError(f"unknown command {command!r}")
    except (EOFError, KeyboardInterrupt):
        pass
    except Exception:  # noqa: BLE001  the parent re-raises it with the worker's traceback
        with contextlib.suppress(BrokenPipeError, OSError):
            connection.send(("error", traceback.format_exc()))
    finally:
        if env is not None:
            env.close()
        del frames
        shared.close()
        connection.close()


class SubprocVectorEnv:
    """``num_envs`` envs, each in its own worker process.

    Workers start with the ``spawn`` method on every platform: it is the only one Windows
    has, and a forked child would inherit the parent's CUDA and GL state on Linux.

    Args:
        env_config: The env section of the run config; each worker builds the training env.
        num_envs: How many workers.
        start_timeout_s: How long a worker may take to build its track.

    Raises:
        RuntimeError: If a worker fails to start or fails on a command; the message carries
            the worker's traceback.
    """

    def __init__(
        self, env_config: EnvConfig, num_envs: int, *, start_timeout_s: float = 300.0
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1")
        self.num_envs = num_envs
        context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[Any] = []
        self._shared: shared_memory.SharedMemory | None = None
        self._closed = False
        # The frame shape is only known once a track exists. A probe in this process would
        # cost a GL context here, so the first worker reports it and the rest wait.
        probe_shape = _probe_frame_shape(env_config)
        shape = (num_envs, *probe_shape)
        self._shared = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
        self._frames = np.ndarray(shape, dtype=np.uint8, buffer=self._shared.buf)
        self._frames.fill(0)
        try:
            for index in range(num_envs):
                parent, child = context.Pipe()
                process = context.Process(
                    target=_worker,
                    args=(child, env_config, index, self._shared.name, shape),
                    daemon=True,
                    name=f"fly-env-{index}",
                )
                process.start()
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
            reports = [self._receive(index, start_timeout_s) for index in range(num_envs)]
        except BaseException:
            self.close()
            raise
        frame_shape, frame_rate_hz = reports[0]
        if tuple(frame_shape) != probe_shape:
            self.close()
            raise RuntimeError(f"worker frame shape {frame_shape} != probed {probe_shape}")
        self.frame_shape = tuple(int(size) for size in frame_shape)
        self.frame_rate_hz = float(frame_rate_hz)

    @property
    def frames(self) -> np.ndarray:
        """Every env's current observation (a view of shared memory; copy to keep)."""
        return self._frames

    def _receive(self, index: int, timeout_s: float | None = None) -> Any:
        connection = self._connections[index]
        if timeout_s is not None and not connection.poll(timeout_s):
            raise RuntimeError(f"env worker {index} did not answer within {timeout_s:.0f} s")
        try:
            status, payload = connection.recv()
        except EOFError as error:
            raise RuntimeError(f"env worker {index} died") from error
        if status == "error":
            raise RuntimeError(f"env worker {index} failed:\n{payload}")
        return payload

    def _broadcast(self, command: str, payloads: Sequence[Any]) -> list[Any]:
        for connection, payload in zip(self._connections, payloads, strict=True):
            connection.send((command, payload))
        return [self._receive(index) for index in range(self.num_envs)]

    def reset(self, seeds: Sequence[int]) -> list[dict[str, Any]]:
        """Reset every env."""
        return self._broadcast("reset", [int(seed) for seed in seeds])

    def step(self, actions: np.ndarray, reset_seeds: Sequence[int]) -> StepResult:
        """Step every env in parallel, resetting any whose episode ended."""
        actions = np.asarray(actions, dtype=np.float64)
        payloads = [(actions[index], int(reset_seeds[index])) for index in range(self.num_envs)]
        results = self._broadcast("step", payloads)
        rewards = np.zeros(self.num_envs)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos: list[dict[str, Any]] = []
        reset_infos: list[dict[str, Any] | None] = []
        final_frames: dict[int, np.ndarray] = {}
        for index, (reward, ended, capped, info, reset_info, final) in enumerate(results):
            rewards[index], terminated[index], truncated[index] = reward, ended, capped
            infos.append(info)
            reset_infos.append(reset_info)
            if final is not None:
                final_frames[index] = final
        return StepResult(rewards, terminated, truncated, infos, reset_infos, final_frames)

    def set_training_steps(self, steps: int) -> None:
        """Tell every env how far training has got."""
        self._broadcast("steps", [int(steps)] * self.num_envs)

    def close(self) -> None:
        """Stop every worker and free the shared frames. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        for connection in self._connections:
            with contextlib.suppress(BrokenPipeError, OSError):
                connection.send(("close", None))
        for process in self._processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        for connection in self._connections:
            connection.close()
        if self._shared is not None:
            self._frames = np.zeros((0,), dtype=np.uint8)
            self._shared.close()
            self._shared.unlink()
            self._shared = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):  # interpreter shutdown
            self.close()


def _probe_frame_shape(env_config: EnvConfig) -> tuple[int, int, int]:
    """The frame shape a track of this config will emit, without building one."""
    if env_config.type == "practice_track":
        from fly_driver.interface import FRAME_SHAPE

        shape = env_config.params.get("frame_shape", FRAME_SHAPE)
        return tuple(int(size) for size in shape)  # type: ignore[return-value]
    if env_config.type == "dummy":
        from fly_driver.envs.dummy_track import DummyTrackEnv
        from fly_driver.training.env_factory import env_params

        params = env_params(env_config, training=True)
        env = DummyTrackEnv(max_steps=env_config.max_steps, **params)
        try:
            return tuple(int(size) for size in env.frame_shape)  # type: ignore[return-value]
        finally:
            env.close()
    raise ValueError(
        f"env type {env_config.type!r} cannot run in worker processes; "
        "Assetto Corsa is one real-time game, so use num_envs: 1"
    )
