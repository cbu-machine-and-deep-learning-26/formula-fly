"""Environment wrappers that perturb observations or actions for robustness evals.

Each wrapper keeps the env protocol (``reset``/``step``/``render``/``close``) and
draws its randomness from a generator seeded in ``reset``, so a perturbed
evaluation is as reproducible as a clean one. Frames must be uint8 RGB; anything
else raises rather than being coerced (``AGENTS.md`` §11).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "PERTURBATIONS",
    "ActionDelay",
    "Brightness",
    "ObservationNoise",
    "PerturbationSpec",
    "apply_perturbation",
]

Frame = npt.NDArray[np.uint8]


def _as_frame(observation: object) -> Frame:
    if not isinstance(observation, np.ndarray) or observation.dtype != np.uint8:
        raise TypeError("perturbations expect uint8 frame observations")
    return observation


class _Wrapper:
    def __init__(self, env: Any) -> None:
        self.env = env

    def reset(self, *, seed: int | None = None, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action: object) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self.env.step(action)

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()


class ObservationNoise(_Wrapper):
    """Add seeded Gaussian pixel noise to every frame.

    Args:
        env: Wrapped environment.
        std: Noise standard deviation in ``[0, 1]`` luminance units.
    """

    def __init__(self, env: Any, std: float) -> None:
        super().__init__(env)
        if std < 0:
            raise ValueError("std must be non-negative")
        self.std = float(std)
        self._rng = np.random.default_rng()

    def reset(self, *, seed: int | None = None, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._rng = np.random.default_rng(None if seed is None else seed + 1_000_003)
        observation, info = self.env.reset(seed=seed, **kwargs)
        return self._perturb(observation), info

    def step(self, action: object) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._perturb(observation), reward, terminated, truncated, info

    def _perturb(self, observation: object) -> Frame:
        frame = _as_frame(observation)
        noise = self._rng.normal(0.0, self.std * 255.0, size=frame.shape)
        return np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)


class Brightness(_Wrapper):
    """Scale every frame's brightness.

    Args:
        env: Wrapped environment.
        scale: Multiplier applied before clipping to ``[0, 255]``.
    """

    def __init__(self, env: Any, scale: float) -> None:
        super().__init__(env)
        if scale < 0:
            raise ValueError("scale must be non-negative")
        self.scale = float(scale)

    def reset(self, *, seed: int | None = None, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, **kwargs)
        return self._perturb(observation), info

    def step(self, action: object) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._perturb(observation), reward, terminated, truncated, info

    def _perturb(self, observation: object) -> Frame:
        frame = _as_frame(observation)
        return np.clip(frame.astype(np.float64) * self.scale, 0, 255).astype(np.uint8)


class ActionDelay(_Wrapper):
    """Deliver each action ``steps`` frames late; neutral actions fill the gap.

    Args:
        env: Wrapped environment.
        steps: Delay in frames (``0`` is a no-op).
    """

    def __init__(self, env: Any, steps: int) -> None:
        super().__init__(env)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self.steps = int(steps)
        self._queue: deque[object] = deque()

    def reset(self, *, seed: int | None = None, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        neutral = np.zeros(3, dtype=np.float32)
        self._queue = deque(neutral.copy() for _ in range(self.steps))
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action: object) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self._queue.append(action)
        return self.env.step(self._queue.popleft())


PERTURBATIONS: dict[str, type[_Wrapper]] = {
    "observation_noise": ObservationNoise,
    "brightness": Brightness,
    "action_delay": ActionDelay,
}


@dataclass(frozen=True)
class PerturbationSpec:
    """A named perturbation and its keyword parameters.

    Args:
        name: Key in :data:`PERTURBATIONS`.
        params: Keyword arguments for the wrapper.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name not in PERTURBATIONS:
            raise ValueError(
                f"unknown perturbation {self.name!r}; choose from {sorted(PERTURBATIONS)}"
            )
        object.__setattr__(self, "params", dict(self.params))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> PerturbationSpec:
        """Build from ``{"name": ..., <param>: ...}`` as written in YAML."""
        if "name" not in data:
            raise ValueError(f"perturbation entry needs a 'name': {dict(data)}")
        params = {key: value for key, value in data.items() if key != "name"}
        return cls(name=str(data["name"]), params=params)

    def to_dict(self) -> dict[str, Any]:
        """Inverse of :meth:`from_dict`."""
        return {"name": self.name, **self.params}

    @property
    def label(self) -> str:
        """Condition label such as ``observation_noise(std=0.1)``."""
        inner = ", ".join(f"{key}={value}" for key, value in sorted(self.params.items()))
        return f"{self.name}({inner})"


def apply_perturbation(env: Any, spec: PerturbationSpec) -> _Wrapper:
    """Wrap ``env`` according to ``spec``."""
    return PERTURBATIONS[spec.name](env, **spec.params)
