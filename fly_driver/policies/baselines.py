"""Baseline agents that drive from raw frames without an eye or a brain.

They exist to exercise envs and the evaluation harness before any learned policy
exists. Both keep their randomness in a private generator seeded through
``reset(seed=...)`` so the harness's "same seed, same actions" guarantee holds
regardless of global RNG state.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["ConstantAgent", "RandomAgent"]

Action = npt.NDArray[np.float32]


class RandomAgent:
    """Uniformly random ``(steer, throttle, brake)`` at every step.

    Args:
        seed: Initial generator seed; ``reset`` replaces it per episode.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the action generator for a new episode."""
        self._rng = np.random.default_rng(seed)

    def act(self, observation: object) -> Action:
        """Return a random in-range action; the observation is ignored."""
        steer = self._rng.uniform(-1.0, 1.0)
        throttle, brake = self._rng.uniform(0.0, 1.0, size=2)
        return np.array([steer, throttle, brake], dtype=np.float32)


class ConstantAgent:
    """The same action at every step.

    Args:
        steer: Steering in ``[-1, 1]``.
        throttle: Throttle in ``[0, 1]``.
        brake: Brake in ``[0, 1]``.
    """

    def __init__(
        self, steer: float = 0.0, throttle: float = 0.0, brake: float = 0.0
    ) -> None:
        self._action = np.array([steer, throttle, brake], dtype=np.float32)

    def act(self, observation: object) -> Action:
        """Return the configured action; the observation is ignored."""
        return self._action.copy()
