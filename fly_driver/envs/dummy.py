"""Deterministic stand-in env for eval-seed tests (no gymnasium)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fly_driver.interface import DEFAULT_FRAME_SHAPE, ControlVector
from fly_driver.training.reward import RewardTerms, shaped_reward


@dataclass
class DummyStep:
    frame: npt.NDArray[np.uint8]
    reward: float
    terms: RewardTerms
    terminated: bool
    truncated: bool


class DummyRaceEnv:
    """Seeded synthetic track. Progress reward only increases when throttle > 0.2."""

    def __init__(
        self,
        seed: int = 0,
        frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
        max_steps: int = 64,
    ) -> None:
        self.frame_shape = frame_shape
        self.max_steps = int(max_steps)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self.t = 0
        self.progress = 0.0
        self._frame = self._paint(0)

    def reset(self, seed: int | None = None) -> npt.NDArray[np.uint8]:
        if seed is not None:
            self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self.t = 0
        self.progress = 0.0
        self._frame = self._paint(0)
        return self._frame.copy()

    def _paint(self, t: int) -> npt.NDArray[np.uint8]:
        h, w, c = self.frame_shape
        frame = np.zeros((h, w, c), dtype=np.uint8)
        # Vertical stripe whose column encodes time, plus a seeded noise plane.
        col = t % w
        frame[:, col, 1] = 255
        noise = self._rng.integers(0, 20, size=(h, w), dtype=np.uint8)
        frame[:, :, 2] = noise
        return frame

    def step(self, action: ControlVector) -> DummyStep:
        a = action.clipped()
        self.t += 1
        delta = 0.0
        if a.throttle > 0.2 and a.brake < 0.5:
            delta = 0.1 * float(a.throttle)
            self.progress += delta
        on_track = 0.01
        terms = RewardTerms(progress=delta, on_track=on_track, alive=0.0, collision=0.0)
        terminated = self.progress >= 5.0
        truncated = self.t >= self.max_steps
        self._frame = self._paint(self.t)
        return DummyStep(
            frame=self._frame.copy(),
            reward=shaped_reward(terms),
            terms=terms,
            terminated=terminated,
            truncated=truncated,
        )
