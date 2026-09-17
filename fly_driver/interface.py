"""Shared stage interface: frame in, control vector out.

Every pipeline stage is replaceable. Direct-drive skips the body; a dummy env
stands in for CarRacing; identity brain stands in for the central complex.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FrameArray = npt.NDArray[np.generic]
FeatureArray = npt.NDArray[np.float32]

DEFAULT_FRAME_SHAPE: tuple[int, int, int] = (96, 96, 3)
STEER_RANGE: tuple[float, float] = (-1.0, 1.0)
THROTTLE_RANGE: tuple[float, float] = (0.0, 1.0)
BRAKE_RANGE: tuple[float, float] = (0.0, 1.0)


def _clip(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(min(hi, max(lo, value)))


@dataclass(frozen=True, slots=True)
class ControlVector:
    """Car command in CarRacing-style ranges.

    steer: [-1, 1], throttle: [0, 1], brake: [0, 1]
    """

    steer: float
    throttle: float
    brake: float

    def clipped(self) -> ControlVector:
        return ControlVector(
            steer=_clip(self.steer, STEER_RANGE),
            throttle=_clip(self.throttle, THROTTLE_RANGE),
            brake=_clip(self.brake, BRAKE_RANGE),
        )

    def as_array(self) -> npt.NDArray[np.float32]:
        c = self.clipped()
        return np.array([c.steer, c.throttle, c.brake], dtype=np.float32)


def require_frame(
    frame: FrameArray,
    expected: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
) -> npt.NDArray[np.float32]:
    """Reject unexpected shapes. Callers must not silently resize."""
    arr = np.asarray(frame)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected HWC frame {expected}, got ndim={arr.ndim} shape={arr.shape}; "
            "no silent channel expand or resize."
        )
    if arr.shape != expected:
        raise ValueError(f"Expected frame shape {expected}, got {arr.shape}; no silent resize.")
    return arr.astype(np.float32, copy=False)


def require_features(features: FeatureArray, size: int) -> npt.NDArray[np.float32]:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1-D features of length {size}, got shape {arr.shape}.")
    if arr.shape[0] != size:
        raise ValueError(f"Expected feature length {size}, got {arr.shape[0]}.")
    if not np.isfinite(arr).all():
        raise ValueError("Features contain NaN or Inf.")
    return arr


class Eye(ABC):
    """Pixels → feature vector."""

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return DEFAULT_FRAME_SHAPE

    @property
    @abstractmethod
    def output_size(self) -> int:
        """Length of the feature vector returned by encode()."""

    def reset(self) -> None:
        return None

    @abstractmethod
    def encode(self, frame: FrameArray) -> npt.NDArray[np.float32]:
        """Return float32 features of shape ``(output_size,)``."""


class Brain(ABC):
    """Optional recurrent stage between eye and policy."""

    @abstractmethod
    def step(self, features: FeatureArray) -> npt.NDArray[np.float32]:
        """Consume one feature vector; return features for the policy."""

    def reset(self) -> None:
        return None


class Policy(ABC):
    """Features → control."""

    @abstractmethod
    def act(self, features: FeatureArray) -> ControlVector:
        """Return a (possibly unclipped) control vector. Agents clip."""

    def reset(self) -> None:
        return None


class IdentityBrain(Brain):
    """Pass-through stand-in so DirectDriveAgent always has a Brain."""

    def step(self, features: FeatureArray) -> npt.NDArray[np.float32]:
        return np.asarray(features, dtype=np.float32)


class DirectDriveAgent:
    """Compose Eye → Brain → Policy. Skips the fly body (track B)."""

    def __init__(
        self,
        eye: Eye,
        policy: Policy,
        brain: Brain | None = None,
    ) -> None:
        self.eye = eye
        self.policy = policy
        self.brain: Brain = brain if brain is not None else IdentityBrain()

    def reset(self) -> None:
        self.eye.reset()
        self.brain.reset()
        self.policy.reset()

    def act(self, frame: FrameArray) -> ControlVector:
        features = self.eye.encode(frame)
        features = require_features(features, self.eye.output_size)
        features = self.brain.step(features)
        return self.policy.act(features).clipped()
