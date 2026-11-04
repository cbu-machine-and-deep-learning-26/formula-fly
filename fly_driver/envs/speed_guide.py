"""A target speed round the lap, for a reward that teaches the policy to brake (GH-104).

The policy doesn't brake for corners: it reaches Vale at about 130 km/h, still accelerating, and
none of those arrivals get through. With a governor holding 60% of the speed profile, its own
steering gets through 14 times in 20. So it needs to learn to slow down, and that needs a signal
its own pedals can act on straight away. A crash a few seconds later is a weak one.

The target has a floor, because a fraction of the slowest corner's speed can fall below the env's
30 km/h stall rule, and a car obeying the target must not be ended for it.

This reports how far the car is above a fraction of
:func:`~fly_driver.envs.braking_line.speed_profile`, for
:class:`~fly_driver.envs.lap_time_reward.LapTimeReward` to charge for. The fraction rises through
training, from a speed the policy can drive at up to what the car can really do. It never touches
the controls or the physics, so every drive is one the policy really drove, and a checkpoint can't
come to depend on it in order to stay on the road.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fly_driver.envs.braking_line import speed_profile
from fly_driver.envs.centerline import Centerline

__all__ = ["SpeedGuide", "SpeedTarget"]


@dataclass(frozen=True)
class SpeedGuide:
    """How the target speed changes through training.

    Args:
        start_factor: Fraction of the speed profile at the first training step.
        end_factor: Fraction once the ramp is over.
        ramp_steps: Training steps the fraction takes to get from start to end. 0 uses
            ``end_factor`` from the first step, and keeps the shaping's guarantee exact.
        floor_mps: The target is never below this. The default is about 40 km/h.

    Raises:
        ValueError: If a factor isn't positive, ``ramp_steps`` is negative, or the floor is.
    """

    start_factor: float
    end_factor: float
    ramp_steps: int
    floor_mps: float = 11.0

    def __post_init__(self) -> None:
        if self.start_factor <= 0.0 or self.end_factor <= 0.0:
            raise ValueError(
                f"factors must be positive, got {self.start_factor} and {self.end_factor}"
            )
        if self.ramp_steps < 0:
            raise ValueError(f"ramp_steps must be non-negative, got {self.ramp_steps}")
        if self.floor_mps < 0.0:
            raise ValueError(f"floor_mps must be non-negative, got {self.floor_mps}")

    def factor(self, training_steps: int) -> float:
        """The fraction of the speed profile after this many training steps."""
        if self.ramp_steps == 0:
            return self.end_factor
        progress = min(1.0, max(0, training_steps) / self.ramp_steps)
        return self.start_factor + progress * (self.end_factor - self.start_factor)


class SpeedTarget:
    """A :class:`SpeedGuide` applied to one circuit's speed profile.

    Args:
        guide: The schedule.
        centerline: The circuit.
    """

    def __init__(self, guide: SpeedGuide, centerline: Centerline) -> None:
        profile = speed_profile(centerline)
        self.guide = guide
        self._lap_m = float(centerline.length)
        self._arclength_m = np.asarray(profile.arclength_m, dtype=np.float64)
        self._speed_mps = np.asarray(profile.speed_mps, dtype=np.float64)
        self._factor = guide.factor(0)

    def set_training_steps(self, steps: int) -> None:
        """Move the target to where this many training steps have brought it."""
        self._factor = self.guide.factor(steps)

    def speed_mps(self, arclength_m: float) -> float:
        """The target speed at a point on the lap, wrapping across the start line."""
        profile = np.interp(arclength_m, self._arclength_m, self._speed_mps, period=self._lap_m)
        return max(self.guide.floor_mps, self._factor * float(profile))

    def over_speed_mps(self, arclength_m: float, speed_mps: float) -> float:
        """How far ``speed_mps`` is above the target here, 0 if it isn't."""
        return max(0.0, speed_mps - self.speed_mps(arclength_m))
