"""A driving teacher for training only: steering and pedals from physics, on any track (GH-27).

Nov 5, 2026: corner practice gave the policy hundreds of attempts at T3 and Vale, and it crashed
at the same metre of each. Tracing a sampled policy showed why: it arrived 50-100 km/h over what
the corner allows, and its steering was noise -- flipping between full lock either way, riding
5 m off the centreline, so its mean action crashed within 2 s (every deterministic evaluation
ended near 550 m). PPO was not finding either the brake point or the road by trial.

This teacher is a complete driver. Alone it laps Silverstone in 135.6 s at ``fraction`` 0.85
(128.7 s at 0.9; ``scripts/teacher_lap.py``):

- **Steering**: a Stanley controller on the centreline. The wheel angle is
  ``-k_heading * heading_error - atan(k_lateral * lateral / (v + 1))``, as a share of the
  steering lock (``steer`` +1 is right; ``lateral`` and ``heading_error`` are positive left).
- **Pedals**: ``fraction`` of the physics speed profile along the centreline
  (:func:`~fly_driver.envs.braking_line.speed_profile` with ``line="centre"``), read
  ``lookahead_s`` ahead: brake in proportion to the excess, throttle in proportion to the
  shortfall.

It reads the car's arclength, speed, lateral offset and heading error from the env's info --
privileged state the policy never sees. :class:`~fly_driver.training.ppo.PPOLearner` adds
``teacher_coef`` times the pedal error and ``teacher_steer_coef`` times the steering error,
between the policy's *mean* action and these targets, on the states the policy itself visits
(DAgger's distribution, not the teacher's). The policy still drives on fly vision alone, and
the profile comes from the car's physics and the centreline, so the teacher works on any track.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from fly_driver.envs.braking_line import SpeedProfile, speed_profile
from fly_driver.envs.car import CarConfig
from fly_driver.envs.centerline import Centerline

if TYPE_CHECKING:
    import torch

__all__ = ["BrakingTeacher", "DrivingTeacher", "OFF_TARGET", "teacher_loss"]

#: The raw target for a released pedal. :func:`teacher_loss` treats it, and a full pedal (1), as
#: one-sided bounds: a policy already further from the pressed range is not pulled back. On Nov 5 a
#: two-sided -0.5 pulled the brake mean up near 0, exploration noise (std ~0.7) pressed the brake
#: about a quarter of the time, and 382 of the first 600 drives stalled.
OFF_TARGET = -1.0


@dataclass
class DrivingTeacher:
    """Steering, throttle and brake targets from the car's place on the track.

    Args:
        profile: The speed profile along the line the teacher follows.
        lap_length_m: Closed-loop length, so lookups wrap past the line.
        fraction: Share of the profile's speed to aim for.
        lookahead_s: How far ahead, in seconds at the current speed, to read the profile.
        gain_mps: Speed error, in m/s, that asks for a full pedal.
        k_lateral: Stanley gain on the lateral offset.
        k_heading: Gain on the heading error.
        max_steer_rad: The car's steering lock, to turn a wheel angle into ``steer``.
    """

    profile: SpeedProfile
    lap_length_m: float
    fraction: float = 0.85
    lookahead_s: float = 0.3
    gain_mps: float = 5.0
    k_lateral: float = 1.0
    k_heading: float = 1.0
    max_steer_rad: float = CarConfig().max_steer_rad

    @classmethod
    def for_centerline(
        cls,
        centerline: Centerline | None = None,
        *,
        fraction: float = 0.85,
        lookahead_s: float = 0.3,
        gain_mps: float = 5.0,
    ) -> DrivingTeacher:
        """The teacher for ``centerline`` (default: the practice track's)."""
        centerline = centerline or Centerline.load()
        return cls(
            speed_profile(centerline, line="centre"),
            float(centerline.length),
            fraction=fraction,
            lookahead_s=lookahead_s,
            gain_mps=gain_mps,
        )

    def target_speed(
        self, arclength_m: npt.ArrayLike, speed_mps: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """``fraction`` of the profile at ``lookahead_s`` ahead of each car, in m/s."""
        speed = np.asarray(speed_mps, dtype=float)
        ahead = np.mod(
            np.asarray(arclength_m, dtype=float) + speed * self.lookahead_s, self.lap_length_m
        )
        profile = np.interp(
            ahead, self.profile.arclength_m, self.profile.speed_mps, period=self.lap_length_m
        )
        return self.fraction * profile

    def pedal_targets(
        self, arclength_m: npt.ArrayLike, speed_mps: npt.ArrayLike
    ) -> npt.NDArray[np.float32]:
        """``(n, 2)`` raw throttle and brake targets.

        Over the target speed: brake ``excess / gain`` (up to 1), throttle released. Under it:
        throttle ``shortfall / gain`` (up to 1), brake released. A released pedal's target is
        :data:`OFF_TARGET`.
        """
        speed = np.asarray(speed_mps, dtype=float)
        excess = speed - self.target_speed(arclength_m, speed)
        press = np.clip(np.abs(excess) / self.gain_mps, 0.0, 1.0)
        braking = excess > 0.0
        throttle = np.where(braking, OFF_TARGET, press)
        brake = np.where(braking, press, OFF_TARGET)
        return np.stack([throttle, brake], axis=-1).astype(np.float32)

    def steer_targets(
        self,
        lateral_m: npt.ArrayLike,
        heading_error_rad: npt.ArrayLike,
        speed_mps: npt.ArrayLike,
    ) -> npt.NDArray[np.float32]:
        """``(n,)`` ``steer`` targets in ``[-1, 1]``, +1 full right, from a Stanley controller."""
        lateral = np.asarray(lateral_m, dtype=float)
        heading = np.asarray(heading_error_rad, dtype=float)
        speed = np.asarray(speed_mps, dtype=float)
        # A positive wheel angle turns left: away from a car pointing right or sitting right.
        wheel = -self.k_heading * heading - np.arctan2(self.k_lateral * lateral, speed + 1.0)
        return np.clip(-wheel / self.max_steer_rad, -1.0, 1.0).astype(np.float32)

    def targets(
        self,
        arclength_m: npt.ArrayLike,
        speed_mps: npt.ArrayLike,
        lateral_m: npt.ArrayLike,
        heading_error_rad: npt.ArrayLike,
    ) -> npt.NDArray[np.float32]:
        """``(n, 3)`` raw ``(steer, throttle, brake)`` targets, the policy's action order.

        Scalars count as one car.
        """
        arclength, speed, lateral, heading = (
            np.atleast_1d(np.asarray(x, dtype=float))
            for x in (arclength_m, speed_mps, lateral_m, heading_error_rad)
        )
        steer = self.steer_targets(lateral, heading, speed)
        pedals = self.pedal_targets(arclength, speed)
        return np.column_stack([steer, pedals]).astype(np.float32)


#: The name of the pedals-only teacher the first lap-push job ran with.
BrakingTeacher = DrivingTeacher


def teacher_loss(
    prediction: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean squared error, one-sided at both ends of the action range.

    A target at -1 (a released pedal, :data:`OFF_TARGET`, or full left lock) only penalises a
    mean above it, and one at +1 (a full pedal or full right lock) only a mean below it;
    anything between is two-sided. The clip turns a mean past either end into the same action,
    so pulling it back would only shrink the margin exploration noise has before it eases off.

    Args:
        prediction: ``(batch, k)`` policy means.
        targets: ``(batch, k)`` teacher targets in the same columns.
        weights: ``(batch,)`` per-row weights, normalised to mean 1 within the batch; ``None``
            weighs every row the same.
    """
    import torch  # noqa: F811  the env side of this module must import without torch

    error = prediction - targets
    error = torch.where(targets <= OFF_TARGET, error.clamp(min=0.0), error)
    error = torch.where(targets >= 1.0, error.clamp(max=0.0), error)
    squared = error.square().mean(dim=-1)
    if weights is None:
        return squared.mean()
    return (squared * weights).sum() / weights.sum().clamp(min=1e-8)
