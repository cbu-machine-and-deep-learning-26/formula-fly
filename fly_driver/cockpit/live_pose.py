"""The cockpit fly's pose from the live controls and brain (GH-26).

Direct drive has no MuJoCo body, so the fly in Assetto Corsa is posed from what the loop
already computes: the :class:`~fly_driver.interface.ControlVector` and the central complex's
activity (:class:`~fly_driver.brains.central_complex.CentralComplexBrain`).

- **Front legs** turn with ``steer``. ``wheel_deg`` is the wheel angle, positive clockwise
  from the driver's seat, the same sign as ``steer`` (positive is right, as in
  :mod:`fly_driver.envs.car` and the Assetto Corsa bridge).
- **Hind legs** press the pedals: the right foot with ``throttle``, the left with ``brake``.
- **Head yaw** follows the compass ring's heading bump: the angle of the activation-weighted
  mean over the compass neurons, cell ``i`` of ``n`` sitting at ``2 pi i / n``. The yaw is
  ``HEAD_YAW_LIMIT_DEG * sin(heading)``, so it stays continuous where the bump wraps round
  the ring. Positive turns the head to the driver's right.
- **Wings and antennae** move a little with the descending neurons, the outputs the policy
  reads: each wing with the mean absolute activity of its half, each antenna with the
  signed mean of its half.

:func:`fly_pose` is pure. :func:`smooth_pose` low-passes successive poses so the mesh does
not jitter.
"""

from __future__ import annotations

import math
from dataclasses import astuple, dataclass

import numpy as np
import numpy.typing as npt

from fly_driver.brains.populations import PopulationCounts
from fly_driver.interface import ControlVector

__all__ = ["LIMITS_DEG", "FlyPose", "compass_heading", "fly_pose", "smooth_pose"]

WHEEL_LOCK_DEG = 90.0
PEDAL_PRESS_LIMIT_DEG = 25.0
HEAD_YAW_LIMIT_DEG = 30.0
WING_LIMIT_DEG = 10.0
ANTENNA_LIMIT_DEG = 15.0


@dataclass(frozen=True)
class FlyPose:
    """Joint angles for the cockpit fly, in degrees. Field order is the wire order."""

    wheel_deg: float = 0.0
    throttle_press_deg: float = 0.0
    brake_press_deg: float = 0.0
    head_yaw_deg: float = 0.0
    wing_left_deg: float = 0.0
    wing_right_deg: float = 0.0
    antenna_left_deg: float = 0.0
    antenna_right_deg: float = 0.0

    def values(self) -> tuple[float, ...]:
        """The angles in field order."""
        return astuple(self)


#: ``(low, high)`` per :class:`FlyPose` field. Every pose :func:`fly_pose` returns is inside.
LIMITS_DEG: dict[str, tuple[float, float]] = {
    "wheel_deg": (-WHEEL_LOCK_DEG, WHEEL_LOCK_DEG),
    "throttle_press_deg": (0.0, PEDAL_PRESS_LIMIT_DEG),
    "brake_press_deg": (0.0, PEDAL_PRESS_LIMIT_DEG),
    "head_yaw_deg": (-HEAD_YAW_LIMIT_DEG, HEAD_YAW_LIMIT_DEG),
    "wing_left_deg": (0.0, WING_LIMIT_DEG),
    "wing_right_deg": (0.0, WING_LIMIT_DEG),
    "antenna_left_deg": (-ANTENNA_LIMIT_DEG, ANTENNA_LIMIT_DEG),
    "antenna_right_deg": (-ANTENNA_LIMIT_DEG, ANTENNA_LIMIT_DEG),
}


def compass_heading(compass_rates: npt.ArrayLike) -> float | None:
    """Heading of the ring's bump in radians, ``(-pi, pi]``, or ``None`` with no bump.

    Only positive activity votes; ``tanh`` rates below zero are a silent cell, not a vote for
    the opposite side of the ring.
    """
    weights = np.maximum(np.asarray(compass_rates, dtype=np.float64), 0.0)
    angles = 2.0 * np.pi * np.arange(weights.size) / max(weights.size, 1)
    x, y = float(weights @ np.cos(angles)), float(weights @ np.sin(angles))
    if math.hypot(x, y) < 1e-9:
        return None
    return math.atan2(y, x)


def fly_pose(
    control: ControlVector,
    rates: npt.ArrayLike | None = None,
    populations: PopulationCounts | None = None,
) -> FlyPose:
    """Map one frame's control and brain activity to joint angles.

    Args:
        control: What the policy sent the car.
        rates: The brain's bounded activity, ``brain.rates()``, ``(neuron_count,)``.
            ``None`` (a driver with no brain) holds head, wings and antennae at rest.
        populations: ``brain.populations``; says which cells are compass and descending.

    Raises:
        ValueError: If ``rates`` is given without ``populations`` or does not match them.
    """
    pose = {
        "wheel_deg": control.steer * WHEEL_LOCK_DEG,
        "throttle_press_deg": control.throttle * PEDAL_PRESS_LIMIT_DEG,
        "brake_press_deg": control.brake * PEDAL_PRESS_LIMIT_DEG,
    }
    if rates is not None:
        if populations is None:
            raise ValueError("rates need populations to find the compass and descending cells")
        activity = np.nan_to_num(np.asarray(rates, dtype=np.float64))
        if activity.shape != (populations.total,):
            raise ValueError(f"rates shape {activity.shape} != ({populations.total},)")
        heading = compass_heading(activity[: populations.compass])
        pose["head_yaw_deg"] = 0.0 if heading is None else HEAD_YAW_LIMIT_DEG * math.sin(heading)
        left, right = np.array_split(activity[populations.total - populations.descending :], 2)
        pose["wing_left_deg"] = WING_LIMIT_DEG * float(np.mean(np.abs(left)))
        pose["wing_right_deg"] = WING_LIMIT_DEG * float(np.mean(np.abs(right)))
        pose["antenna_left_deg"] = ANTENNA_LIMIT_DEG * float(np.mean(left))
        pose["antenna_right_deg"] = ANTENNA_LIMIT_DEG * float(np.mean(right))
    return FlyPose(**{name: _clip(name, value) for name, value in pose.items()})


def smooth_pose(previous: FlyPose | None, target: FlyPose, alpha: float = 0.5) -> FlyPose:
    """Move ``alpha`` of the way from ``previous`` to ``target``; ``None`` jumps straight there.

    At 50 Hz, ``alpha=0.5`` settles within about 0.1 s: enough to hide frame-to-frame noise in
    the brain without visibly lagging the wheel.
    """
    if previous is None:
        return target
    return FlyPose(
        *(
            old + alpha * (new - old)
            for old, new in zip(previous.values(), target.values(), strict=True)
        )
    )


def _clip(name: str, value: float) -> float:
    low, high = LIMITS_DEG[name]
    return min(max(float(value), low), high)
