"""Overriding one control channel at run time, to find which one limits the car (GH-104).

A trained driver's steering and pedals are hard to judge together. These overrides replace one
of them while the driver keeps running every frame, so its eye and brain stay live and its other
channel still sees the drive it would normally see:

- ``own``: the driver's own steering and pedals, the control.
- ``gov_<k>``: its own steering, with a speed governor at ``k`` times the speed profile.
- ``vale_<kmh>``: its own steering and pedals, with a speed cap only on the way into Vale.
- ``script_own``: scripted smooth steering, with its own pedals.
- ``script_gov_<k>``: scripted steering and the governor, which checks the harness itself.

Like the other run-time tests in this package, these are one-sided. The policy never trained
with an override, so one that hurts doesn't clear the channel it replaced.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from fly_driver.interface import ControlVector

__all__ = [
    "VALE_CAP_ZONE_M",
    "ChannelOverride",
    "ChannelSpec",
    "WatchedEnv",
    "governor_pedals",
    "scripted_steer",
]

#: Where ``vale_<kmh>`` applies: from before the braking zone to where most drives have ended.
VALE_CAP_ZONE_M = (5300.0, 5600.0)

#: Above the target by this factor before the governor brakes, so it doesn't chatter on the line.
_GOVERNOR_BRAKE_MARGIN = 1.02

_NUMBER = r"[0-9]+(?:\.[0-9]+)?"
_CHANNEL_LABEL = re.compile(rf"(?:(script)_)?(?:own|gov_({_NUMBER}))|vale_({_NUMBER})")


@dataclass(frozen=True)
class ChannelSpec:
    """What a channel label asks for.

    Args:
        script_steer: Replace the steering with :func:`scripted_steer`.
        governor: Hold this multiple of the speed profile with the pedals. ``None`` is off.
        vale_cap_kmh: Cap the speed inside :data:`VALE_CAP_ZONE_M`. ``None`` is off.
    """

    script_steer: bool = False
    governor: float | None = None
    vale_cap_kmh: float | None = None

    @classmethod
    def parse(cls, label: str) -> ChannelSpec:
        """Read a label from the list in the module docstring.

        Raises:
            ValueError: If the label is none of those.
        """
        match = _CHANNEL_LABEL.fullmatch(label)
        if match is None:
            raise ValueError(f"unknown channel condition {label!r}")
        script, governor, vale_cap = match.groups()
        return cls(
            script_steer=script is not None,
            governor=None if governor is None else float(governor),
            vale_cap_kmh=None if vale_cap is None else float(vale_cap),
        )


def governor_pedals(speed_mps: float, target_mps: float) -> tuple[float, float]:
    """Flat out below the target, braking above it, and coasting just over it.

    Args:
        speed_mps: The car's speed.
        target_mps: The speed to hold.

    Returns:
        ``(throttle, brake)``.
    """
    throttle = 1.0 if speed_mps < target_mps else 0.0
    brake = 1.0 if speed_mps > target_mps * _GOVERNOR_BRAKE_MARGIN else 0.0
    return throttle, brake


def scripted_steer(env: Any, info: dict[str, Any], *, gain: float = 2.5) -> float:
    """A smooth steering command that aims at a point ahead on the centreline.

    Args:
        env: A :class:`~fly_driver.envs.practice_track.PracticeTrack`.
        info: The env's latest info.
        gain: Steering per radian of aim error.

    Returns:
        The clipped steering command.
    """
    speed_mps = float(info.get("speed_mps", 0.0))
    lookahead_m = max(12.0, 0.45 * speed_mps)
    aim_at = (float(info["arclength_m"]) + lookahead_m) % env.centerline.length
    (target_x, target_y), _ = env.centerline.pose_at(aim_at)
    root = env.model.joint("car_root").qposadr[0]
    position = env.data.qpos[root : root + 7]
    yaw = 2.0 * math.atan2(position[6], position[3])
    bearing = math.atan2(target_y - position[1], target_x - position[0])
    error = (bearing - yaw + math.pi) % (2.0 * math.pi) - math.pi
    # Positive steering turns the car the negative way: see car.steering_angle_rad.
    return float(np.clip(-gain * error, -1.0, 1.0))


class WatchedEnv:
    """Passes everything through to an env and remembers which env and info are current.

    Args:
        env: The env.
        board: Shared with :class:`ChannelOverride`, so one override can serve several envs.
    """

    def __init__(self, env: Any, board: dict[str, Any]) -> None:
        self.env = env
        self.board = board

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def reset(self, *, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        """Reset the env and put it on the board."""
        frame, info = self.env.reset(seed=seed)
        self.board["env"] = self.env
        self.board["info"] = info
        return frame, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Step the env and put its info on the board."""
        frame, reward, terminated, truncated, info = self.env.step(action)
        self.board["info"] = info
        return frame, reward, terminated, truncated, info


class ChannelOverride:
    """Runs a driver every frame, then replaces the channels its spec names.

    Args:
        driver: Anything with ``reset(seed=...)`` and ``act(frame)``.
        board: The board its :class:`WatchedEnv` writes to.
        spec: Which channels to replace.
        profile_m: Arclengths of the speed profile, ascending.
        profile_mps: Speeds of the speed profile.
        steer_function: Replaces the steering when the spec asks. Tests pass a stub.
    """

    def __init__(
        self,
        driver: Any,
        board: dict[str, Any],
        spec: ChannelSpec,
        profile_m: npt.NDArray[np.float64],
        profile_mps: npt.NDArray[np.float64],
        *,
        steer_function: Callable[[Any, dict[str, Any]], float] = scripted_steer,
    ) -> None:
        self.driver = driver
        self.board = board
        self.spec = spec
        self.profile_m = profile_m
        self.profile_mps = profile_mps
        self.steer_function = steer_function

    def reset(self, seed: int | None = None) -> None:
        """Start an episode; forwarded to the driver."""
        self.driver.reset(seed=seed)

    def act(self, frame: Any) -> ControlVector:
        """The driver's control, with the spec's channels replaced."""
        control = ControlVector.from_any(self.driver.act(frame))
        env, info = self.board["env"], self.board["info"]
        arclength_m = float(info.get("arclength_m", 0.0))
        speed_mps = float(info.get("speed_mps", 0.0))
        steer, throttle, brake = control.steer, control.throttle, control.brake
        if self.spec.script_steer:
            steer = self.steer_function(env, info)
        if self.spec.governor is not None:
            profile = float(np.interp(arclength_m, self.profile_m, self.profile_mps))
            throttle, brake = governor_pedals(speed_mps, self.spec.governor * profile)
        if self.spec.vale_cap_kmh is not None:
            low, high = VALE_CAP_ZONE_M
            if low <= arclength_m <= high and speed_mps > self.spec.vale_cap_kmh / 3.6:
                throttle, brake = 0.0, 1.0
        return ControlVector(steer=steer, throttle=throttle, brake=brake)
