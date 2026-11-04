"""Does a steering wobble alone stop a car on a straight, and at what speed? (GH-104)

No policy runs here, so this is the physics alone: flat throttle from rest along a straight, a
scripted aim holding the car to the centreline, and a wobble added on top. The wobble is either
a square wave or a trained policy's recorded steering.

A recorded wobble doesn't react to the car, so read what it shows as an estimate of the speed
ceiling that steering imposes, not as an exact limit.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from fly_driver.analysis.channel_overrides import scripted_steer
from fly_driver.envs.segments import Segment

__all__ = ["StraightRun", "drive_straight", "longest_straights", "recorded_wobble", "square_wobble"]

#: Stop this far before a straight's end, so the corner after it plays no part.
_STOP_BEFORE_END_M = 25.0

#: Start this far in, so the car isn't placed on a segment boundary.
_START_AFTER_BEGIN_M = 5.0

Wobble = Callable[[int], float]


@dataclass(frozen=True)
class StraightRun:
    """One drive along a straight.

    Args:
        end_speed_kmh: Speed on the last frame.
        distance_m: Forward distance driven.
        crashed: Whether the env ended the drive at the track limit.
    """

    end_speed_kmh: float
    distance_m: float
    crashed: bool


class _FixedStart:
    """Stands in for the env's generator, so a random-start reset lands on a chosen arclength."""

    def __init__(self, start_m: float) -> None:
        self.start_m = start_m

    def uniform(self, low: float, high: float) -> float:
        del low, high
        return self.start_m


def square_wobble(flip_frames: int, amplitude: float) -> Wobble:
    """``+amplitude`` then ``-amplitude``, flipping every ``flip_frames`` frames.

    Raises:
        ValueError: If ``flip_frames`` is below 1.
    """
    if flip_frames < 1:
        raise ValueError(f"flip_frames must be at least 1, got {flip_frames}")

    def wobble(frame: int) -> float:
        return amplitude if (frame // flip_frames) % 2 == 0 else -amplitude

    return wobble


def recorded_wobble(
    steer: npt.NDArray[np.float64], *, offset: int = 0, gain: float = 1.0
) -> Wobble:
    """A recorded steering series, looping, from ``offset`` and multiplied by ``gain``.

    Raises:
        ValueError: If ``steer`` is empty.
    """
    if steer.size == 0:
        raise ValueError("steer is empty")

    def wobble(frame: int) -> float:
        return gain * float(steer[(offset + frame) % steer.size])

    return wobble


def longest_straights(segments: Sequence[Segment], count: int) -> list[Segment]:
    """The ``count`` longest straights, longest first."""
    straights = [segment for segment in segments if not segment.is_corner]
    return sorted(straights, key=lambda segment: -segment.length_m)[:count]


def drive_straight(env: Any, segment: Segment, wobble: Wobble) -> StraightRun:
    """Drive a straight flat out with a wobble on top of a scripted aim.

    Args:
        env: A :class:`~fly_driver.envs.practice_track.PracticeTrack` with ``random_start=True``.
        segment: The straight.
        wobble: Added to the scripted steering, by frame number.

    Returns:
        How it went.
    """
    goal_m = segment.length_m - _STOP_BEFORE_END_M
    env.rng = _FixedStart(segment.start_m + _START_AFTER_BEGIN_M)
    _, info = env.reset()
    travelled_m = 0.0
    frame = 0
    done = False
    while not done and travelled_m < goal_m:
        steer = float(np.clip(scripted_steer(env, info) + wobble(frame), -1.0, 1.0))
        _, _, terminated, truncated, info = env.step((steer, 1.0, 0.0))
        done = bool(terminated or truncated)
        travelled_m += float(info["progress_m"])
        frame += 1
    return StraightRun(
        end_speed_kmh=float(info["speed_mps"]) * 3.6,
        distance_m=travelled_m,
        crashed=bool(info.get("crashed")),
    )
