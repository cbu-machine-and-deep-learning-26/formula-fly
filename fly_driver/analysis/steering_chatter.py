"""Measuring steering chatter on a real drive (GH-104).

``mean_abs_steer`` in ``updates.csv`` cannot tell a car that flips left and right from one
that holds a steady lock, and in training it also carries exploration noise. This module
records every frame of a drive and counts what the steering actually did, split into
straights and corners, because holding a lock through a corner is correct and flipping on a
straight is not.

A **flip** is the steering crossing from one side of a small deadband to the other. Values
inside the deadband keep the last side, so ``+1, 0, -1`` is one flip, and a car wobbling
around dead centre is not counted as flipping.

Only numpy and the shared interface are imported here, so the numbers can be tested without
torch. :mod:`fly_driver.policies.sampled_policy` supplies the noise-on policy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from fly_driver.envs.segments import Segment
from fly_driver.interface import ControlVector

__all__ = [
    "FLIP_DEADBAND",
    "FULL_LOCK",
    "DriveTrace",
    "SmoothedSteering",
    "SteeringStats",
    "combine_stats",
    "count_flips",
    "drive",
    "straight_exit_speeds",
    "straight_mask",
    "steering_stats",
    "trace_stats",
    "zone_outcome",
]

#: Steering within this of centre counts as neither side: 5% of lock, about 0.9 degrees.
FLIP_DEADBAND = 0.05

#: Steering at or past this counts as full lock. Below 1 so a clipped float still counts.
FULL_LOCK = 0.99

FloatArray = npt.NDArray[np.float64]


def count_flips(steer: Sequence[float] | FloatArray, deadband: float = FLIP_DEADBAND) -> int:
    """Count how many times the steering crosses from one side of the deadband to the other.

    Args:
        steer: Steering commands in time order.
        deadband: Values with ``abs(value) <= deadband`` keep the previous side.

    Returns:
        The number of crossings.

    Raises:
        ValueError: If ``deadband`` is negative.
    """
    if deadband < 0.0:
        raise ValueError(f"deadband must be non-negative, got {deadband}")
    values = np.asarray(steer, dtype=np.float64)
    sides = np.sign(values[np.abs(values) > deadband])
    return int(np.count_nonzero(sides[1:] != sides[:-1]))


@dataclass(frozen=True)
class SteeringStats:
    """Raw counts for a stretch of driving, kept as counts so stretches can be pooled.

    Args:
        frames: Frames in the stretch.
        seconds: Their length in simulated time.
        flips: Steering flips, from :func:`count_flips`.
        abs_steer_sum: Sum of ``abs(steer)``.
        full_lock_frames: Frames at or past :data:`FULL_LOCK`.
        step_sum: Sum of ``abs(steer[t] - steer[t - 1])`` over adjacent frames.
        step_count: How many adjacent pairs that covers.
    """

    frames: int = 0
    seconds: float = 0.0
    flips: int = 0
    abs_steer_sum: float = 0.0
    full_lock_frames: int = 0
    step_sum: float = 0.0
    step_count: int = 0

    @property
    def flips_per_s(self) -> float:
        """Flips per simulated second, NaN for an empty stretch."""
        return self.flips / self.seconds if self.seconds > 0.0 else float("nan")

    @property
    def mean_abs_steer(self) -> float:
        """Mean ``abs(steer)``, NaN for an empty stretch."""
        return self.abs_steer_sum / self.frames if self.frames else float("nan")

    @property
    def full_lock_share(self) -> float:
        """Share of frames at full lock, NaN for an empty stretch."""
        return self.full_lock_frames / self.frames if self.frames else float("nan")

    @property
    def mean_step(self) -> float:
        """Mean frame-to-frame change in steering, NaN with no adjacent frames."""
        return self.step_sum / self.step_count if self.step_count else float("nan")


def _contiguous_runs(keep: npt.NDArray[np.bool_]) -> list[tuple[int, int]]:
    """``(start, stop)`` index pairs of each run of True in ``keep``."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], keep, [False])).astype(np.int8)))
    return [(int(start), int(stop)) for start, stop in zip(edges[::2], edges[1::2], strict=True)]


def steering_stats(
    steer: Sequence[float] | FloatArray,
    frame_rate_hz: float,
    *,
    keep: npt.NDArray[np.bool_] | None = None,
    deadband: float = FLIP_DEADBAND,
) -> SteeringStats:
    """Summarise a steering series, optionally only the frames in ``keep``.

    Each run of kept frames is counted on its own, so a change of steering across a gap, such
    as the far side of a corner when ``keep`` marks straights, is never a flip or a step.

    Args:
        steer: Steering commands in time order.
        frame_rate_hz: Frames per simulated second.
        keep: Boolean mask the same length as ``steer``. ``None`` keeps every frame.
        deadband: See :func:`count_flips`.

    Raises:
        ValueError: If ``frame_rate_hz`` is not positive or ``keep`` has the wrong length.
    """
    if frame_rate_hz <= 0.0:
        raise ValueError(f"frame_rate_hz must be positive, got {frame_rate_hz}")
    values = np.asarray(steer, dtype=np.float64)
    kept = np.ones(values.shape, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    if kept.shape != values.shape:
        raise ValueError(f"keep has shape {kept.shape} but steer has shape {values.shape}")
    pooled = SteeringStats()
    for start, stop in _contiguous_runs(kept):
        run = values[start:stop]
        pooled = combine_stats(
            (
                pooled,
                SteeringStats(
                    frames=int(run.size),
                    seconds=run.size / frame_rate_hz,
                    flips=count_flips(run, deadband),
                    abs_steer_sum=float(np.abs(run).sum()),
                    full_lock_frames=int(np.count_nonzero(np.abs(run) >= FULL_LOCK)),
                    step_sum=float(np.abs(np.diff(run)).sum()),
                    step_count=int(run.size - 1),
                ),
            )
        )
    return pooled


def combine_stats(stats: Iterable[SteeringStats]) -> SteeringStats:
    """Pool stretches of driving by adding their counts."""
    pooled = SteeringStats()
    for item in stats:
        pooled = SteeringStats(
            frames=pooled.frames + item.frames,
            seconds=pooled.seconds + item.seconds,
            flips=pooled.flips + item.flips,
            abs_steer_sum=pooled.abs_steer_sum + item.abs_steer_sum,
            full_lock_frames=pooled.full_lock_frames + item.full_lock_frames,
            step_sum=pooled.step_sum + item.step_sum,
            step_count=pooled.step_count + item.step_count,
        )
    return pooled


def straight_mask(
    arclengths: Sequence[float] | FloatArray, segments: Sequence[Segment]
) -> npt.NDArray[np.bool_]:
    """Whether each arclength lies on a straight rather than in a corner.

    Args:
        arclengths: Positions along the centreline, in metres.
        segments: The whole lap, from :func:`~fly_driver.envs.segments.find_segments`.

    Returns:
        A boolean array the same length as ``arclengths``. A position outside every segment
        is False.
    """
    positions = np.asarray(arclengths, dtype=np.float64)
    straight = np.zeros(positions.shape, dtype=bool)
    for segment in segments:
        if not segment.is_corner:
            straight |= (positions >= segment.start_m) & (positions < segment.end_m)
    return straight


@dataclass(frozen=True)
class DriveTrace:
    """Every frame of one drive.

    Args:
        seed: The reset seed, which fixes the start on a random-start env.
        start_m: Arclength the drive began at.
        distance_m: Net forward distance driven.
        outcome: ``lap``, ``crashed``, ``stalled``, ``diverged``, ``truncated`` or ``capped``.
        steer: Commanded steering, one value per frame.
        throttle: Commanded throttle.
        brake: Commanded brake.
        kingpin_rad: Left front kingpin angle after each frame, NaN if the env has none.
        speed_mps: Speed after each frame.
        arclength_m: Arclength after each frame.
        lateral_m: Distance from the centreline after each frame.
    """

    seed: int
    start_m: float
    distance_m: float
    outcome: str
    steer: FloatArray
    throttle: FloatArray
    brake: FloatArray
    kingpin_rad: FloatArray
    speed_mps: FloatArray
    arclength_m: FloatArray
    lateral_m: FloatArray

    @property
    def end_m(self) -> float:
        """Arclength where the drive ended, or where it began if it never moved."""
        return float(self.arclength_m[-1]) if self.arclength_m.size else self.start_m


def _kingpin_angle(env: Any) -> float:
    """The left front kingpin's angle, or NaN for an env with no MuJoCo car."""
    model = getattr(env, "model", None)
    if model is None:
        return float("nan")
    return float(env.data.qpos[model.joint("steer_fl").qposadr[0]])


def _outcome(info: dict[str, Any], *, truncated: bool, laps: int) -> str:
    """Why the drive ended, most specific first. A drive that lapped counts as a lap."""
    if laps:
        return "lap"
    for name in ("diverged", "crashed", "stalled"):
        if info.get(name):
            return name
    return "truncated" if truncated else "capped"


def drive(env: Any, agent: Any, *, seed: int, max_steps: int | None = None) -> DriveTrace:
    """Drive one episode and record every frame.

    Args:
        env: Anything with the env's ``reset`` and ``step``.
        agent: Anything with ``reset(seed=...)`` and ``act(frame)``.
        seed: Passed to both resets.
        max_steps: Stop after this many frames, with outcome ``capped``. ``None`` leaves it to
            the env.

    Returns:
        The drive's :class:`DriveTrace`.

    Raises:
        ValueError: If ``max_steps`` is below 1.
    """
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be at least 1, got {max_steps}")
    frame, info = env.reset(seed=seed)
    agent.reset(seed=seed)
    start_m = float(info.get("arclength_m", 0.0))
    rows: list[tuple[float, ...]] = []
    distance_m = 0.0
    laps = 0
    terminated = truncated = False
    while not (terminated or truncated) and (max_steps is None or len(rows) < max_steps):
        control = ControlVector.from_any(agent.act(frame))
        frame, _, terminated, truncated, info = env.step(control)
        distance_m += float(info.get("progress_m", 0.0))
        laps += bool(info.get("lap_complete"))
        rows.append(
            (
                control.steer,
                control.throttle,
                control.brake,
                _kingpin_angle(env),
                float(info.get("speed_mps", 0.0)),
                float(info.get("arclength_m", 0.0)),
                float(info.get("lateral_m", 0.0)),
            )
        )
    columns = np.asarray(rows, dtype=np.float64).reshape(-1, 7).T
    return DriveTrace(
        seed=seed,
        start_m=start_m,
        distance_m=distance_m,
        outcome=_outcome(info, truncated=bool(truncated), laps=laps),
        steer=columns[0],
        throttle=columns[1],
        brake=columns[2],
        kingpin_rad=columns[3],
        speed_mps=columns[4],
        arclength_m=columns[5],
        lateral_m=columns[6],
    )


class SmoothedSteering:
    """A driver whose steering passes through a first-order low-pass filter at run time.

    This asks what the steering chatter costs a policy that was not trained with the filter,
    so it is a one-sided test. If the car drives better, chatter was costing something. If it
    drives worse, that clears nothing, because the policy learned with its own steering in the
    loop and the filter adds a lag it never saw. Throttle and brake pass through untouched.

    Args:
        agent: Anything with ``reset(seed=...)`` and ``act(frame)``.
        blend: Share of the gap to the new command closed each frame, in ``(0, 1]``. 1 changes
            nothing. At 50 Hz, 0.5 is a time constant of about 29 ms and 0.2 about 90 ms.

    Raises:
        ValueError: If ``blend`` is outside ``(0, 1]``.
    """

    def __init__(self, agent: Any, blend: float) -> None:
        if not 0.0 < blend <= 1.0:
            raise ValueError(f"blend must be in (0, 1], got {blend}")
        self.agent = agent
        self.blend = float(blend)
        self._steer = 0.0

    def reset(self, seed: int | None = None) -> None:
        """Start an episode with the steering centred, as the car starts."""
        self.agent.reset(seed=seed)
        self._steer = 0.0

    def act(self, frame: Any) -> ControlVector:
        """The wrapped driver's control, with its steering filtered."""
        control = ControlVector.from_any(self.agent.act(frame))
        self._steer += self.blend * (control.steer - self._steer)
        return ControlVector(steer=self._steer, throttle=control.throttle, brake=control.brake)


def zone_outcome(trace: DriveTrace, start_m: float, end_m: float) -> str:
    """What a drive did at a stretch of track.

    Args:
        trace: The drive.
        start_m: Where the stretch begins.
        end_m: Where it ends.

    Returns:
        ``started_in_zone`` if the drive began there, because it never made the approach and
        must not count as getting through. Otherwise ``not_reached``, ``ended_in_zone``, or
        ``passed`` if the drive left it and carried on.
    """
    if start_m <= trace.start_m <= end_m:
        return "started_in_zone"
    inside = (trace.arclength_m >= start_m) & (trace.arclength_m <= end_m)
    if not inside.any():
        return "not_reached"
    return "ended_in_zone" if inside[-1] else "passed"


def straight_exit_speeds(trace: DriveTrace, segments: Sequence[Segment]) -> dict[str, list[float]]:
    """Speed on the last frame of each straight the drive drove off the end of.

    A straight the drive ended on has no exit, because it never finished the straight.

    Args:
        trace: The drive.
        segments: The whole lap.

    Returns:
        Exit speeds in m/s, keyed by segment name. A straight driven twice has two.
    """
    exits: dict[str, list[float]] = {}
    for segment in segments:
        if segment.is_corner:
            continue
        inside = (trace.arclength_m >= segment.start_m) & (trace.arclength_m < segment.end_m)
        for _, stop in _contiguous_runs(inside):
            if stop < inside.size:
                exits.setdefault(segment.name, []).append(float(trace.speed_mps[stop - 1]))
    return exits


def trace_stats(
    trace: DriveTrace,
    frame_rate_hz: float,
    segments: Sequence[Segment],
    *,
    deadband: float = FLIP_DEADBAND,
) -> dict[str, SteeringStats]:
    """A drive's steering stats for the whole drive, its straights and its corners.

    Args:
        trace: The drive.
        frame_rate_hz: Frames per simulated second.
        segments: The whole lap.
        deadband: See :func:`count_flips`.

    Returns:
        ``{"all": ..., "straight": ..., "corner": ...}``.
    """
    straight = straight_mask(trace.arclength_m, segments)
    return {
        "all": steering_stats(trace.steer, frame_rate_hz, deadband=deadband),
        "straight": steering_stats(trace.steer, frame_rate_hz, keep=straight, deadband=deadband),
        "corner": steering_stats(trace.steer, frame_rate_hz, keep=~straight, deadband=deadband),
    }
