"""Human laps for behaviour cloning (GH-29): replay inputs, frames, cached eye features.

A human drive arrives as an Assetto Corsa replay (``.acreplay``). The replay holds the
driver's inputs and the car's pose every ``recording interval`` (30 ms at AC's default
quality), but no images. The frames come from playing the replay back in AC through our
nose camera (``scripts/bc_replay_capture.py``), one frame per 20 ms instant, so frame ``k``
is replay time ``k / FRAME_RATE_HZ`` and pairs with the inputs interpolated there.

The pieces, in the order a drive goes through them:

- :func:`parse_replay` reads the player car's inputs and pose from a version-16 replay.
- :func:`replay_controls` turns them into our ``(steer, throttle, brake)`` at any times,
  with the same convention and ranges the env takes (:class:`~fly_driver.interface.ControlVector`).
- :class:`HumanLaps` is a drive's frames and controls on disk (``.npz``), in segments
  that the eye and brain run through in order, resetting at each segment's start.
- :func:`cache_eye_features` runs the frozen eye over a :class:`HumanLaps` once and keeps
  the features beside it, so training never runs the eye.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from fly_driver.interface import FRAME_RATE_HZ, FRAME_SHAPE, Eye

__all__ = [
    "STEER_LOCK_DEG",
    "HumanLaps",
    "Replay",
    "cache_eye_features",
    "eye_cache_key",
    "parse_replay",
    "replay_controls",
]

#: Steering-wheel degrees at full lock, per car: ``steer = ±1`` in our env. The SF70H's
#: ``steer_map.csv`` maps ``±0.4006`` vJoy to ``±180°``, and the env sends ``steer * 0.4006``.
STEER_LOCK_DEG: dict[str, float] = {"ks_ferrari_sf70h": 180.0}

_REPLAY_VERSION = 16
# One CarFrame (abchouhan/acreplay-parser's layout, checked against this project's own
# drive and AC's telemetry export) plus 20 bytes and 4 per wing before the next frame.
_CAR_FRAME_BYTES = 256
_FRAME_GAP_BYTES = 20


@dataclass(frozen=True)
class Replay:
    """The player car's track in an AC replay, one row per recorded frame.

    Attributes:
        car: AC's car id, e.g. ``ks_ferrari_sf70h``.
        track: AC's track and layout, e.g. ``ks_silverstone-gp``.
        interval_s: Time between recorded frames.
        time_s: ``(n,)`` replay time of each frame, from 0.
        position_m: ``(n, 3)`` world position, y up.
        speed_mps: ``(n,)`` speed.
        steer_deg: ``(n,)`` steering-wheel angle, positive right.
        gas: ``(n,)`` throttle pedal in ``[0, 1]``.
        brake: ``(n,)`` brake pedal in ``[0, 1]``.
        gear: ``(n,)`` AC's gear: 0 reverse, 1 neutral, 2 first.
        lap: ``(n,)`` 0-based lap the car is on.
    """

    car: str
    track: str
    interval_s: float
    time_s: npt.NDArray[np.float64]
    position_m: npt.NDArray[np.float32]
    speed_mps: npt.NDArray[np.float32]
    steer_deg: npt.NDArray[np.float32]
    gas: npt.NDArray[np.float32]
    brake: npt.NDArray[np.float32]
    gear: npt.NDArray[np.uint8]
    lap: npt.NDArray[np.uint8]

    @property
    def duration_s(self) -> float:
        """Time of the last recorded frame."""
        return float(self.time_s[-1])


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.offset = 0

    def take(self, fmt: str) -> Any:
        values = struct.unpack_from(fmt, self.data, self.offset)
        self.offset += struct.calcsize(fmt)
        return values[0] if len(values) == 1 else values

    def text(self) -> str:
        length = self.take("<I")
        value = self.data[self.offset : self.offset + length].decode("latin-1")
        self.offset += length
        return value


def parse_replay(path: str | os.PathLike[str]) -> Replay:
    """Read the first car of a version-16 ``.acreplay`` (the player's, in a Hotlap).

    Raises:
        ValueError: On another replay version, or a file shorter than its header says.
    """
    data = Path(path).read_bytes()
    reader = _Reader(data)
    version = reader.take("<I")
    if version != _REPLAY_VERSION:
        raise ValueError(f"{path}: replay version {version}, only {_REPLAY_VERSION} is supported")
    interval_ms = reader.take("<d")
    reader.text()  # weather
    track, layout = reader.text(), reader.text()
    reader.take("<I")  # car count
    reader.take("<I")  # current recording index
    frame_count = reader.take("<I")
    track_objects = reader.take("<I")
    reader.offset += (4 + 12 * track_objects) * frame_count  # sun angles, track objects
    car = reader.text()
    for _ in range(4):  # driver name, nation, team, skin
        reader.text()
    car_frames = reader.take("<I")
    wings = reader.take("<I")
    start = reader.offset + _FRAME_GAP_BYTES
    stride = _CAR_FRAME_BYTES + _FRAME_GAP_BYTES + 4 * wings
    if start + stride * (car_frames - 1) + _CAR_FRAME_BYTES > len(data):
        raise ValueError(f"{path}: truncated, {car_frames} car frames do not fit")
    frames = np.lib.stride_tricks.as_strided(
        np.frombuffer(data, np.uint8, offset=start),
        shape=(car_frames, _CAR_FRAME_BYTES),
        strides=(stride, 1),
    ).copy()

    def floats(offset: int, count: int, dtype: str) -> npt.NDArray[np.float32]:
        width = np.dtype(dtype).itemsize * count
        return frames[:, offset : offset + width].copy().view(dtype).astype(np.float32)

    velocity = floats(164, 3, "<f2")
    return Replay(
        car=car,
        track=f"{track}-{layout}" if layout else track,
        interval_s=interval_ms / 1000.0,
        time_s=np.arange(car_frames, dtype=np.float64) * interval_ms / 1000.0,
        position_m=floats(0, 3, "<f4"),
        speed_mps=np.linalg.norm(velocity, axis=1).astype(np.float32),
        steer_deg=floats(212, 1, "<f2")[:, 0],
        gas=frames[:, 244].astype(np.float32) / 255.0,
        brake=frames[:, 245].astype(np.float32) / 255.0,
        gear=frames[:, 234].copy(),
        lap=frames[:, 246].copy(),
    )


def replay_controls(
    replay: Replay, time_s: npt.ArrayLike, *, steer_lock_deg: float | None = None
) -> npt.NDArray[np.float32]:
    """The driver's ``(steer, throttle, brake)`` at ``time_s``, linearly interpolated.

    Steer is the wheel angle over the car's full lock (:data:`STEER_LOCK_DEG`), clipped to
    ``[-1, 1]``; throttle and brake are the pedals in ``[0, 1]``. These are the env's
    ranges, so a policy's clipped mean compares with them directly.

    Raises:
        KeyError: If the car has no known lock and ``steer_lock_deg`` is not given.
    """
    lock = STEER_LOCK_DEG[replay.car] if steer_lock_deg is None else float(steer_lock_deg)
    times = np.asarray(time_s, dtype=np.float64)
    steer = np.interp(times, replay.time_s, replay.steer_deg) / lock
    throttle = np.interp(times, replay.time_s, replay.gas)
    brake = np.interp(times, replay.time_s, replay.brake)
    controls = np.stack([np.clip(steer, -1.0, 1.0), throttle, brake], axis=1)
    return controls.astype(np.float32)


@dataclass(frozen=True)
class HumanLaps:
    """A human drive as BC data: frames, the controls at each, and contiguous segments.

    Attributes:
        frames: ``(n, *FRAME_SHAPE)`` uint8 RGB, at ``FRAME_RATE_HZ``.
        controls: ``(n, 3)`` float32 ``(steer, throttle, brake)`` in the env's ranges.
        segments: ``(n,)`` int segment id. A segment is a run of consecutive frames (a
            lap, or what is left of one after dropping parked frames); the eye and brain
            reset at its first frame. Ids are non-decreasing.
        laps: ``(n,)`` int lap each frame was driven on, for held-out splits.
    """

    frames: npt.NDArray[np.uint8]
    controls: npt.NDArray[np.float32]
    segments: npt.NDArray[np.int64]
    laps: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        count = len(self.frames)
        if self.frames.dtype != np.uint8 or self.frames.shape[1:] != FRAME_SHAPE:
            raise ValueError(
                f"frames must be uint8 (n, {FRAME_SHAPE}), got {self.frames.dtype} "
                f"{self.frames.shape}"
            )
        for name in ("controls", "segments", "laps"):
            if len(getattr(self, name)) != count:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows for {count} frames")
        if self.controls.shape[1:] != (3,) or not np.all(np.isfinite(self.controls)):
            raise ValueError("controls must be finite (n, 3)")
        low, high = np.array([-1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])
        if np.any(self.controls < low) or np.any(self.controls > high):
            raise ValueError("controls outside steer [-1, 1], throttle and brake [0, 1]")
        if np.any(np.diff(self.segments) < 0):
            raise ValueError("segment ids must be non-decreasing")

    def __len__(self) -> int:
        return len(self.frames)

    def segment_bounds(self) -> list[tuple[int, int]]:
        """``(start, stop)`` index of each segment, in order."""
        starts = np.flatnonzero(np.r_[True, np.diff(self.segments) != 0])
        stops = np.r_[starts[1:], len(self)]
        return list(zip(starts.tolist(), stops.tolist(), strict=True))

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write the drive as one compressed ``.npz``."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            frames=self.frames,
            controls=self.controls,
            segments=self.segments,
            laps=self.laps,
        )
        return target

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> HumanLaps:
        """Read a drive written by :meth:`save`; the constructor checks it."""
        with np.load(Path(path)) as data:
            return cls(
                frames=data["frames"],
                controls=data["controls"].astype(np.float32),
                segments=data["segments"].astype(np.int64),
                laps=data["laps"].astype(np.int64),
            )

    @classmethod
    def from_replay(
        cls,
        replay: Replay,
        frames: npt.NDArray[np.uint8],
        *,
        parked_speed_mps: float = 1.0,
        steer_lock_deg: float | None = None,
    ) -> HumanLaps:
        """Pair frames rendered at ``k / FRAME_RATE_HZ`` of ``replay`` with its controls.

        Frames where the car is parked (under ``parked_speed_mps`` with no throttle) are
        dropped: they teach nothing but "sit still", and a Hotlap starts with the car
        parked. A launch from rest is kept, and each run of kept frames is a segment;
        runs shorter than a second are dropped too.
        """
        times = np.arange(len(frames)) / FRAME_RATE_HZ
        if len(frames) and times[-1] > replay.duration_s + 1e-9:
            raise ValueError(
                f"{len(frames)} frames run to {times[-1]:.2f} s, past the replay's "
                f"{replay.duration_s:.2f} s"
            )
        controls = replay_controls(replay, times, steer_lock_deg=steer_lock_deg)
        speed = np.interp(times, replay.time_s, replay.speed_mps)
        laps = replay.lap[np.searchsorted(replay.time_s, times, side="right") - 1].astype(np.int64)
        kept = ~((speed < parked_speed_mps) & (controls[:, 1] < 0.05))
        # A new segment wherever frames were dropped or the lap changes.
        breaks = np.r_[True, ~kept[:-1] | (np.diff(laps) != 0)]
        segments = np.cumsum(breaks) - 1
        # Segments under a second (a roll before the car parks) are too short to warm a brain.
        lengths = np.bincount(segments[kept], minlength=segments[-1] + 1 if len(segments) else 0)
        kept &= lengths[segments] >= int(FRAME_RATE_HZ)
        return cls(
            frames=frames[kept],
            controls=controls[kept],
            segments=segments[kept],
            laps=laps[kept],
        )


def eye_cache_key(eye_config: Mapping[str, Any]) -> str:
    """A short stable hash of an eye config, so caches of different eyes never mix."""
    text = json.dumps(dict(eye_config), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def cache_eye_features(
    laps: HumanLaps, eye: Eye, path: str | os.PathLike[str]
) -> npt.NDArray[np.float32]:
    """Eye features for every frame, read from ``path`` (an ``.npz``) or computed once and saved.

    The eye resets at each segment's first frame and sees the segment's frames in order,
    exactly as in a closed-loop episode.

    Raises:
        ValueError: If a cached file has the wrong shape for ``laps`` and ``eye``.
    """
    target = Path(path)
    shape = (len(laps), int(eye.feature_dim))
    if target.exists():
        with np.load(target) as cached:
            features = cached["features"]
        if features.shape != shape:
            raise ValueError(f"{target} holds {features.shape} features, expected {shape}")
        return features
    features = np.empty(shape, dtype=np.float32)
    for start, stop in laps.segment_bounds():
        eye.reset()
        for index in range(start, stop):
            features[index] = eye.encode(laps.frames[index])
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, features=features)
    return features
