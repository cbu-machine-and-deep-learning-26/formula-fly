"""A numpy-only stand-in track for exercising the evaluation harness.

This is not a car model. It exists so the harness, its determinism test, and CI
can run without MuJoCo, torch, or a GPU while still speaking the project
contract: a uint8 RGB frame out, a ``(steer, throttle, brake)`` action in, and
the Gymnasium ``reset``/``step``/``render`` surface every real env will expose.

The dynamics are one-dimensional progress along a track plus a lateral offset.
The track bends (a sinusoidal curvature in progress) and a seeded wind nudges
the car sideways, so steering matters, the seed matters, and a proportional
controller can complete a lap while a passive one drifts off. Reward is the
distance progressed minus a small lateral penalty, with a bonus on lap
completion and a penalty for leaving the track, so standing still never earns
reward (``AGENTS.md`` §11).

``info`` carries the lap contract the harness reads: ``lap_complete`` and
``lap_time`` (seconds, ``None`` until a lap completes).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["DEFAULT_DUMMY_FRAME_SHAPE", "DummyTrackEnv"]

DEFAULT_DUMMY_FRAME_SHAPE = (64, 64, 3)

Frame = npt.NDArray[np.uint8]
Action = npt.NDArray[np.float32]

_ACTION_BOUNDS = ((-1.0, 1.0), (0.0, 1.0), (0.0, 1.0))
_ACTION_NAMES = ("steer", "throttle", "brake")


class DummyTrackEnv:
    """One-dimensional track with a bend, a wind, and a camera-like frame.

    Args:
        frame_shape: Observation shape ``(height, width, 3)``.
        track_length: Lap length in metres.
        half_width: Distance from the centreline at which the car is off track.
        max_steps: Steps before an episode is truncated.
        frame_rate_hz: Simulation rate; one step integrates ``1 / frame_rate_hz``.
        curvature: Amplitude of the sinusoidal lateral drift per metre travelled.
        turns: Number of full bend cycles per lap.
        wind_std: Standard deviation (metres per step) of the seeded lateral wind.
    """

    max_speed = 40.0
    max_accel = 8.0
    max_brake = 15.0
    drag = 0.1
    steer_gain = 0.5
    lap_bonus = 100.0
    off_track_penalty = -10.0
    lateral_penalty = 0.1

    def __init__(
        self,
        *,
        frame_shape: tuple[int, int, int] = DEFAULT_DUMMY_FRAME_SHAPE,
        track_length: float = 200.0,
        half_width: float = 3.0,
        max_steps: int = 1000,
        frame_rate_hz: float = 50.0,
        curvature: float = 0.3,
        turns: int = 2,
        wind_std: float = 0.02,
    ) -> None:
        if len(frame_shape) != 3 or frame_shape[2] != 3:
            raise ValueError(f"frame_shape must be (height, width, 3), got {frame_shape}")
        if track_length <= 0 or half_width <= 0 or max_steps < 1 or frame_rate_hz <= 0:
            raise ValueError("track_length, half_width, frame_rate_hz > 0; max_steps >= 1")
        self.frame_shape = tuple(int(size) for size in frame_shape)
        self.track_length = float(track_length)
        self.half_width = float(half_width)
        self.max_steps = int(max_steps)
        self.frame_rate_hz = float(frame_rate_hz)
        self.dt = 1.0 / self.frame_rate_hz
        self.curvature = float(curvature)
        self.turns = int(turns)
        self.wind_std = float(wind_std)

        self._rng = np.random.default_rng()
        self._progress = 0.0
        self._lateral = 0.0
        self._speed = 0.0
        self._steps = 0
        self._done = True
        self._frame = self._render_state()

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape of the uint8 RGB frame returned by ``reset`` and ``step``."""
        return self.frame_shape

    @property
    def action_shape(self) -> tuple[int]:
        """Shape of the ``(steer, throttle, brake)`` action."""
        return (3,)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Frame, dict[str, Any]]:
        """Start a new episode at the start line.

        Args:
            seed: Seeds the wind. The same seed gives the same episode for the
                same action sequence.
            options: Unused; present for Gymnasium compatibility.

        Returns:
            The first frame and the info dict.
        """
        self._rng = np.random.default_rng(seed)
        self._progress = 0.0
        self._lateral = 0.0
        self._speed = 0.0
        self._steps = 0
        self._done = False
        self._frame = self._render_state()
        return self._frame, self._info(lap_complete=False, off_track=False)

    def step(self, action: object) -> tuple[Frame, float, bool, bool, dict[str, Any]]:
        """Advance one frame.

        Args:
            action: ``(steer, throttle, brake)`` array-like, or an object with a
                ``to_array()`` method returning one. Out-of-range values raise;
                nothing is clipped silently.

        Returns:
            ``(frame, reward, terminated, truncated, info)`` in Gymnasium order.
        """
        if self._done:
            raise RuntimeError("call reset() before step(); the episode has ended")
        steer, throttle, brake = self._validate_action(action)

        accel = throttle * self.max_accel - brake * self.max_brake - self.drag * self._speed
        self._speed = float(np.clip(self._speed + accel * self.dt, 0.0, self.max_speed))
        bend = self.curvature * np.sin(
            2.0 * np.pi * self.turns * self._progress / self.track_length
        )
        wind = float(self._rng.normal(0.0, self.wind_std)) if self.wind_std > 0 else 0.0
        self._lateral = float(
            self._lateral + (steer * self.steer_gain - bend) * self._speed * self.dt + wind
        )
        progress_delta = self._speed * self.dt
        self._progress += progress_delta
        self._steps += 1

        lap_complete = bool(self._progress >= self.track_length)
        off_track = bool(abs(self._lateral) > self.half_width)
        reward = progress_delta - self.lateral_penalty * abs(self._lateral) * self.dt
        if lap_complete:
            reward += self.lap_bonus
        elif off_track:
            reward += self.off_track_penalty
        terminated = lap_complete or off_track
        truncated = not terminated and self._steps >= self.max_steps
        self._done = terminated or truncated
        self._frame = self._render_state()
        return (
            self._frame,
            float(reward),
            terminated,
            truncated,
            self._info(lap_complete=lap_complete, off_track=off_track),
        )

    def render(self) -> Frame:
        """Return the current frame (identical to the last observation)."""
        return self._frame.copy()

    def close(self) -> None:
        """Nothing to release; present for Gymnasium compatibility."""

    def _validate_action(self, action: object) -> tuple[float, float, float]:
        raw = action.to_array() if hasattr(action, "to_array") else action
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (3,):
            raise ValueError(f"action must have shape (3,), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"action must be finite, got {values.tolist()}")
        for name, value, (low, high) in zip(_ACTION_NAMES, values, _ACTION_BOUNDS, strict=True):
            if not low <= value <= high:
                raise ValueError(
                    f"{name}={float(value)!r} is outside [{low}, {high}]; "
                    "clip explicitly before stepping."
                )
        return float(values[0]), float(values[1]), float(values[2])

    def _info(self, *, lap_complete: bool, off_track: bool) -> dict[str, Any]:
        return {
            "progress": self._progress,
            "lateral": self._lateral,
            "speed": self._speed,
            "lap_complete": lap_complete,
            "off_track": off_track,
            "lap_time": self._steps * self.dt if lap_complete else None,
        }

    def _render_state(self) -> Frame:
        height, width, _ = self.frame_shape
        frame = np.full((height, width), 40, dtype=np.uint8)
        # The road shifts opposite to the car's lateral offset, like a camera would see.
        shift = -self._lateral / self.half_width * (width / 2) * 0.8
        centre = width / 2 + shift
        band_half = max(2, width // 6)
        left = int(np.clip(round(centre - band_half), 0, width))
        right = int(np.clip(round(centre + band_half), 0, width))
        frame[:, left:right] = 160
        # Stripes that scroll with progress so consecutive frames carry motion.
        phase = int(self._progress * 5) % 16
        rows = np.arange(height)
        stripe_rows = ((rows + phase) % 16) < 2
        frame[stripe_rows, left:right] = 220
        return np.repeat(frame[:, :, None], 3, axis=2)
