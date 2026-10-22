"""Assetto Corsa as a training environment (GH-29), with :class:`PracticeTrack`'s interface.

``reset(seed) -> (frame, info)`` and ``step(action) -> (frame, reward, terminated,
truncated, info)``, the same ``info`` keys as far as AC supplies them, and the same
:class:`~fly_driver.envs.practice_track.RewardFunction`, so the trainer, the evaluation
harness and any reward written for the practice track run here unchanged.

The car is anything with :class:`~fly_driver.bridge.env_link.AcCar`'s methods. In training
it is a :class:`~fly_driver.bridge.env_link.RemoteCar` talking to ``scripts/ac_env_server.py``
on the AC box (``127.0.0.1`` when the trainer runs there too); in tests it is
:class:`~fly_driver.bridge.fake_ac.FakeAc`. See ``docs/assetto-corsa-training.md``.

What differs from the practice track, because AC differs:

- **Starts.** AC has no call that places the car at a chosen point. A reset puts the car
  back on the road where it stopped, at rest (CSP's ``ac.ext_resetCar()``), so the next
  episode starts where the last one ended. ``seed`` is accepted and does nothing; there is
  no ``random_start`` and no fixed evaluation start.
- **Track position** is AC's own: progress along AC's spline (``NormalizedSplinePosition``
  times its length, ``0`` at the line), distance from AC's racing line for ``lateral_m``,
  and tyres off the surface for the track limit. ``off_track_fraction`` is
  ``tyres_out / 4``; ``on_track`` is the car centre inside upstream's occupancy grid.
- **Real time.** A step blocks until the next 50 Hz tick. A trainer that comes back late
  misses ticks while the car drives on with its last control. Each step reports
  ``missed_ticks``, :attr:`AssettoCorsaTrack.missed_ticks` keeps the total, and a step
  that missed more than ``max_missed_ticks`` truncates the episode, so the policy is never
  credited with driving it did not do. PPO updates are such gaps, so every rollout
  boundary ends an episode on AC.
- **No off-track time penalty.** A lap's time is the stopwatch time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

from fly_driver.bridge.env_link import AcCar, RemoteCar
from fly_driver.bridge.wire import DEFAULT_ENV_PORT, Telemetry
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.lap import LapTimer
from fly_driver.envs.practice_track import ProgressReward, RewardFunction
from fly_driver.interface import FRAME_RATE_HZ, FRAME_SHAPE, ControlVector, Frame, validate_frame

__all__ = ["DEFAULT_CAR", "DEFAULT_TRACK", "AssettoCorsaTrack"]

logger = logging.getLogger(__name__)

DEFAULT_TRACK = "ks_silverstone-gp"
DEFAULT_CAR = "dallara_f317"


@dataclass(frozen=True)
class _Loop:
    """AC's spline as :class:`~fly_driver.envs.lap.LapTimer` needs it: a length and a
    seam-safe progress delta."""

    length: float

    def progress_delta(self, previous_s: float, current_s: float) -> float:
        """Signed metres forward; a jump of more than half a lap is a seam crossing."""
        delta = current_s - previous_s
        if delta > self.length / 2.0:
            delta -= self.length
        elif delta < -self.length / 2.0:
            delta += self.length
        return delta


class AssettoCorsaTrack:
    """One AC car as a Gymnasium-style env.

    Args:
        car: The car, usually :meth:`connect`'s :class:`RemoteCar`.
        max_steps: Truncate after this many steps; ``None`` never.
        frame_shape: Every frame is checked against it and never resized.
        frame_rate_hz: AC's ego tick rate.
        track_limit: Fraction of tyres off the surface that ends the episode when
            ``terminate_off_track`` is on. ``0.5`` ends it at three tyres out, upstream's
            rule; ``0`` turns the rule off.
        terminate_off_track: End the episode on leaving the track.
        stall_s: End the episode once the car has been slower than ``stall_speed_mps`` for
            this long. ``None`` turns it off. A reset leaves the car at rest, so this is
            also how long a new episode has to get moving.
        stall_speed_mps: The stall threshold. Upstream uses 1 m/s.
        max_missed_ticks: Truncate the episode when one step missed more ticks than this;
            ``None`` only counts them.
        reward: Scores each step. Defaults to
            :class:`~fly_driver.envs.practice_track.ProgressReward`.
    """

    def __init__(
        self,
        car: AcCar,
        *,
        max_steps: int | None = None,
        frame_shape: tuple[int, int, int] = FRAME_SHAPE,
        frame_rate_hz: float = FRAME_RATE_HZ,
        track_limit: float = 0.5,
        terminate_off_track: bool = True,
        stall_s: float | None = None,
        stall_speed_mps: float = 1.0,
        max_missed_ticks: int | None = 5,
        reward: RewardFunction | None = None,
    ) -> None:
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"max_steps must be positive or None, got {max_steps}")
        if not 0.0 <= track_limit <= 1.0:
            raise ValueError(f"track_limit must be in [0, 1], got {track_limit}")
        if stall_s is not None and stall_s <= 0:
            raise ValueError(f"stall_s must be positive or None, got {stall_s}")
        if max_missed_ticks is not None and max_missed_ticks < 0:
            raise ValueError(f"max_missed_ticks must be >= 0 or None, got {max_missed_ticks}")
        self.car = car
        self.reward = reward or ProgressReward()
        self.frame_shape = tuple(int(size) for size in frame_shape)
        self.frame_rate_hz = float(frame_rate_hz)
        self._max_steps = max_steps
        self._track_limit = float(track_limit)
        self._terminate_off_track = bool(terminate_off_track)
        self._stall_s = stall_s
        self._stall_speed_mps = float(stall_speed_mps)
        self._max_missed_ticks = max_missed_ticks
        #: Ticks missed since construction, over every episode.
        self.missed_ticks = 0
        self._done = True
        self._frame: Frame | None = None

    @classmethod
    def connect(
        cls,
        host: str = "127.0.0.1",
        port: int | None = None,
        *,
        track: str = DEFAULT_TRACK,
        car: str = DEFAULT_CAR,
        **kwargs: Any,
    ) -> AssettoCorsaTrack:
        """Connect to ``scripts/ac_env_server.py`` and check it serves what was asked.

        Raises:
            ValueError: If the server's track, car, frame shape or rate differ.
        """
        port = DEFAULT_ENV_PORT if port is None else port
        remote = RemoteCar(host, port)
        env = cls(remote, **kwargs)
        expected = {
            "track": track,
            "car": car,
            "frame_shape": list(env.frame_shape),
            "frame_rate_hz": env.frame_rate_hz,
        }
        served = {key: remote.hello.get(key) for key in expected}
        if served != expected:
            remote.close()
            raise ValueError(f"env server at {host}:{port} serves {served}, expected {expected}")
        return env

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape of the uint8 RGB frame."""
        return self.frame_shape

    @property
    def action_shape(self) -> tuple[int]:
        """``(steer, throttle, brake)``."""
        return (3,)

    @property
    def dt(self) -> float:
        """Seconds per step at the nominal tick rate."""
        return 1.0 / self.frame_rate_hz

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Frame, dict[str, Any]]:
        """Reset the car on the road where it stopped; return the first frame and info.

        Args:
            seed: Accepted for the harness and ignored: AC cannot choose the start.
            options: Unused.
        """
        del seed, options
        self._frame = validate_frame(self.car.reset(), self.frame_shape)
        telemetry = self.car.telemetry()
        self._loop = _Loop(telemetry.track_length_m)
        self._lap = LapTimer(cast(Centerline, self._loop))
        self._lap.reset(telemetry.lap_dist_m, telemetry.time_s)
        self._previous = telemetry
        self._steps = 0
        self._episode_missed = 0
        self._slow_s = 0.0
        self._done = False
        info = self._info(telemetry, progress_m=0.0, missed=0, lap_time=None)
        info.update(stalled=False, interrupted=False)
        return self._frame, info

    def step(self, action: object) -> tuple[Frame, float, bool, bool, dict[str, Any]]:
        """Apply ``action`` now and return the next tick.

        Args:
            action: A :class:`~fly_driver.interface.ControlVector` or a length-3 array;
                out-of-range values raise, as on the practice track.
        """
        if self._done:
            raise RuntimeError("call reset() before step(); the episode has ended")
        self.car.apply(ControlVector.from_any(action))
        frame, interrupted = self.car.step()
        self._frame = validate_frame(frame, self.frame_shape)
        telemetry = self.car.telemetry()
        previous, self._previous = self._previous, telemetry
        self._steps += 1

        missed = max(0, telemetry.tick - previous.tick - 1)
        if missed:
            self.missed_ticks += missed
            self._episode_missed += missed
            logger.info("missed %d ticks (%.0f ms) behind AC", missed, missed * self.dt * 1e3)
        progress_m = self._loop.progress_delta(previous.lap_dist_m, telemetry.lap_dist_m)
        lap_time = self._lap.update(
            telemetry.lap_dist_m, telemetry.time_s, on_track=telemetry.tyres_out == 0
        )
        if telemetry.speed_mps < self._stall_speed_mps:
            self._slow_s += telemetry.time_s - previous.time_s
        else:
            self._slow_s = 0.0
        stalled = self._stall_s is not None and self._slow_s >= self._stall_s

        info = self._info(telemetry, progress_m=progress_m, missed=missed, lap_time=lap_time)
        info.update(stalled=stalled, interrupted=interrupted)
        reward, terms = self.reward(info, self.dt)
        info["reward_terms"] = terms

        off_limit = (
            self._terminate_off_track
            and self._track_limit > 0.0
            and info["off_track_fraction"] > self._track_limit
        )
        terminated = off_limit or stalled
        late = self._max_missed_ticks is not None and missed > self._max_missed_ticks
        capped = self._max_steps is not None and self._steps >= self._max_steps
        truncated = not terminated and (capped or late or interrupted)
        self._done = terminated or truncated
        if self._done and self._episode_missed:
            logger.warning(
                "episode ended after %d steps with %d missed ticks",
                self._steps,
                self._episode_missed,
            )
        return self._frame, float(reward), terminated, truncated, info

    def render(self) -> Frame:
        """The last frame, copied."""
        if self._frame is None:
            raise RuntimeError("call reset() before render()")
        return self._frame.copy()

    def close(self) -> None:
        """Release the car (for a :class:`RemoteCar`, hang up)."""
        self.car.close()

    def __enter__(self) -> AssettoCorsaTrack:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _info(
        self, telemetry: Telemetry, *, progress_m: float, missed: int, lap_time: float | None
    ) -> dict[str, Any]:
        """PracticeTrack's keys, as far as AC supplies them. ``lap_time`` is the completed
        lap's time and ``None`` otherwise, which is what the evaluation harness reads."""
        return {
            "lap_complete": lap_time is not None,
            "lap_time": lap_time,
            "lap_time_raw": lap_time,
            "off_track": telemetry.tyres_out > 0,
            "off_track_fraction": telemetry.tyres_out / 4.0,
            "on_track": telemetry.centre_on_track,
            "reward_terms": {},
            "progress_m": float(progress_m),
            "speed_mps": telemetry.speed_mps,
            "lateral_m": telemetry.lateral_m,
            "arclength_m": telemetry.lap_dist_m,
            "steps": self._steps,
            "sim_time": telemetry.time_s,
            "lap_elapsed_s": self._lap.current_lap_time(telemetry.time_s),
            "lap_fraction": self._lap.lap_fraction,
            "lap_count": self._lap.completed,
            "on_out_lap": not self._lap.timing,
            "tyres_out": telemetry.tyres_out,
            "missed_ticks": missed,
        }
