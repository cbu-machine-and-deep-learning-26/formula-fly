"""A stand-in for Assetto Corsa: the same car methods, no AC, no Windows.

:class:`FakeAc` has the methods :class:`~fly_driver.bridge.assetto.AssettoCorsaCar` has
(``reset``, ``apply``, ``step``, ``telemetry``, ``close``), so the env server, the remote
protocol and the training env are all exercised without the game. It copies the parts of
AC's behaviour the training env depends on:

- **Real time.** With ``realtime=True`` ticks happen on the wall clock whether or not
  anyone is stepping. ``step`` waits for the next tick, or returns the newest tick at once
  if the caller is late, and the ticks it skipped show as a gap in ``Telemetry.tick``.
  The car drives on with the last control during the gap, as AC's does.
- **Resets.** ``reset`` puts the car back on the road where it is, at rest, and restarts
  the tick counter, like CSP's ``ac.ext_resetCar()`` behind upstream's reset. The very
  first reset starts at ``start_m``.

The track is a straight line folded into a loop: a lap of ``track_length_m`` metres, a
sinusoidal drift that steering must cancel, and a road ``half_width_m`` either side of the
centre. Runs on AC's Python 3.9.
"""

from __future__ import annotations

import math
import time

import numpy as np

from fly_driver.bridge.wire import Telemetry
from fly_driver.interface import FRAME_RATE_HZ, FRAME_SHAPE, ControlVector, Frame

__all__ = ["FakeAc"]

#: Half the car's width, for counting tyres over the edge.
_HALF_CAR_M = 1.0


class FakeAc:
    """A one-dimensional car on a loop, ticking at ``frame_rate_hz``.

    Args:
        track_length_m: Lap length.
        half_width_m: Road half-width; the centre past it is off the track.
        start_m: Where the first episode starts.
        frame_shape: Frame size.
        frame_rate_hz: Tick rate.
        realtime: Tick on the wall clock. Off, every ``step`` is exactly one tick.
        reset_s: Wall time a reset takes, to mimic upstream's one-second reset.
        max_accel: Full-throttle acceleration, m/s².
        drift: Lateral drift per metre driven, at the bend's peak.
    """

    def __init__(
        self,
        track_length_m: float = 600.0,
        half_width_m: float = 6.0,
        start_m: float = 0.0,
        frame_shape: tuple[int, int, int] = FRAME_SHAPE,
        frame_rate_hz: float = FRAME_RATE_HZ,
        realtime: bool = True,
        reset_s: float = 0.0,
        max_accel: float = 10.0,
        drift: float = 0.05,
    ) -> None:
        self.track_length_m = float(track_length_m)
        self.half_width_m = float(half_width_m)
        self.frame_shape = tuple(int(size) for size in frame_shape)
        self.dt = 1.0 / float(frame_rate_hz)
        self.realtime = bool(realtime)
        self.reset_s = float(reset_s)
        self.max_accel = float(max_accel)
        self.drift = float(drift)
        self.lap_dist_m = float(start_m) % self.track_length_m
        self.lateral_m = 0.0
        self.speed_mps = 0.0
        self.control = ControlVector.neutral()
        self.resets = 0
        self._tick = 0
        self._time_s = 0.0
        self._started = time.perf_counter()

    def reset(self) -> Frame:
        """Back on the road where the car is, at rest; the tick counter restarts."""
        if self.reset_s:
            time.sleep(self.reset_s)
        self.resets += 1
        self.lateral_m = 0.0
        self.speed_mps = 0.0
        self.control = ControlVector.neutral()
        self._tick = 0
        self._started = time.perf_counter()
        return self._frame()

    def apply(self, control: ControlVector) -> None:
        """Hold ``control`` until the next one."""
        self.control = control

    def step(self) -> tuple[Frame, bool]:
        """Advance to the next tick, or to the newest one if the caller is late."""
        due = self._tick + 1
        if self.realtime:
            wait = self._started + due * self.dt - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                due = int((time.perf_counter() - self._started) / self.dt)
        while self._tick < due:
            self._integrate()
        return self._frame(), False

    def telemetry(self) -> Telemetry:
        """The current tick's state."""
        overhang = abs(self.lateral_m) - self.half_width_m
        if overhang > _HALF_CAR_M:
            tyres_out = 4
        elif overhang > -_HALF_CAR_M:
            tyres_out = 2
        else:
            tyres_out = 0
        return Telemetry(
            tick=self._tick,
            time_s=self._time_s,
            lap_dist_m=self.lap_dist_m,
            track_length_m=self.track_length_m,
            lateral_m=self.lateral_m,
            speed_mps=self.speed_mps,
            tyres_out=tyres_out,
            centre_on_track=overhang <= 0.0,
        )

    def close(self) -> None:
        """Nothing to release."""

    def _integrate(self) -> None:
        control = self.control
        accel = self.max_accel * (control.throttle - 1.5 * control.brake) - 0.01 * self.speed_mps
        self.speed_mps = max(0.0, self.speed_mps + accel * self.dt)
        distance = self.speed_mps * self.dt
        bend = math.sin(2.0 * math.pi * 2.0 * self.lap_dist_m / self.track_length_m)
        self.lateral_m += (self.drift * bend + 0.2 * control.steer) * distance
        self.lap_dist_m = (self.lap_dist_m + distance) % self.track_length_m
        self._tick += 1
        self._time_s += self.dt

    def _frame(self) -> Frame:
        """A grey road on a dark field, shifted by the car's lateral offset."""
        height, width, channels = self.frame_shape
        columns = np.arange(width) - width / 2.0 + self.lateral_m * width / (4 * self.half_width_m)
        road = np.abs(columns) < width / 4.0
        frame = np.full((height, width, channels), 40, dtype=np.uint8)
        frame[height // 2 :, road] = 160
        return frame
