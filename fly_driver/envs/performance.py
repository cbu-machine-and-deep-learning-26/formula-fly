"""Measure the car's performance envelope (GH-16).

Lives in the package rather than in ``scripts/`` because three callers need it:
``scripts/validate_car.py`` prints it for a human, ``tests/test_car_performance.py``
asserts it in CI, and the evaluation harness (GH-18) will want the same primitives.

Every measurement drives through :class:`~fly_driver.envs.car.CarDynamics`, which is the
only path that applies aerodynamics. Stepping MuJoCo directly would silently measure a car
with no downforce, which is the difference between 6 g and 1.6 g.

A note on peak accelerations, because getting this wrong wasted real time. A single-step
derivative of a contact solver's output is noise, not physics: it reported 36 g under
braking and 10 g on a launch that is traction-limited to about 1.1 g. Every peak here is
averaged over :data:`PEAK_WINDOW_S`. Lateral acceleration is read from body-frame linear
acceleration in a settled turn, never ``yaw_rate * speed`` -- that shortcut reported 6.94 g
on a car whose theoretical ceiling was 1.6 g, because a spinning car's yaw rate stops
meaning anything.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from fly_driver.envs.car import (
    CarConfig,
    CarDynamics,
    car_actuators_xml,
    car_assets_xml,
    car_body_xml,
)
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.scene import SceneConfig, build_scene_xml
from fly_driver.interface import ControlVector

__all__ = [
    "PEAK_WINDOW_S",
    "PerformanceBed",
    "measure_acceleration",
    "measure_braking",
    "measure_lateral",
    "measure_top_speed",
]

G = 9.81

#: Seconds over which peak accelerations are averaged. See the module docstring.
PEAK_WINDOW_S = 0.05

_FULL_THROTTLE = ControlVector(steer=0.0, throttle=1.0, brake=0.0)
_FULL_BRAKE = ControlVector(steer=0.0, throttle=0.0, brake=1.0)


@dataclass
class PerformanceBed:
    """A compiled car on a large flat oval, with nothing track-specific about it."""

    model: mujoco.MjModel
    data: mujoco.MjData
    dynamics: CarDynamics
    car: CarConfig

    @classmethod
    def build(cls, car: CarConfig | None = None) -> PerformanceBed:
        """Kilometres of flat road so no measurement runs out of track."""
        car = car or CarConfig()
        points = [(0.0, 0.0), (9000.0, 0.0), (9000.0, 4000.0), (0.0, 4000.0)]
        centerline = Centerline(
            points=points, half_width_right=[60.0] * 4, half_width_left=[60.0] * 4
        )
        position, yaw = centerline.pose_at(0.0)
        model = mujoco.MjModel.from_xml_string(
            build_scene_xml(
                centerline,
                SceneConfig(mesh_spacing_m=500.0, kerb_width_m=0.0),
                extra_assets=car_assets_xml(car),
                extra_bodies=car_body_xml(position, yaw, car),
                extra_actuators=car_actuators_xml(car),
            )
        )
        data = mujoco.MjData(model)
        return cls(model=model, data=data, dynamics=CarDynamics(model, car), car=car)

    @property
    def dt(self) -> float:
        return float(self.model.opt.timestep)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.dynamics.reset()

    def speed(self) -> float:
        return self.dynamics.speed_mps(self.data)

    def position(self) -> np.ndarray:
        return self.data.xpos[self.dynamics._body][:2].copy()

    def accelerate_to(self, target_mps: float, limit_s: float = 60.0) -> bool:
        """Full throttle until ``target_mps``. False if the car never gets there."""
        for _ in range(int(limit_s / self.dt)):
            self.dynamics.step(_FULL_THROTTLE, self.data, 1)
            if self.speed() >= target_mps:
                return True
        return False


def _windowed_g(history: deque[float], dt: float) -> float:
    """Acceleration across the whole window, in g, positive for speeding up."""
    if len(history) < 2:
        return 0.0
    return (history[-1] - history[0]) / ((len(history) - 1) * dt) / G


def measure_acceleration(bed: PerformanceBed) -> dict[str, float]:
    """Times to 100, 200 and 300 km/h from rest, plus peak launch g."""
    bed.reset()
    marks: dict[float, float | None] = {100.0: None, 200.0: None, 300.0: None}
    launch_peak_g, time = 0.0, 0.0
    history: deque[float] = deque(maxlen=max(2, int(PEAK_WINDOW_S / bed.dt)))

    while time < 60.0 and any(value is None for value in marks.values()):
        bed.dynamics.step(_FULL_THROTTLE, bed.data, 1)
        time += bed.dt
        speed = bed.speed()
        history.append(speed)
        # Start once the wheels have taken up load; the first moments are solver settling.
        if 0.1 < time <= 1.0:
            launch_peak_g = max(launch_peak_g, _windowed_g(history, bed.dt))
        for kmh in marks:
            if marks[kmh] is None and speed * 3.6 >= kmh:
                marks[kmh] = time

    return {
        "0-100 km/h (s)": marks[100.0] if marks[100.0] else float("nan"),
        "0-200 km/h (s)": marks[200.0] if marks[200.0] else float("nan"),
        "0-300 km/h (s)": marks[300.0] if marks[300.0] else float("nan"),
        "launch peak (g)": launch_peak_g,
    }


def measure_top_speed(bed: PerformanceBed) -> dict[str, float]:
    """Full throttle until speed stops changing.

    This only terminates because drag exists. Without it the car accelerated until the
    integrator produced NaN at 667 km/h, so a finite answer here is itself the check that
    aerodynamics is actually being applied.
    """
    bed.reset()
    previous, settled, time = 0.0, 0, 0.0
    while time < 180.0:
        bed.dynamics.step(_FULL_THROTTLE, bed.data, 1)
        time += bed.dt
        speed = bed.speed()
        settled = settled + 1 if abs(speed - previous) < 1e-5 else 0
        if settled > 2000:
            break
        previous = speed
    return {"top speed (km/h)": bed.speed() * 3.6, "top gear": float(bed.dynamics.gear + 1)}


def measure_braking(bed: PerformanceBed, from_kmh: float, to_kmh: float) -> dict[str, float]:
    """Distance and deceleration braking between two speeds."""
    label = f"{from_kmh:.0f}->{to_kmh:.0f} km/h"
    bed.reset()
    if not bed.accelerate_to(from_kmh / 3.6):
        return {f"brake {label} (m)": float("nan")}

    start, entry = bed.position(), bed.speed()
    peak_g, time = 0.0, 0.0
    history: deque[float] = deque(maxlen=max(2, int(PEAK_WINDOW_S / bed.dt)))
    # A car never reaches exactly zero, so stop just above it. Waiting for 0 ran the full
    # timeout and divided the average by 30 s instead of the real stopping time.
    stop_at = max(to_kmh / 3.6, 0.5)

    while bed.speed() > stop_at and time < 30.0:
        bed.dynamics.step(_FULL_BRAKE, bed.data, 1)
        time += bed.dt
        history.append(bed.speed())
        if time > 0.1:
            peak_g = max(peak_g, -_windowed_g(history, bed.dt))

    return {
        f"brake {label} (m)": float(np.linalg.norm(bed.position() - start)),
        f"brake {label} avg (g)": (entry - bed.speed()) / time / G if time else float("nan"),
        f"brake {label} peak (g)": peak_g,
    }


def measure_lateral(bed: PerformanceBed, speed_kmh: float) -> dict[str, float]:
    """Peak *settled* lateral acceleration through a progressively tightened turn.

    Each lock is held for 0.25 s and only the last 0.1 s is averaged. A single sample
    taken the instant the lock changed was measuring the kingpin's transient, not grip:
    with the undamped steering it read 4.37 g against a settled 2.8 g, and fixing the
    steering damper "lost" 0.7 g that was never there.
    """
    key = f"lateral at {speed_kmh:.0f} km/h (g)"
    bed.reset()
    if not bed.accelerate_to(speed_kmh / 3.6):
        return {key: float("nan")}

    best = 0.0
    hold_steps = int(0.25 / bed.dt)
    settle_steps = int(0.1 / bed.dt)
    for steer in np.linspace(0.0, -0.6, 12):
        control = ControlVector(steer=float(steer), throttle=0.55, brake=0.0)
        samples = []
        for step in range(hold_steps):
            bed.dynamics.step(control, bed.data, 1)
            if step >= hold_steps - settle_steps:
                accel = np.zeros(6)
                mujoco.mj_objectAcceleration(
                    bed.model, bed.data, mujoco.mjtObj.mjOBJ_BODY, bed.dynamics._body, accel, 1
                )
                samples.append(abs(float(accel[4])) / G)
        best = max(best, float(np.mean(samples)))
    return {key: best}
