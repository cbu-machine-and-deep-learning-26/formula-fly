"""Where to brake, lift and push, along the racing line (GH-27).

Round 6 crashed most often at the braking zones after straights, so v7 paints the answer on
the road: a dark stripe along :func:`~fly_driver.envs.racing_line.racing_line` where the car
must brake. The profile also marks where it carries cornering speed and where it pushes, but
the scene paints only the brake band (:class:`~fly_driver.envs.scene.SceneConfig`): painting
all three put a stripe in nearly every frame and a warm-started policy stopped launching.

The bands come from a point-mass speed profile on that line:

- **Grip.** ``a_lat(v) = mu * (g + downforce(v) / m)``, with downforce from the SF70H's
  aero (:mod:`fly_driver.envs.aero`) and ``mu`` fitted so that ``a_lat`` is 4.4 g at
  250 km/h, the measured envelope in ``docs/training.md``. That gives 2.3 g at 108 km/h.
- **Corner limit.** ``v_lim = sqrt(a_lat(v) / curvature)``, solved for ``v`` because grip
  grows with ``v^2``, and capped at ``TOP_SPEED_MPS``.
- **Acceleration.** A forward pass with ``min(rear share * a_lat(v), P / (m v)) - drag / m``:
  traction-limited at low speed, 758 kW (:mod:`fly_driver.envs.powertrain`) above it.
- **Braking.** A backward pass from each corner with deceleration ``0.84 * a_lat(v)``: the
  measured 3.7 g average from 300 to 100 km/h over 4.4 g, the same tyres and aero.

A point is **brake** where the backward pass lowered the profile: a corner ahead forces it
down. It is **push** where the profile is rising or at top speed. It is **carry**
everywhere else: in the corner, at the limit, waiting for the apex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fly_driver.envs.aero import SF70H_AERO, downforce_n, drag_n
from fly_driver.envs.car import CarConfig
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.powertrain import SF70H_POWERTRAIN
from fly_driver.envs.racing_line import racing_line
from fly_driver.envs.segments import signed_curvature

__all__ = ["BRAKE", "CARRY", "PUSH", "SpeedProfile", "speed_profile"]

GRAVITY = 9.81
#: docs/training.md's measured envelope.
LATERAL_G_AT_250_KMH = 4.4
BRAKING_G = 3.7
#: The profile's cap on straights, about the SF70H's top speed at Silverstone.
TOP_SPEED_MPS = 92.0

#: Band codes, darkest first.
BRAKE, CARRY, PUSH = 0, 1, 2


@dataclass(frozen=True)
class SpeedProfile:
    """Target speed and pedal band at each point of the racing line.

    Args:
        points: ``(N, 2)`` racing-line points, world metres.
        arclength_m: Each point's arclength along the centreline it was built from.
        limit_mps: The cornering limit at each point, capped at ``TOP_SPEED_MPS``.
        speed_mps: The profile: the limit, lowered where braking for a corner ahead.
        bands: ``BRAKE``, ``CARRY`` or ``PUSH`` at each point.
    """

    points: npt.NDArray[np.float64]
    arclength_m: npt.NDArray[np.float64]
    limit_mps: npt.NDArray[np.float64]
    speed_mps: npt.NDArray[np.float64]
    bands: npt.NDArray[np.int64]


def speed_profile(centerline: Centerline, spacing_m: float = 4.0) -> SpeedProfile:
    """The braking profile along ``centerline``'s racing line."""
    points = racing_line(centerline, spacing_m=spacing_m)
    arclength = np.asarray(centerline.resample(spacing_m).arclength, dtype=float)[: len(points)]
    steps = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    curvature = np.abs(signed_curvature(points))

    mass = CarConfig().mass_kg
    # Downforce is 0.5 * rho * cla * v^2, so per unit mass it is aero * v^2.
    aero = float(downforce_n(1.0, SF70H_AERO)) / mass
    v_ref = 250.0 / 3.6
    mu = LATERAL_G_AT_250_KMH * GRAVITY / (GRAVITY + aero * v_ref**2)

    def lateral(speed: npt.NDArray[np.float64] | float) -> npt.NDArray[np.float64] | float:
        return mu * (GRAVITY + aero * np.square(speed))

    # v^2 * k = mu * (g + aero * v^2)  =>  v^2 = mu * g / (k - mu * aero).
    room = curvature - mu * aero
    with np.errstate(divide="ignore"):
        limit = np.where(room > 0, np.sqrt(mu * GRAVITY / np.maximum(room, 1e-12)), TOP_SPEED_MPS)
    limit = np.minimum(limit, TOP_SPEED_MPS)

    car = CarConfig()
    rear_share = 1.0 - car.front_weight_fraction
    power_w = SF70H_POWERTRAIN.peak_power_w
    speed = limit.copy()
    count = len(points)
    # Twice round, from rest at the line, so the second lap starts at its real speed.
    for step in range(1, 2 * count):
        i, behind = step % count, (step - 1) % count
        before = max(speed[behind], 1.0)
        drive = min(rear_share * lateral(before), power_w / (mass * before))
        accel = max(drive - float(drag_n(before, SF70H_AERO)) / mass, 0.0)
        speed[i] = min(speed[i], np.sqrt(before**2 + 2.0 * accel * steps[behind]))

    accelerating = speed.copy()
    braking = BRAKING_G / LATERAL_G_AT_250_KMH
    # Twice round the closed loop, so braking for the first corner reaches back past the line.
    for step in range(2 * count - 1, -1, -1):
        i, ahead = step % count, (step + 1) % count
        reachable = np.sqrt(speed[ahead] ** 2 + 2.0 * braking * lateral(speed[ahead]) * steps[i])
        speed[i] = min(speed[i], reachable)

    rising = np.roll(speed, -1) > speed + 0.05
    bands = np.full(count, CARRY, dtype=np.int64)
    bands[rising | (speed >= TOP_SPEED_MPS - 0.05)] = PUSH
    bands[speed < accelerating - 0.5] = BRAKE
    return SpeedProfile(points, arclength, limit, speed, bands)
