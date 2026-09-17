"""Engine, gearbox and brakes for the SF70H (GH-16).

Pure numpy, no MuJoCo, for the same reason as :mod:`fly_driver.envs.aero`: the arithmetic
is checkable by hand and testable without a compiled model.

Why this replaces a constant drive torque
-----------------------------------------

The car previously applied a fixed 900 N·m per rear wheel regardless of speed. That gives
a flat tractive force, which is wrong at both ends: far too little off the line, and
nothing stopping it at the top end (with drag also missing, the car simply accelerated to
667 km/h until the integrator broke).

A real car is **torque-limited at low speed and power-limited at high speed**. Wheel torque
is engine torque multiplied by the gear ratio, and since power is torque times speed, the
same 746 kW that gives enormous force in first gear gives very little in eighth. That
crossover is what makes acceleration feel the way it does, and it is what makes a top speed
exist at all.

An interesting consequence, worth stating because it looks like a bug otherwise: an F1
launch is **traction-limited, not power-limited**. With 45.5% of 728 kg on the front axle,
the rear axle carries ~3.9 kN plus weight transfer; at μ = 1.7 that is barely 8 kN of grip,
which on 728 kg is about 1.1 g. That is exactly the published 0–100 km/h in 2.6 s. The
widely quoted "F1 cars pull 2 g accelerating" only becomes true higher up the speed range,
once downforce has loaded the tyres. So the model is expected to spin its wheels off the
line, and the validation asserts launch g stays *below* 1.5.

Brakes
------

Previously MuJoCo ``damper`` actuators: force proportional to wheel speed. That is why
braking measured 1.37 g from 150 km/h but only 0.6 g from 64 km/h — the brakes faded
exactly when trying to stop. Real brakes are close to constant torque until lockup, so
:func:`brake_torque` returns a constant magnitude opposing rotation, clamped so it can
bring a wheel to rest but never spin it backwards.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["PowertrainConfig", "SF70H_POWERTRAIN", "brake_torque", "drive_torque", "select_gear"]

#: Ratios for the SF70H's 8-speed sequential box, plus the final drive. Ferrari do not
#: publish theirs, so these are derived from the two speeds that are known: first gear
#: should reach the limiter around 120 km/h and eighth at the ~340 km/h top speed. That
#: fixes the overall spread at ~2.8, and the eight ratios are spaced geometrically across
#: it so each upshift drops the same fraction of engine speed.
_DEFAULT_GEAR_RATIOS = (2.52, 2.17, 1.87, 1.62, 1.39, 1.20, 1.03, 0.89)
_DEFAULT_FINAL_DRIVE = 6.26


@dataclass(frozen=True)
class PowertrainConfig:
    """Engine, transmission and brake parameters.

    Args:
        peak_power_w: Combined ICE + electrical output at the crankshaft.
        peak_torque_nm: Maximum crankshaft torque. Together with ``peak_power_w`` this
            defines the corner of the torque curve: torque is flat up to the speed where
            the two meet, then falls as ``P / omega``.
        max_engine_rads: Rev limit in rad/s. 15000 rpm is the 2017 regulation ceiling.
        idle_engine_rads: Below this the engine is treated as off the throttle rather than
            producing negative torque, which keeps a stationary car from creeping.
        gear_ratios: Gearbox ratios, first to top.
        final_drive: Differential ratio, applied on top of the gear ratio.
        driveline_efficiency: Fraction of crank torque reaching the wheels.
        shift_up_fraction: Upshift when engine speed passes this fraction of the limiter.
        shift_down_fraction: Downshift below this fraction. The gap between the two is
            what stops the box hunting between gears at a steady speed.
        max_brake_torque_nm: Total braking torque across all four wheels at full pedal.
        brake_bias_front: Fraction of that torque on the front axle. Real cars run
            54-58% front; too far rearward and the car spins under braking.

    Raises:
        ValueError: On non-positive values, an empty or non-descending gear set, or
            shift thresholds that would make the gearbox hunt.
    """

    peak_power_w: float
    peak_torque_nm: float
    max_engine_rads: float
    idle_engine_rads: float
    max_brake_torque_nm: float
    gear_ratios: tuple[float, ...] = _DEFAULT_GEAR_RATIOS
    final_drive: float = _DEFAULT_FINAL_DRIVE
    driveline_efficiency: float = 0.90
    shift_up_fraction: float = 0.97
    shift_down_fraction: float = 0.55
    brake_bias_front: float = 0.57

    def __post_init__(self) -> None:
        positives = {
            "peak_power_w": self.peak_power_w,
            "peak_torque_nm": self.peak_torque_nm,
            "max_engine_rads": self.max_engine_rads,
            "idle_engine_rads": self.idle_engine_rads,
            "max_brake_torque_nm": self.max_brake_torque_nm,
            "final_drive": self.final_drive,
        }
        for name, value in positives.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not self.gear_ratios:
            raise ValueError("gear_ratios must not be empty")
        if any(a <= b for a, b in zip(self.gear_ratios, self.gear_ratios[1:], strict=False)):
            raise ValueError(f"gear_ratios must descend from first to top, got {self.gear_ratios}")
        if not 0.0 < self.driveline_efficiency <= 1.0:
            raise ValueError(
                f"driveline_efficiency must be in (0, 1], got {self.driveline_efficiency}"
            )
        if not 0.0 < self.shift_down_fraction < self.shift_up_fraction <= 1.0:
            raise ValueError(
                f"need 0 < shift_down_fraction < shift_up_fraction <= 1, got "
                f"{self.shift_down_fraction} and {self.shift_up_fraction}; overlapping "
                f"thresholds make the gearbox hunt"
            )
        if not 0.0 < self.brake_bias_front < 1.0:
            raise ValueError(f"brake_bias_front must be in (0, 1), got {self.brake_bias_front}")
        if self.idle_engine_rads >= self.max_engine_rads:
            raise ValueError("idle_engine_rads must be below max_engine_rads")

    @property
    def num_gears(self) -> int:
        return len(self.gear_ratios)

    def total_ratio(self, gear: int) -> float:
        """Combined gear and final-drive ratio for a 0-indexed gear."""
        if not 0 <= gear < self.num_gears:
            raise IndexError(f"gear {gear} outside 0..{self.num_gears - 1}")
        return self.gear_ratios[gear] * self.final_drive


#: Ferrari SF70H. 746 kW is the published ~1000 hp. Peak torque is not published, so it is
#: set to put the power/torque crossover near 10,000 rpm, where a turbo hybrid's peak sits.
SF70H_POWERTRAIN = PowertrainConfig(
    peak_power_w=746_000.0,
    peak_torque_nm=700.0,
    max_engine_rads=15_000.0 * 2.0 * np.pi / 60.0,
    idle_engine_rads=4_000.0 * 2.0 * np.pi / 60.0,
    # Sized so the tyres, not the discs, are the limit: 5.5 g on 728 kg needs ~39 kN,
    # which at a 0.335 m radius is ~13 kNm. The headroom above that is what lets a driver
    # lock a wheel, which is a real thing a policy can do wrong.
    max_brake_torque_nm=16_000.0,
)


def engine_torque(engine_rads: float, config: PowertrainConfig) -> float:
    """Crankshaft torque at a given engine speed, in N·m, at full throttle.

    Flat at ``peak_torque_nm`` until the power limit bites, then ``P / omega``. This is a
    deliberately smooth idealisation of a turbo hybrid's real curve: the shape that matters
    for lap time is the power ceiling, not the ripples.
    """
    speed = float(np.clip(engine_rads, config.idle_engine_rads, config.max_engine_rads))
    power_limited = config.peak_power_w / speed
    return float(min(config.peak_torque_nm, power_limited))


def engine_speed_rads(wheel_rads: float, gear: int, config: PowertrainConfig) -> float:
    """Engine speed for a given wheel speed and gear."""
    return abs(float(wheel_rads)) * config.total_ratio(gear)


def select_gear(wheel_rads: float, current_gear: int, config: PowertrainConfig) -> int:
    """Choose a gear for the current wheel speed.

    Upshifts near the limiter and downshifts well below it. The gap between the two
    thresholds is hysteresis: without it the box would shift up, immediately find itself
    below the downshift point in the taller gear, and oscillate every step.

    Args:
        wheel_rads: Driven-wheel angular speed, rad/s.
        current_gear: Currently engaged 0-indexed gear.
        config: Powertrain parameters.

    Returns:
        The gear to engage, always within range.
    """
    gear = int(np.clip(current_gear, 0, config.num_gears - 1))
    up_limit = config.max_engine_rads * config.shift_up_fraction
    down_limit = config.max_engine_rads * config.shift_down_fraction

    if engine_speed_rads(wheel_rads, gear, config) > up_limit and gear < config.num_gears - 1:
        return gear + 1
    if engine_speed_rads(wheel_rads, gear, config) < down_limit and gear > 0:
        return gear - 1
    return gear


def drive_torque(throttle: float, wheel_rads: float, gear: int, config: PowertrainConfig) -> float:
    """Torque delivered to **one** driven wheel, in N·m.

    Args:
        throttle: 0 to 1.
        wheel_rads: That wheel's angular speed, rad/s.
        gear: Engaged 0-indexed gear.
        config: Powertrain parameters.

    Returns:
        Torque for a single driven wheel, already halved for a two-wheel-drive axle.
    """
    throttle = float(np.clip(throttle, 0.0, 1.0))
    if throttle <= 0.0:
        return 0.0
    ratio = config.total_ratio(gear)
    crank = engine_torque(engine_speed_rads(wheel_rads, gear, config), config)
    axle = crank * ratio * config.driveline_efficiency * throttle
    return axle / 2.0


def brake_torque(
    brake: float,
    wheel_rads: float,
    front: bool,
    config: PowertrainConfig,
    *,
    dt: float,
    wheel_inertia: float,
) -> float:
    """Braking torque for **one** wheel, in N·m, signed to oppose rotation.

    Constant magnitude rather than proportional to speed, which is the whole point: the
    previous damper-based brakes faded to nothing at low speed.

    The magnitude is clamped to what would bring this wheel exactly to rest within one
    timestep. That is what preserves the one property the damper gave for free — a brake
    can stop a wheel but must never drive it backwards — without the sign chatter of an
    unclamped opposing torque near zero speed.

    Args:
        brake: 0 to 1.
        wheel_rads: That wheel's angular speed, rad/s.
        front: True for a front wheel, which gets the larger share under ``brake_bias_front``.
        config: Powertrain parameters.
        dt: Control timestep, seconds. Used for the anti-reversal clamp.
        wheel_inertia: Rotational inertia of the wheel, kg·m², for the same clamp.

    Returns:
        Signed torque opposing rotation; ``0.0`` when the wheel is already stopped.
    """
    brake = float(np.clip(brake, 0.0, 1.0))
    if brake <= 0.0 or wheel_rads == 0.0:
        return 0.0
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if wheel_inertia <= 0:
        raise ValueError(f"wheel_inertia must be positive, got {wheel_inertia}")

    axle_share = config.brake_bias_front if front else (1.0 - config.brake_bias_front)
    # Two wheels per axle.
    magnitude = config.max_brake_torque_nm * axle_share * brake / 2.0
    # Never more than enough to stop this wheel this step.
    magnitude = min(magnitude, abs(wheel_rads) * wheel_inertia / dt)
    return -float(np.sign(wheel_rads)) * magnitude
