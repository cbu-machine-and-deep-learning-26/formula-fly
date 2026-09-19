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

__all__ = [
    "PowertrainConfig",
    "SF70H_POWERTRAIN",
    "abs_factor",
    "traction_factor",
    "brake_torque",
    "drive_torque",
    "limiter_fraction",
    "select_gear",
]

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
        torque_curve: Full-throttle crankshaft torque as ``(rpm, N*m)`` points, rising
            rpm. Empty falls back to the flat-then-``P/omega`` idealisation built from
            ``peak_torque_nm`` and ``peak_power_w``, which is what every config without
            measured data gets.
        gear_ratios: Gearbox ratios, first to top.
        final_drive: Differential ratio, applied on top of the gear ratio.
        driveline_efficiency: Fraction of crank torque reaching the wheels.
        shift_up_fraction: Upshift when engine speed passes this fraction of the limiter.
            Must sit below the start of the limiter taper, with margin. Gear selection is
            referenced to ground speed while drive torque follows the wheel's true speed,
            which runs a few percent faster under load; with the upshift at 97% and the
            taper from 96%, that slip put the engine into the taper before the box would
            shift, the cut torque could no longer beat drag, and the car was trapped at
            284 km/h -- the 7th-to-8th shift point -- in 7th. Validated and tested.
        shift_down_fraction: Downshift below this fraction. The gap between the two is
            what stops the box hunting between gears at a steady speed.
        max_brake_torque_nm: Total braking torque across all four wheels at full pedal.
        brake_bias_front: Fraction of that torque on the front axle. Real cars run
            54-58% front; too far rearward and the car spins under braking.
        limiter_taper_fraction: Fraction of the rev range over which the limiter tapers
            torque to zero. A hard cut makes torque chatter on and off every step.
        abs_enabled: Slip-limit the brakes so a wheel cannot lock. Real F1 has no ABS,
            but Assetto Corsa offers it as a driving assist on this car, and here it is on
            by default for a measured reason: with brake torque sized past tyre grip (so
            high-speed braking can use the downforce), a 40% pedal at 150 km/h locked the
            front wheels 88% of the time. A locked, steered wheel has no directional grip
            -- the car went straight on and shook. See :func:`abs_factor`.
        abs_slip_full: Longitudinal slip below which the brakes get full torque.
        abs_slip_release: Slip at which brake torque is cut to zero. Torque ramps linearly
            between the two, which settles the wheel near 12-18% slip rather than
            bang-banging between rolling and locked. Tightened from 12/30% after Payton
            asked for a little more.
        traction_control_enabled: Slip-limit the drive torque so the rear wheels cannot
            spin up. Assetto Corsa has this **active** on the SF70H --
            ``SLIP_RATIO_LIMIT=0.10``, ``ACTIVE=1``, above 30 km/h -- and has ABS switched
            off, which is the opposite of what this model had. Without it the car spins on
            the throttle out of slow corners, which is what Payton hit. It is still on;
            what changed is how much slip it allows before intervening, which
            ``traction_slip_full`` covers.
        traction_slip_full: Wheelspin slip below which the engine gets full torque.

            0.20, which is **looser than Assetto Corsa's 0.10** and a deliberate departure
            from matching it. At 0.10 the car could not be made to break traction by hand:
            full lock and full throttle at 60 km/h produced 1.9 degrees of sideslip, which
            is a rail, not a car. Payton's call, on the grounds that a car that cannot
            oversteer teaches a driver -- or a policy -- nothing about catching one.

            Measured at 60 km/h with the clumsiest input available, sideslip goes 1.9
            degrees at 0.10, 10.6 at 0.20, and 24.4 with the limiter switched off
            entirely. 0.20 is the setting where the back steps out and can still be
            caught. 0-100 km/h is 2.40 s at all three, so none of this is paid for in a
            straight line.

            Worth knowing when this is revisited: AC's own SF70H really does run
            ``SLIP_RATIO_LIMIT=0.10, ACTIVE=1`` above 30 km/h, so 0.10 was the faithful
            value and this is not. If a policy trained here transfers badly because it
            has learned to catch slides the AC car will not give it, this is the first
            number to put back.
        traction_slip_cut: Slip at which drive torque is cut to zero, ramping linearly from
            ``traction_slip_full``. AC cuts on a curve rather than a ramp; this is the same
            shape the brake limiter already uses, so the two read alike. Widened with
            ``traction_slip_full`` to keep the ramp's width, so the limiter still eases in
            rather than becoming a switch.
        traction_min_speed_mps: Below this the limiter stands down, or it would strangle
            every standing start. AC uses 30 km/h.
        abs_min_speed_mps: Below this ground speed slip is ill-conditioned and the limiter
            stands down, so the car can be braked to a dead stop.

    Raises:
        ValueError: On non-positive values, an empty or non-descending gear set, or
            shift thresholds that would make the gearbox hunt.
    """

    peak_power_w: float
    peak_torque_nm: float
    max_engine_rads: float
    idle_engine_rads: float
    max_brake_torque_nm: float
    torque_curve: tuple[tuple[float, float], ...] = ()
    gear_ratios: tuple[float, ...] = _DEFAULT_GEAR_RATIOS
    final_drive: float = _DEFAULT_FINAL_DRIVE
    driveline_efficiency: float = 0.90
    shift_up_fraction: float = 0.94
    shift_down_fraction: float = 0.55
    brake_bias_front: float = 0.54
    limiter_taper_fraction: float = 0.04
    abs_enabled: bool = True
    abs_slip_full: float = 0.10
    abs_slip_release: float = 0.25
    abs_min_speed_mps: float = 2.0
    traction_control_enabled: bool = True
    traction_slip_full: float = 0.20
    traction_slip_cut: float = 0.45
    traction_min_speed_mps: float = 8.3

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
        if self.shift_up_fraction > 1.0 - self.limiter_taper_fraction - 0.01:
            raise ValueError(
                f"shift_up_fraction={self.shift_up_fraction} is inside the limiter taper "
                f"(from {1.0 - self.limiter_taper_fraction}); torque is cut before the box "
                f"shifts and the car can be trapped below the shift point"
            )
        if not 0 <= self.traction_slip_full < self.traction_slip_cut <= 1:
            raise ValueError(
                "need 0 <= traction_slip_full < traction_slip_cut <= 1, got "
                f"{self.traction_slip_full} and {self.traction_slip_cut}"
            )
        if not 0.0 <= self.abs_slip_full < self.abs_slip_release <= 1.0:
            raise ValueError(
                f"need 0 <= abs_slip_full < abs_slip_release <= 1, got "
                f"{self.abs_slip_full} and {self.abs_slip_release}"
            )
        if self.abs_min_speed_mps < 0.0:
            raise ValueError(
                f"abs_min_speed_mps must be non-negative, got {self.abs_min_speed_mps}"
            )
        if not 0.0 <= self.limiter_taper_fraction < 1.0:
            raise ValueError(
                f"limiter_taper_fraction must be in [0, 1), got {self.limiter_taper_fraction}"
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


#: Full-throttle crankshaft torque, in ``(rpm, N*m)``. Derived from Assetto Corsa's data
#: for this car: its naturally aspirated torque table multiplied by the turbo model in the
#: same file (3.5 bar maximum, 3.4 bar wastegate, spooled by 5,500 rpm), then sampled at
#: round engine speeds. Peak torque is 642 N*m at 10,000 rpm and peak power 758 kW
#: (1,016 hp) at 12,000 -- the ~1000 hp the car is usually quoted at.
#:
#: The collapse above 12,000 rpm is real and is the reason this table exists at all. The
#: old flat-then-``P/omega`` idealisation held peak torque all the way to the limiter,
#: which made revving it out free; here a third of the torque is gone by 13,000.
#:
#: **ERS is not included.** The same data adds 182 N*m of electrical torque at low rpm
#: falling to 66 at the limiter -- with it, peak power would be 867 kW (1,162 hp). It is
#: left out because it is energy-limited (4 MJ per lap in the data, the FIA allowance) and
#: deployed in bursts, so modelling it as always-on would flatter the car everywhere. This
#: engine is therefore the ERS-depleted case, which is the conservative one.
SF70H_TORQUE_CURVE: tuple[tuple[float, float], ...] = (
    (2_000.0, 227.0),
    (3_000.0, 291.0),
    (4_000.0, 355.0),
    (5_000.0, 535.0),
    (6_000.0, 559.0),
    (7_000.0, 581.0),
    (8_000.0, 612.0),
    (9_000.0, 634.0),
    (10_000.0, 642.0),
    (11_000.0, 616.0),
    (12_000.0, 603.0),
    (13_000.0, 414.0),
    (14_000.0, 356.0),
    (15_000.0, 277.0),
)

#: Ferrari SF70H, from Assetto Corsa's own drivetrain and engine data for this car.
#:
#: The eight ratios and the 4.42 final drive are AC's. They are much taller than the
#: 6.26 final drive guessed before -- top gear went from 5.57 overall to 4.60 -- which is
#: what lets eighth reach terminal velocity instead of running into the limiter.
#:
#: ``shift_up_fraction`` is computed from the curve rather than set near the limiter.
#: With roughly a 1.10 step between the upper gears, the crossover where the power after
#: an upshift matches the power before it sits near 12,300 rpm; above that the torque
#: collapse costs more than the extra revs are worth. 0.82 of 15,000 puts the shift there.
#: The old 0.94 shifted at 14,100, deep into the part of the curve that has given up.
SF70H_POWERTRAIN = PowertrainConfig(
    peak_power_w=758_000.0,
    peak_torque_nm=642.0,
    torque_curve=SF70H_TORQUE_CURVE,
    max_engine_rads=15_000.0 * 2.0 * np.pi / 60.0,
    idle_engine_rads=2_950.0 * 2.0 * np.pi / 60.0,
    gear_ratios=(2.9688, 2.3943, 2.0411, 1.7155, 1.4800, 1.2800, 1.1400, 1.0400),
    final_drive=4.4200,
    shift_up_fraction=0.82,
    # Assetto Corsa's brakes.ini gives MAX_TORQUE=3900 for this car, which is per wheel:
    # read as a whole-car total it stops the car at 1.84 g and takes 111 m from 300 km/h,
    # which no Formula 1 car does. Read per wheel -- 15.6 kNm across four -- it gives
    # 4.67 g and 43.4 m, identical to the 20 kNm of headroom guessed before, because in
    # both cases the tyres run out before the discs do. So this is AC's number, and the
    # measurement that says which reading of it is the right one.
    #
    # The headroom above grip is also what lets a driver lock a wheel, which is a real
    # thing a policy can do wrong.
    #
    # Not modelled: the real car harvests energy through the rear axle under braking
    # (AC's ers.ini removes rear brake torque in proportion), so its rear discs do less
    # work than these do.
    max_brake_torque_nm=15_600.0,
)


def engine_torque(engine_rads: float, config: PowertrainConfig) -> float:
    """Crankshaft torque at a given engine speed, in N·m, at full throttle.

    With a ``torque_curve`` this interpolates it. Without one it falls back to flat at
    ``peak_torque_nm`` until the power limit bites, then ``P / omega`` -- a smooth
    idealisation for configs that have no measured data.

    The idealisation turned out to matter more than "the power ceiling is what counts"
    assumed. A real turbo hybrid does not hold peak torque to the limiter: this engine
    peaks at 10,000 rpm and has lost a third of it by 13,000, which moves the useful
    shift point a long way down from the rev limit.
    """
    speed = float(np.clip(engine_rads, config.idle_engine_rads, config.max_engine_rads))
    if config.torque_curve:
        rpm = speed * 60.0 / (2.0 * np.pi)
        points = config.torque_curve
        return float(np.interp(rpm, [p[0] for p in points], [p[1] for p in points]))
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


def limiter_fraction(engine_rads: float, config: PowertrainConfig) -> float:
    """Throttle multiplier from the rev limiter: 1 below the cut, 0 at it.

    **This is what stops a spinning wheel running away**, and leaving it out is not a
    subtle error. Without it a driven wheel gets full torque no matter how fast it spins.
    Measured with it missing: the rear wheel reached 4321 rad/s with 1435 m/s of slip, the
    car plateaued at 45 km/h on pure wheelspin, and the integrator produced NaN after 30 s.

    Tapered over the last few percent of the rev range rather than cut hard, because a hard
    cut makes torque chatter on and off every step as friction drags the wheel back below
    the threshold.
    """
    taper = config.max_engine_rads * config.limiter_taper_fraction
    if taper <= 0.0:
        return 1.0 if engine_rads < config.max_engine_rads else 0.0
    return float(np.clip((config.max_engine_rads - engine_rads) / taper, 0.0, 1.0))


def drive_torque(throttle: float, wheel_rads: float, gear: int, config: PowertrainConfig) -> float:
    """Torque delivered to **one** driven wheel, in N·m.

    ``wheel_rads`` must be the **actual** angular speed of that wheel, not a value derived
    from ground speed. The engine is geared to the wheel, so wheelspin revs it, and the
    resulting limiter cut is the only thing that bounds a spinning wheel.

    Args:
        throttle: 0 to 1.
        wheel_rads: That wheel's true angular speed, rad/s.
        gear: Engaged 0-indexed gear.
        config: Powertrain parameters.

    Returns:
        Torque for a single driven wheel, already halved for a two-wheel-drive axle.
    """
    throttle = float(np.clip(throttle, 0.0, 1.0))
    if throttle <= 0.0:
        return 0.0
    engine_rads = engine_speed_rads(wheel_rads, gear, config)
    ratio = config.total_ratio(gear)
    crank = engine_torque(engine_rads, config) * limiter_fraction(engine_rads, config)
    axle = crank * ratio * config.driveline_efficiency * throttle
    return axle / 2.0


def abs_factor(
    ground_speed_mps: float, wheel_rads: float, wheel_radius_m: float, config: PowertrainConfig
) -> float:
    """Multiplier on one wheel's brake torque from the slip limiter: 1 rolling, 0 locked.

    Longitudinal slip is ``(v - omega * r) / v``: 0 for a freely rolling wheel, 1 for a
    locked one. Full torque up to ``abs_slip_full``, zero at ``abs_slip_release``, linear
    between. Below ``abs_min_speed_mps`` the ratio is ill-conditioned and the limiter
    stands down so the car can actually stop.

    The ground speed is the car's, not the wheel's own -- in a corner the inner and outer
    wheels differ by half the track width times the yaw rate, under a metre per second at
    racing speed. ``abs_slip_full`` leaves margin for that.
    """
    if not config.abs_enabled or ground_speed_mps < config.abs_min_speed_mps:
        return 1.0
    slip = (ground_speed_mps - abs(float(wheel_rads)) * wheel_radius_m) / ground_speed_mps
    if slip <= config.abs_slip_full:
        return 1.0
    if slip >= config.abs_slip_release:
        return 0.0
    return 1.0 - (slip - config.abs_slip_full) / (config.abs_slip_release - config.abs_slip_full)


def traction_factor(
    ground_speed_mps: float, wheel_rads: float, wheel_radius_m: float, config: PowertrainConfig
) -> float:
    """Multiplier on one driven wheel's torque from traction control: 1 gripping, 0 spinning.

    The mirror image of :func:`abs_factor`. Wheelspin slip is
    ``(omega * r - v) / (omega * r)``: 0 for a wheel matching the road, approaching 1 for
    one spinning freely against a stationary car. Full torque up to ``traction_slip_full``,
    nothing at ``traction_slip_cut``, linear between.

    Referenced to the wheel rather than the road because that is the term that blows up:
    at the moment of a spin the car is barely moving and the wheel is doing hundreds of
    rad/s, so dividing by ground speed would give a slip of thousands and no useful ramp.
    """
    if not config.traction_control_enabled:
        return 1.0
    if ground_speed_mps < config.traction_min_speed_mps:
        return 1.0
    surface = abs(float(wheel_rads)) * wheel_radius_m
    if surface <= 1e-6:
        return 1.0
    slip = (surface - ground_speed_mps) / surface
    if slip <= config.traction_slip_full:
        return 1.0
    if slip >= config.traction_slip_cut:
        return 0.0
    return 1.0 - (slip - config.traction_slip_full) / (
        config.traction_slip_cut - config.traction_slip_full
    )


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
