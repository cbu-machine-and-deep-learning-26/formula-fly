"""Aerodynamic downforce and drag (GH-16).

This is the single most important piece of physics for an F1 car, and the simulation had
none of it. Without downforce a car is capped near ``mu * g`` in every direction -- about
1.6 g with racing slicks. A 2017 Formula 1 car brakes and corners at 5-6 g, and it does so
*only* because aerodynamic load grows with the square of speed and multiplies tyre grip.
No amount of tyre-friction tuning gets there; the missing term is the whole difference.

Measured on the build before this module existed: 0-100 km/h in 4.85 s, 1.37 g braking,
and no top speed at all -- with no drag the car simply accelerated until the integrator
broke at 667 km/h.

Deliberately pure numpy with no MuJoCo import, so the physics can be unit-tested without a
compiled model or a GL context, and so a reviewer can check the arithmetic against the
published figures in :data:`SF70H_AERO` without running a simulation.

Reference model
---------------

Both forces follow the standard quadratic form::

    F = 0.5 * rho * C * A * v^2

``C * A`` is carried as a single coefficient (``cla``, ``cda``) rather than splitting
coefficient from reference area. Teams quote ``ClA`` and ``CdA`` that way because only the
product is measurable, and splitting them invents a frontal-area number nobody published.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["AeroConfig", "SF70H_AERO", "drag_n", "downforce_n", "terminal_speed_mps"]

#: Sea-level air density at 15 C, kg/m^3. The ISA standard, and what teams quote against.
STANDARD_AIR_DENSITY = 1.225


@dataclass(frozen=True)
class AeroConfig:
    """Aerodynamic coefficients.

    Args:
        cla: Downforce coefficient times reference area, in m^2. Downforce at speed ``v``
            is ``0.5 * rho * cla * v^2``, acting downward in the car's own frame.
        cda: Drag coefficient times reference area, in m^2, opposing the velocity vector.
        air_density: kg/m^3. Lower at altitude, which is why cars are set up differently
            in Mexico City -- exposed rather than hardcoded for that reason.
        balance_front: Fraction of total downforce carried by the front axle. Sits slightly
            ahead of neutral so the centre of pressure ends up marginally *behind* the
            centre of gravity, which is what makes a car stable rather than
            aerodynamically twitchy at speed.

    Raises:
        ValueError: On non-positive coefficients or a balance outside ``(0, 1)``.
    """

    cla: float
    cda: float
    air_density: float = STANDARD_AIR_DENSITY
    balance_front: float = 0.45

    def __post_init__(self) -> None:
        if self.cla <= 0:
            raise ValueError(f"cla must be positive, got {self.cla}")
        if self.cda <= 0:
            raise ValueError(f"cda must be positive, got {self.cda}")
        if self.air_density <= 0:
            raise ValueError(f"air_density must be positive, got {self.air_density}")
        if not 0.0 < self.balance_front < 1.0:
            raise ValueError(f"balance_front must be in (0, 1), got {self.balance_front}")

    @property
    def lift_to_drag(self) -> float:
        """Downforce produced per unit of drag. 2017-era F1 sat around 3."""
        return self.cla / self.cda


#: Ferrari SF70H (2017). ``cla`` and ``cda`` are summed from Assetto Corsa's own wing data
#: for this car; ``balance_front`` is deliberately **not**, and that gap is the interesting
#: part.
#:
#: The coefficients used to be inferred -- ``cda`` back-solved from a ~340 km/h top speed,
#: ``cla`` from Copse being taken at 290 km/h -- which gave 4.0 and 1.35. Kunos models the
#: car as nine wings, each with a chord, span, angle and lookup tables for lift and drag
#: against angle of attack and ride height. Summing ``chord * span * C`` over all nine at a
#: 50 mm ride height gives 3.62 and 1.30, so the guesses were about 11% high on downforce
#: and 4% high on drag. Lift-to-drag lands at 2.79 against the ~3.0 the inference assumed.
#:
#: The same sum puts the centre of pressure 0.170 m **ahead** of the centre of gravity --
#: a front aero balance of 0.503, not the 0.45 kept here. That is a real number and it is
#: not adopted, because this model cannot run it. Measured at 250 km/h with 0.503: the car
#: reaches 28.9 degrees of sideslip in a fast corner at 1.8/2.0 grip and spins outright at
#: 1.8/1.9, and no friction pair tried both held the car and kept the rest of the envelope.
#: At 0.45 the same car settles at 4.9 degrees.
#:
#: The reason is the tyre model, not the aero. A real front-biased aero balance is stable
#: because the rear tyre keeps its grip coefficient better under load than the front does
#: (see :attr:`~fly_driver.envs.car.CarConfig.wheel_friction_rear`), and MuJoCo's friction
#: is exactly proportional to load with no way to express that. Adopting AC's balance needs
#: load sensitivity implemented first; until then 0.45 is an honest stand-in and this
#: comment is the record of what it stands in for.
#:
#: Also not modelled, from the same data: DRS, ride-height coupling (ClA runs 3.51 at 30 mm
#: to 3.56 at 70 mm, but balance swings from +0.296 m to +0.102 m), and the dynamic wing
#: controllers -- including one that scales front-floor downforce to zero at full steering
#: lock, modelling the floor stalling in yaw.
SF70H_AERO = AeroConfig(cla=3.62, cda=1.30, balance_front=0.45)


def _dynamic_pressure(speed_mps: npt.ArrayLike, config: AeroConfig) -> npt.NDArray[np.float64]:
    """``0.5 * rho * v^2``, the term both forces share."""
    speed = np.abs(np.asarray(speed_mps, dtype=np.float64))
    return 0.5 * config.air_density * speed**2


def downforce_n(speed_mps: npt.ArrayLike, config: AeroConfig) -> npt.NDArray[np.float64]:
    """Downforce in newtons at a given speed.

    Always non-negative: a car going backwards is still pushed *down*, not lifted. The
    absolute value inside :func:`_dynamic_pressure` is what guarantees that, and it matters
    because an RL policy early in training reverses constantly.

    Args:
        speed_mps: Speed in m/s. Scalar or array.
        config: Aerodynamic coefficients.
    """
    return _dynamic_pressure(speed_mps, config) * config.cla


def drag_n(speed_mps: npt.ArrayLike, config: AeroConfig) -> npt.NDArray[np.float64]:
    """Drag magnitude in newtons at a given speed.

    Returns a magnitude only. The direction is the caller's problem, because drag opposes
    the *velocity vector*, not the car's heading -- a detail that matters when the car is
    sliding sideways, which is exactly when the difference is largest.
    """
    return _dynamic_pressure(speed_mps, config) * config.cda


def terminal_speed_mps(power_w: float, config: AeroConfig) -> float:
    """Speed at which drag absorbs all available power, in m/s.

    Solves ``0.5 * rho * cda * v^3 = P``. This is the car's top speed on a flat road with
    no other losses, so it is the sanity check on ``cda``: if this is nowhere near the
    published top speed, the coefficient is wrong before any simulation is run.

    Args:
        power_w: Power available at the wheels, in watts.
        config: Aerodynamic coefficients.
    """
    if power_w <= 0:
        raise ValueError(f"power_w must be positive, got {power_w}")
    return float((2.0 * power_w / (config.air_density * config.cda)) ** (1.0 / 3.0))
