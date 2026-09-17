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


#: Ferrari SF70H (2017). Chosen to reproduce published on-track performance rather than
#: copied from a spec sheet, because teams do not publish ClA or CdA.
#:
#: ``cda`` is set by top speed: at ~340 km/h the power available at the wheels equals
#: ``0.5 * rho * cda * v^3``. ``cla`` is set by cornering: Copse was taken at 290 km/h in
#: 2017, which needs roughly 5.5 g, which needs this much aero load on top of the car's
#: weight at ``mu = 1.7``. The resulting lift-to-drag of ~3.0 lands where 2017 cars did,
#: which is a useful independent check that the two numbers are mutually consistent.
#:
#: Replaceable: once Assetto Corsa is installed, the real wing coefficients are in
#: ``content/cars/ks_ferrari_sf70h/data.acd`` and should supersede these.
SF70H_AERO = AeroConfig(cla=4.0, cda=1.35, balance_front=0.45)


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
