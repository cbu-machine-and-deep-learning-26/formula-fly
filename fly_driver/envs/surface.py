"""Grip by surface: asphalt on the circuit, grass off it (GH-16).

Until this existed there was no grass in the physics at all. Two things hid that.

**The track ribbon is visual, not collidable** (see :mod:`fly_driver.envs.scene`) -- the car
runs on one infinite ground plane, so every wheel everywhere was on the same surface.

**MuJoCo combines contact friction by element-wise maximum**, not by any kind of average, so
the ground plane's own ``friction`` was dead the moment it was lower than the tyre's. With a
1.8 tyre on a 1.0 plane the contact came out at 1.8; lowering the plane changed nothing, and
would have gone on changing nothing however low it went.

So grip is carried entirely by the wheel geoms, and the ground plane is deliberately set
slippier than any tyre so that the maximum is always the tyre's own value. This class then
scales each wheel's sliding friction by where that wheel actually is.

**Per wheel, not per car.** Two wheels on the grass and two on the asphalt is the whole
point: the asymmetry is what spins the car, and a single car-wide grip factor could not
produce it.

**Blended across the wheel's width** rather than switched. A wheel half over the line gets
half the difference. A hard step would put a discontinuity in the contact forces exactly
where the car is most loaded, which MuJoCo resolves as a jolt.

**Applied once per control step**, not per physics substep. Projecting four wheels onto the
centerline costs about 190 us a step, against 1.9 ms if it were done every substep -- a
tenth of the 20 ms frame, for a grip transition that cannot be acted on faster than the
50 Hz the policy runs at anyway. The lag is at most one frame.

Only the sliding coefficient is scaled. Grass also has far higher rolling resistance, which
would slow a car that ran wide rather than merely unsettle it; that is a separate effect and
is deliberately not modelled here.
"""

from __future__ import annotations

import mujoco
import numpy as np

from fly_driver.envs.car import CarConfig
from fly_driver.envs.centerline import Centerline

__all__ = ["WHEEL_NAMES", "SurfaceGrip"]

#: The four wheels, in the order this class reports their grip.
WHEEL_NAMES: tuple[str, str, str, str] = ("fl", "fr", "rl", "rr")


class SurfaceGrip:
    """Scales each wheel's friction by the surface under it.

    Args:
        model: Compiled model containing the four ``wheel_*`` bodies and geoms.
        centerline: The circuit, for deciding where the track edge is.
        car: Vehicle parameters, for the wheel widths. Defaults to :class:`CarConfig`.
        kerb_width_m: Width of the kerb outside the track edge, which grips like asphalt.
            The same value the track-limits rule uses, so what a driver is penalised for
            and what costs them grip are the same line.
        grass_friction_scale: Fraction of the tyre's asphalt grip left on the grass.

    Raises:
        ValueError: If the model is missing a wheel, if the scale is not in ``(0, 1]``, or
            if ``kerb_width_m`` is negative.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        centerline: Centerline,
        *,
        car: CarConfig | None = None,
        kerb_width_m: float = 0.0,
        grass_friction_scale: float = 0.35,
    ) -> None:
        if not 0.0 < grass_friction_scale <= 1.0:
            raise ValueError(f"grass_friction_scale must be in (0, 1], got {grass_friction_scale}")
        if kerb_width_m < 0:
            raise ValueError(f"kerb_width_m must be non-negative, got {kerb_width_m}")

        self._model = model
        self.centerline = centerline
        self.car = car or CarConfig()
        self.kerb_width_m = float(kerb_width_m)
        self.grass_friction_scale = float(grass_friction_scale)

        self._bodies: list[int] = []
        self._geoms: list[int] = []
        for name in WHEEL_NAMES:
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"wheel_{name}")
            geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{name}_geom")
            if body < 0 or geom < 0:
                raise ValueError(f"model has no wheel named {name!r}")
            self._bodies.append(body)
            self._geoms.append(geom)

        # Asphalt grip comes from the config, deliberately, and NOT from the compiled
        # model. Reading it back from the model looks equivalent and is not: this class
        # mutates geom_friction in place, so a SurfaceGrip built while the car happened to
        # be on the grass would take the reduced value as its baseline and scale down from
        # there. Two of them on one model, or one rebuilt mid-episode, and grip decays
        # every time with nothing to show for it. The config cannot drift that way.
        self._baseline = np.array(
            [
                self.car.wheel_friction[0]
                if name.startswith("f")
                else self.car.wheel_friction_rear[0]
                for name in WHEEL_NAMES
            ],
            dtype=float,
        )
        self._widths = np.array(
            [
                self.car.wheel_width_front_m
                if name.startswith("f")
                else self.car.wheel_width_rear_m
                for name in WHEEL_NAMES
            ],
            dtype=float,
        )
        self._scales = np.ones(len(WHEEL_NAMES), dtype=float)

    @property
    def scales(self) -> tuple[float, ...]:
        """Grip multiplier applied to each of :data:`WHEEL_NAMES` at the last :meth:`apply`.

        ``1.0`` is full asphalt grip, :attr:`grass_friction_scale` is fully on the grass.
        """
        return tuple(float(value) for value in self._scales)

    @property
    def is_any_wheel_on_grass(self) -> bool:
        """Whether any wheel had less than full grip at the last :meth:`apply`."""
        return bool(np.any(self._scales < 1.0))

    def grass_fraction_at(self, x: float, y: float, width_m: float) -> float:
        """How much of a wheel of this width at ``(x, y)`` is past the kerb, 0 to 1.

        Blended over exactly one wheel width: ``0`` while the whole contact patch is on the
        circuit, ``1`` once all of it is beyond the kerb.
        """
        projection = self.centerline.project(float(x), float(y))
        edge = (
            projection.half_width_left if projection.lateral >= 0.0 else projection.half_width_right
        )
        beyond = abs(projection.lateral) - (edge + self.kerb_width_m)
        return float(min(1.0, max(0.0, (beyond + width_m / 2.0) / width_m)))

    def reset(self) -> None:
        """Put every wheel back on full asphalt grip.

        Worth calling at an episode boundary: the model is mutated in place, so a car
        respawned on the grid would otherwise keep whatever grip it had when it stopped.
        """
        self._scales[:] = 1.0
        for geom, baseline in zip(self._geoms, self._baseline, strict=True):
            self._model.geom_friction[geom][0] = baseline

    def apply(self, data: mujoco.MjData) -> None:
        """Set each wheel's sliding friction from where that wheel is now.

        Args:
            data: Simulation state. Read only; the friction lives on the model.
        """
        for index, (body, geom) in enumerate(zip(self._bodies, self._geoms, strict=True)):
            position = data.xpos[body]
            fraction = self.grass_fraction_at(
                float(position[0]), float(position[1]), float(self._widths[index])
            )
            scale = 1.0 + (self.grass_friction_scale - 1.0) * fraction
            self._scales[index] = scale
            self._model.geom_friction[geom][0] = self._baseline[index] * scale
