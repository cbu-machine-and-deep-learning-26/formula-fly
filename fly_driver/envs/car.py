"""Rigid-body car for the practice track (GH-16).

A chassis, four wheels on hinge joints, front-wheel steering, rear-wheel drive, and a
camera at the driver's head. Payton chose full rigid-body physics over a kinematic bicycle
model, so the wheels really do contact the ground plane and the car really can understeer,
slide, and spin.

**Actuator layout.** Eight controls, in this fixed order:

``[steer_fl, steer_fr, drive_rl, drive_rr, brake_fl, brake_fr, brake_rl, brake_rr]``

- **steer**: position actuators on the two front kingpin hinges. Both receive the same
  command; no Ackermann correction, which is a real simplification and is fine at the
  steering angles a racing line uses.
- **drive**: torque motors on the two rear wheel hinges.
- **brake**: MuJoCo ``damper`` actuators, which apply ``-kv * ctrl * qvel``. Using dampers
  rather than opposing-torque motors is deliberate: a damper's force is proportional to and
  opposed to the current wheel velocity, so braking can slow a wheel to a stop but can
  never drive it backwards. Hand-rolling that with a motor means computing
  ``-sign(qvel)`` every step and getting the zero-crossing right, which chatters.

Braking is applied to all four wheels while drive is rear-only, so the car brakes straight
but can spin its rear wheels out of a corner.

The env owns the mapping from a :class:`~fly_driver.interface.ControlVector` to these
eight numbers; see :func:`control_to_ctrl`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ACTUATOR_NAMES",
    "CarConfig",
    "car_assets_xml",
    "car_body_xml",
    "car_actuators_xml",
    "control_to_ctrl",
]

# Collision classes. MuJoCo lets two geoms touch when
# ``contype_a & conaffinity_b`` or ``contype_b & conaffinity_a`` is non-zero. MuJoCo only
# auto-excludes *direct* parent-child pairs, and a wheel is a grandchild of the chassis, so
# without these the wheels grind against the bodywork: the car crawls and yaws with the
# steering centred, which reads as a physics-tuning problem rather than a geometry overlap.
#: Ground and walls: collide with chassis and wheels, never with each other.
WORLD_CONTYPE, WORLD_CONAFFINITY = 1, 6
#: Chassis: collides with the world only.
CHASSIS_CONTYPE, CHASSIS_CONAFFINITY = 2, 1
#: Wheels: collide with the world only -- not the chassis, not each other.
WHEEL_CONTYPE, WHEEL_CONAFFINITY = 4, 1

#: Actuator order in ``data.ctrl``. The env indexes by name at construction rather than
#: assuming this order survives, but the order is documented because it is what a human
#: reads in a MuJoCo viewer.
ACTUATOR_NAMES = (
    "steer_fl",
    "steer_fr",
    "drive_rl",
    "drive_rr",
    "brake_fl",
    "brake_fr",
    "brake_rl",
    "brake_rr",
)


@dataclass(frozen=True)
class CarConfig:
    """Vehicle parameters, all in SI units.

    Defaults are a light open-wheel car: roughly Formula-ish mass and wheelbase without
    pretending to model a specific chassis. They are collected here rather than scattered
    through the XML so that tuning is one edit and so a reviewer can see the whole vehicle
    at once.

    Args:
        mass_kg: Chassis mass. Wheel mass is separate and small.
        wheelbase_m: Front-to-rear axle distance.
        track_width_m: Left-to-right wheel separation.
        chassis_length_m: Visual body length.
        chassis_width_m: Visual body width.
        chassis_height_m: Visual body height.
        wheel_radius_m: Wheel radius; also sets the chassis ride height.
        wheel_width_m: Wheel width.
        wheel_mass_kg: Mass of one wheel.
        max_steer_rad: Steering lock at the kingpin, each way.
        steer_gain: Position-actuator stiffness for the steering. High enough that the
            wheels track the command against tyre scrub.
        drive_gear: Torque per unit of throttle at each driven wheel, in N*m.
        brake_gain: Damper coefficient per unit of brake at each wheel.
        wheel_friction: MuJoCo ``friction`` triple for the tyres: sliding, torsional,
            rolling. Sliding friction above 1 is what stops an open-wheel car understeering
            off the road at the first corner.
        camera_forward_m: Camera offset ahead of the chassis centre.
        camera_height_m: Camera height above the chassis centre. Together these put the
            eye where a driver's head sits.
        camera_fovy_deg: Vertical field of view. A real fly sees nearly panoramically; a
            single pinhole camera cannot, so this is a compromise **GH-13 should choose**
            once the hex resampler's coverage is known.
        wheel_damping: Small viscous damping on the wheel hinges. Keeps a free-rolling
            wheel from spinning up indefinitely and settles contact jitter.
        wheel_armature: Rotor inertia added to the wheel hinges. Mostly a stability aid:
            without it, a light wheel driven by a strong motor needs a much smaller
            timestep.
        hub_mass_kg: Mass of each steering upright. Required, not cosmetic: MuJoCo rejects
            any body carrying a joint that has no inertia of its own, and a child body's
            mass does not satisfy it.
    """

    mass_kg: float = 750.0
    wheelbase_m: float = 3.0
    track_width_m: float = 1.6
    chassis_length_m: float = 4.6
    chassis_width_m: float = 1.4
    chassis_height_m: float = 0.5
    wheel_radius_m: float = 0.33
    wheel_width_m: float = 0.30
    wheel_mass_kg: float = 15.0
    max_steer_rad: float = 0.55
    steer_gain: float = 6000.0
    drive_gear: float = 900.0
    brake_gain: float = 700.0
    wheel_friction: tuple[float, float, float] = (1.6, 0.01, 0.001)
    camera_forward_m: float = 0.4
    camera_height_m: float = 0.55
    camera_fovy_deg: float = 90.0
    wheel_damping: float = 0.8
    wheel_armature: float = 0.6
    hub_mass_kg: float = 8.0

    def __post_init__(self) -> None:
        positives = {
            "mass_kg": self.mass_kg,
            "wheelbase_m": self.wheelbase_m,
            "track_width_m": self.track_width_m,
            "wheel_radius_m": self.wheel_radius_m,
            "wheel_width_m": self.wheel_width_m,
            "wheel_mass_kg": self.wheel_mass_kg,
            "max_steer_rad": self.max_steer_rad,
            "drive_gear": self.drive_gear,
            "brake_gain": self.brake_gain,
            "hub_mass_kg": self.hub_mass_kg,
        }
        for name, value in positives.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 0 < self.camera_fovy_deg < 180:
            raise ValueError(f"camera_fovy_deg must be in (0, 180), got {self.camera_fovy_deg}")
        if self.max_steer_rad >= np.pi / 2:
            raise ValueError(
                f"max_steer_rad={self.max_steer_rad} is at or past 90 degrees; "
                f"the kingpin would fold the wheel sideways"
            )

    @property
    def ride_height_m(self) -> float:
        """Chassis centre height when the wheels are resting on flat ground."""
        return self.wheel_radius_m


def _wheel_body_xml(name: str, x: float, y: float, config: CarConfig, *, steerable: bool) -> str:
    """One wheel, optionally wrapped in a steering hinge body.

    The cylinder's axis is set with ``zaxis="0 1 0"`` so it rolls about the car's lateral
    axis. A default-oriented cylinder would stand on its edge like a drum.
    """
    friction = " ".join(str(value) for value in config.wheel_friction)
    wheel = f"""
        <body name="wheel_{name}">
          <joint name="roll_{name}" type="hinge" axis="0 1 0"
                 damping="{config.wheel_damping}" armature="{config.wheel_armature}"/>
          <geom name="wheel_{name}_geom" type="cylinder" zaxis="0 1 0"
                size="{config.wheel_radius_m} {config.wheel_width_m / 2}"
                mass="{config.wheel_mass_kg}"
                friction="{friction}"
                contype="{WHEEL_CONTYPE}" conaffinity="{WHEEL_CONAFFINITY}"
                rgba="0.08 0.08 0.09 1"/>
        </body>"""

    if not steerable:
        return f"""
      <body name="hub_{name}" pos="{x} {y} 0">{wheel}
      </body>"""

    hub_inertia = config.hub_mass_kg * 0.01
    # The kingpin body carries a joint, and MuJoCo requires every *moving* body to have
    # mass and inertia of its own -- the child wheel's mass does not count. An explicit
    # <inertial> for the upright is the fix; without it the model refuses to compile.
    return f"""
      <body name="hub_{name}" pos="{x} {y} 0">
        <joint name="steer_{name}" type="hinge" axis="0 0 1"
               range="{-config.max_steer_rad} {config.max_steer_rad}"
               damping="2.0" armature="0.2"/>
        <inertial pos="0 0 0" mass="{config.hub_mass_kg}"
                  diaginertia="{hub_inertia} {hub_inertia} {hub_inertia}"/>{wheel}
      </body>"""


def car_assets_xml(config: CarConfig | None = None) -> str:
    """MJCF ``<asset>`` fragment for the car's materials."""
    del config  # Reserved: liveries or wheel textures would land here.
    return """
    <material name="car_body" rgba="0.75 0.12 0.12 1" specular="0.4" shininess="0.6"/>"""


def car_body_xml(
    position: np.ndarray | tuple[float, float],
    yaw: float,
    config: CarConfig | None = None,
) -> str:
    """MJCF ``<worldbody>`` fragment for the car, placed at a world pose.

    Args:
        position: ``(x, y)`` in metres. Height is set from the wheel radius so the car
            starts resting on the ground rather than dropping into it.
        yaw: Heading in radians, measured from the +x axis.
        config: Vehicle parameters. Defaults to :class:`CarConfig`.
    """
    config = config or CarConfig()
    half_base = config.wheelbase_m / 2.0
    half_track = config.track_width_m / 2.0

    wheels = "".join(
        [
            _wheel_body_xml("fl", half_base, half_track, config, steerable=True),
            _wheel_body_xml("fr", half_base, -half_track, config, steerable=True),
            _wheel_body_xml("rl", -half_base, half_track, config, steerable=False),
            _wheel_body_xml("rr", -half_base, -half_track, config, steerable=False),
        ]
    )

    # MuJoCo cameras look down their own -z with +y up. To look along the car's +x with
    # the world's +z up: z_cam = -x_car, y_cam = +z_car, so x_cam = y_cam X z_cam = -y_car.
    # Getting this wrong points the eye at the sky or out of the side of the car, which
    # renders plausibly and silently ruins every optic-flow measurement downstream.
    return f"""
    <body name="car" pos="{position[0]:.4f} {position[1]:.4f} {config.ride_height_m}"
          euler="0 0 {yaw:.6f}">
      <freejoint name="car_root"/>
      <geom name="chassis" type="box" material="car_body"
            size="{config.chassis_length_m / 2} {config.chassis_width_m / 2}
                  {config.chassis_height_m / 2}"
            mass="{config.mass_kg}"
            contype="{CHASSIS_CONTYPE}" conaffinity="{CHASSIS_CONAFFINITY}"/>
      <camera name="fly_head" mode="fixed"
              pos="{config.camera_forward_m} 0 {config.camera_height_m}"
              xyaxes="0 -1 0 0 0 1" fovy="{config.camera_fovy_deg}"/>{wheels}
    </body>"""


def car_actuators_xml(config: CarConfig | None = None) -> str:
    """MJCF ``<actuator>`` fragment: steering, drive, and brakes."""
    config = config or CarConfig()
    steer = "".join(
        f"""
    <position name="steer_{side}" joint="steer_{side}" kp="{config.steer_gain}"
              ctrlrange="{-config.max_steer_rad} {config.max_steer_rad}"/>"""
        for side in ("fl", "fr")
    )
    drive = "".join(
        f"""
    <motor name="drive_{side}" joint="roll_{side}" gear="{config.drive_gear}"
           ctrlrange="-1 1"/>"""
        for side in ("rl", "rr")
    )
    # Dampers oppose motion by construction: force = -kv * ctrl * qvel. A brake can stop a
    # wheel but can never reverse it, which is what makes a non-negative ctrlrange correct.
    brakes = "".join(
        f"""
    <damper name="brake_{side}" joint="roll_{side}" kv="{config.brake_gain}"
            ctrlrange="0 1"/>"""
        for side in ("fl", "fr", "rl", "rr")
    )
    return steer + drive + brakes


def control_to_ctrl(
    steer: float, throttle: float, brake: float, config: CarConfig | None = None
) -> dict[str, float]:
    """Map a normalised control triple onto named actuator commands.

    Keeping this a ``dict`` keyed by actuator name, rather than a positional array, means
    the env can look each one up by MuJoCo id. Reordering the XML then cannot silently
    swap throttle for brake.

    Args:
        steer: -1 (full left) to 1 (full right).
        throttle: 0 to 1.
        brake: 0 to 1.
        config: Vehicle parameters, for the steering lock.

    Returns:
        ``{actuator_name: command}`` covering every entry of :data:`ACTUATOR_NAMES`.
    """
    config = config or CarConfig()
    # Positive steer means right, and a right turn is a negative rotation about +z.
    steer_angle = -float(steer) * config.max_steer_rad
    commands = {
        "steer_fl": steer_angle,
        "steer_fr": steer_angle,
        "drive_rl": float(throttle),
        "drive_rr": float(throttle),
    }
    for side in ("fl", "fr", "rl", "rr"):
        commands[f"brake_{side}"] = float(brake)
    return commands
