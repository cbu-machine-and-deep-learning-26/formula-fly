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
- **drive**: torque motors on the two rear wheel hinges. The torque comes from the engine
  curve and the engaged gear in :mod:`fly_driver.envs.powertrain`.
- **brake**: torque motors on all four wheels. Constant torque opposing rotation, clamped
  so a wheel can be stopped but never driven backwards. See
  :func:`fly_driver.envs.powertrain.brake_torque` for why the earlier damper brakes, whose
  force faded with wheel speed, were replaced.

Braking is applied to all four wheels while drive is rear-only, so the car brakes straight
but can spin its rear wheels out of a corner.

:class:`CarDynamics` owns the mapping from a :class:`~fly_driver.interface.ControlVector`
to these eight numbers, and applies the aerodynamic forces on every substep. The bodywork
is visual only; physics runs on one collision box, so the car drives the same whatever it
looks like.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt

from fly_driver.envs.aero import SF70H_AERO, AeroConfig, downforce_n, drag_n
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.powertrain import (
    SF70H_POWERTRAIN,
    PowertrainConfig,
    abs_factor,
    brake_torque,
    drive_torque,
    select_gear,
)
from fly_driver.envs.scene import SceneConfig, build_scene_xml
from fly_driver.interface import ControlVector

__all__ = [
    "ACTUATOR_NAMES",
    "SF70H_REFERENCE",
    "CarConfig",
    "CarDynamics",
    "assemble_model_xml",
    "car_assets_xml",
    "car_body_xml",
    "car_actuators_xml",
    "car_tendons_xml",
    "steering_angle_rad",
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


#: Published Ferrari SF70H figures, kept next to the config they justify so the numbers can
#: be checked rather than trusted. Sources: Ferrari.com, Wikipedia, F1technical, and the 2017
#: FIA technical regulations.
#:
#: ===========================  ===================================
#: Mass incl. driver            728 kg (2017 FIA minimum)
#: Power                        ~746 kW / 1000 hp, 1.6 L V6 turbo hybrid
#: Length x width x height      5000 x 2000 x 950 mm
#: Wheelbase                    ~3600 mm
#: Track, front / rear          1600 / 1550 mm
#: Wheels                       13 inch, 670 mm diameter
#: Tyre width, front / rear     305 / 405 mm
#: Gearbox                      8-speed sequential
#: ===========================  ===================================
#:
#: Performance targets used for validation, all from 2017 at Silverstone specifically,
#: which beats generic figures because it is the circuit we actually simulate:
#:
#: - Copse taken at 290 km/h, the fastest proper corner in F1 that year
#: - Vale: 300 -> 194 km/h in ~71 m, about 5.5 g
#: - Peak cornering up to 6 g through the quick corners
#:
#: Note the overall width of 2000 mm is the car *including* its wheels. ``chassis_width_m``
#: is the bodywork only, so that 2 x (track/2 + tyre width/2) lands near 2 m.
SF70H_REFERENCE = {
    "mass_kg": 728.0,
    "power_w": 746_000.0,
    "wheelbase_m": 3.60,
    "overall_width_m": 2.00,
    "wheel_diameter_m": 0.670,
    "top_speed_kmh": 340.0,
    "zero_to_100_kmh_s": 2.6,
    "copse_speed_kmh": 290.0,
    "vale_braking_g": 5.5,
}


@dataclass(frozen=True)
class CarConfig:
    """Vehicle parameters, all in SI units.

    Defaults describe a **Ferrari SF70H** (2017), the F1 car in Assetto Corsa's Ferrari
    70th Anniversary pack. See :data:`SF70H_REFERENCE` for the published figures these come
    from. Every value lives here rather than scattered through the MJCF so that tuning is one
    edit, a reviewer can see the whole vehicle at once, and an Assetto-Corsa-derived parameter
    set can be dropped in later without touching physics code.

    Args:
        mass_kg: Total car mass including driver, carried by the chassis body. Wheel and
            upright masses are additional and small.
        wheelbase_m: Front-to-rear axle distance.
        track_width_front_m: Left-to-right front wheel separation.
        track_width_rear_m: Rear track. Narrower than the front on an SF70H.
        chassis_length_m: Visual body length.
        chassis_width_m: Visual body width.
        chassis_height_m: Height of the collision box. Shallow, so that the box bottom
            stays clear of the road when the suspension compresses under aero load; the
            visible bodywork is separate and sets its own heights.
        wheel_radius_m: Wheel radius; also sets the chassis ride height.
        wheel_width_front_m: Front tyre width. 305 mm on a 2017 car.
        wheel_width_rear_m: Rear tyre width. 405 mm -- rears are much wider than fronts.
        wheel_mass_kg: Mass of one wheel.
        front_weight_fraction: Share of static mass on the front axle. Single-seaters are
            rear-biased; the 2017 regulations floor was 44% front.
        centre_of_gravity_height_m: CoG height above the road. Low, which is what keeps an
            F1 car flat in a corner instead of rolling onto its side.
        inertia_roll_kgm2: Moment of inertia about the car's long axis.
        inertia_pitch_kgm2: About the lateral axis.
        inertia_yaw_kgm2: About the vertical axis -- the one that governs how quickly the
            car rotates into a corner. Stated explicitly because MuJoCo would otherwise
            derive ~1500 kg m^2 from the chassis box against a real car's ~750.
        max_steer_rad: Steering lock at the kingpin, each way.
        steer_gain: Position-actuator stiffness for the steering. High enough that the
            wheels track the command against tyre scrub.
        steer_damping_nms: Viscous damping on each kingpin. This is the steering damper a
            real car has, and it is not optional. The kingpin carries ~0.75 kg m^2 (the
            wheel and upright) against 12 kN m/rad of actuator stiffness, a 20 Hz mode;
            at the original 2.0 N m s/rad its damping ratio was **0.011**, so any
            asymmetry under braking rang the front wheels at 20 Hz -- the "shake" Payton
            reported. 150 gives a ratio near 0.8: settles in a few cycles, still snappy.
            A test computes the realised ratio from the compiled model so a heavier
            wheel cannot quietly bring the shimmy back.
        suspension_stiffness_front_n_m: Front coilover wheel rate. F1 springs are very
            stiff -- hundreds of N/mm -- because the car is held down by aerodynamic load,
            not by suspension travel, and the floor must not touch the road.
        suspension_stiffness_rear_n_m: Rear rate, stiffer than the front, as a
            single-seater runs it.
        suspension_damping_ns_m: Coilover damper rate. About 0.75 of critical for a
            quarter of the car on the front spring, and well over critical for the
            ~21 kg unsprung corner, so the wheels do not hop.
        suspension_travel_m: Bump-stop each way. Static sag is ~7 mm and aero load at
            top speed adds ~15 mm, so 40 mm leaves headroom for kerbs and braking.
        anti_roll_stiffness_n_m: Anti-roll bar rate per axle, applied as a fixed tendon
            on the difference between the two sides' suspension travel. Resists roll,
            does nothing in pure heave.

            An earlier attempt at suspension was abandoned because it cost grip and top
            speed. That attempt also softened the tyre contact and was measured with
            contact counts; separated out, the contact softening was the whole loss
            (326 -> 284 km/h on its own). This version leaves the contacts alone and is
            measured on settled lateral g, top speed and stopping distance.
        max_actuator_torque_nm: Control range of the drive and brake motors, in N*m.
            A ceiling, not a setpoint -- actual torque comes from the powertrain model.
            Wide enough that MuJoCo never silently clips a legitimate command.
        wheel_friction: MuJoCo ``friction`` triple for the tyres: sliding, torsional,
            rolling. 1.7 is mid-range for a dry slick (published 1.5-1.8).

            Raising it to chase more grip measurably makes the car **worse**. At 1.8 the
            front wheels' time on the ground through a corner fell from 71% to 56% and
            yaw-rate variation doubled, because more grip means more load transfer, which
            lifts wheels. Measured lateral grip did not improve.
        fly_mount_x_m: Longitudinal position of the ``fly_mount`` site in the body frame:
            the cockpit floor, where GH-21 attaches the tethered flybody. Slightly ahead
            of the wheelbase midpoint, as a driver's seat is.
        fly_mount_z_m: Height of that site above the axle line -- the top of the tub.
        camera_height_above_mount_m: The head camera sits this far above the fly mount,
            looking forward. 0.45 m is helmet-top height on a real car, and also where the
            broadcast T-cam is. When the body was a plain 0.35 m box the camera had to
            float over the front axle to see anything; with real bodywork the nose is low
            enough that the eye can sit where the fly actually is. A test measures how
            much of the frame the nose takes.
        camera_fovy_deg: Vertical field of view. A real fly sees nearly panoramically; a
            single pinhole camera cannot, so this is a compromise **GH-13 should choose**
            once the hex resampler's coverage is known.
        wheel_damping: Viscous damping on the wheel hinges, standing in for rolling
            resistance. Small on purpose. At 0.8 -- chosen as a stability aid before the
            powertrain existed -- the four wheels together absorbed 2242 N at 283 km/h,
            about 20x a real F1 car's rolling resistance and 30% of total drag. It capped
            top speed 40 km/h short and looked like an aero problem.
        wheel_armature: Rotor inertia added to the wheel hinges. Mostly a stability aid:
            without it, a light wheel driven by a strong motor needs a much smaller
            timestep.
        hub_mass_kg: Mass of each steering upright. Required, not cosmetic: MuJoCo rejects
            any body carrying a joint that has no inertia of its own, and a child body's
            mass does not satisfy it.
    """

    # --- Ferrari SF70H (2017). See SF70H_REFERENCE for sources. ---
    mass_kg: float = 728.0
    wheelbase_m: float = 3.60
    track_width_front_m: float = 1.60
    track_width_rear_m: float = 1.55
    chassis_length_m: float = 5.00
    chassis_width_m: float = 1.10
    chassis_height_m: float = 0.40
    wheel_radius_m: float = 0.335
    wheel_width_front_m: float = 0.305
    wheel_width_rear_m: float = 0.405
    wheel_mass_kg: float = 13.0
    front_weight_fraction: float = 0.455
    centre_of_gravity_height_m: float = 0.28
    inertia_roll_kgm2: float = 112.0
    inertia_pitch_kgm2: float = 700.0
    inertia_yaw_kgm2: float = 750.0
    max_steer_rad: float = 0.35
    steer_gain: float = 12000.0
    steer_damping_nms: float = 150.0
    suspension_stiffness_front_n_m: float = 250_000.0
    suspension_stiffness_rear_n_m: float = 300_000.0
    suspension_damping_ns_m: float = 10_000.0
    suspension_travel_m: float = 0.04
    anti_roll_stiffness_n_m: float = 50_000.0
    max_actuator_torque_nm: float = 20_000.0
    wheel_friction: tuple[float, float, float] = (1.7, 0.02, 0.001)
    fly_mount_x_m: float = 0.10
    fly_mount_z_m: float = 0.17
    camera_height_above_mount_m: float = 0.45
    camera_fovy_deg: float = 75.0
    wheel_damping: float = 0.05
    wheel_armature: float = 0.6
    hub_mass_kg: float = 8.0

    def __post_init__(self) -> None:
        positives = {
            "mass_kg": self.mass_kg,
            "wheelbase_m": self.wheelbase_m,
            "track_width_front_m": self.track_width_front_m,
            "track_width_rear_m": self.track_width_rear_m,
            "wheel_radius_m": self.wheel_radius_m,
            "wheel_width_front_m": self.wheel_width_front_m,
            "wheel_width_rear_m": self.wheel_width_rear_m,
            "wheel_mass_kg": self.wheel_mass_kg,
            "max_steer_rad": self.max_steer_rad,
            "max_actuator_torque_nm": self.max_actuator_torque_nm,
            "hub_mass_kg": self.hub_mass_kg,
            "steer_gain": self.steer_gain,
            "steer_damping_nms": self.steer_damping_nms,
            "suspension_stiffness_front_n_m": self.suspension_stiffness_front_n_m,
            "suspension_stiffness_rear_n_m": self.suspension_stiffness_rear_n_m,
            "suspension_damping_ns_m": self.suspension_damping_ns_m,
            "suspension_travel_m": self.suspension_travel_m,
        }
        if self.anti_roll_stiffness_n_m < 0:
            raise ValueError(
                f"anti_roll_stiffness_n_m must be non-negative, got {self.anti_roll_stiffness_n_m}"
            )
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
        if not 0.0 < self.front_weight_fraction < 1.0:
            raise ValueError(
                f"front_weight_fraction must be in (0, 1), got {self.front_weight_fraction}"
            )
        if self.centre_of_gravity_height_m >= self.wheel_radius_m * 2:
            raise ValueError(
                f"centre_of_gravity_height_m={self.centre_of_gravity_height_m} is above the top "
                f"of the wheels; a car that top-heavy would roll over in any corner"
            )
        for name in ("inertia_roll_kgm2", "inertia_pitch_kgm2", "inertia_yaw_kgm2"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

    @property
    def ride_height_m(self) -> float:
        """Body-frame origin height with the wheels resting on flat ground.

        The origin sits at axle height, so this is the wheel radius. The centre of gravity
        is lower, which is what :attr:`centre_of_gravity_offset_z_m` expresses.
        """
        return self.wheel_radius_m

    @property
    def centre_of_gravity_x_m(self) -> float:
        """Longitudinal CoG position relative to the body origin, in metres.

        Negative is rearward. A front weight fraction of 0.455 puts the CoG 54.5% of the
        wheelbase behind the front axle, which for a 3.6 m wheelbase is 162 mm behind the
        midpoint. This is what makes the car rear-biased like a real single-seater.
        """
        from_front = (1.0 - self.front_weight_fraction) * self.wheelbase_m
        return self.wheelbase_m / 2.0 - from_front

    @property
    def centre_of_pressure_x_m(self) -> float:
        """Longitudinal centre of pressure relative to the body origin, in metres.

        Derived from the aerodynamic balance the same way the centre of gravity is derived
        from the weight split. Placing it slightly *behind* the CoG is what makes a car
        aerodynamically stable rather than twitchy at speed.
        """
        from fly_driver.envs.aero import SF70H_AERO

        from_front = (1.0 - SF70H_AERO.balance_front) * self.wheelbase_m
        return self.wheelbase_m / 2.0 - from_front

    @property
    def centre_of_gravity_offset_z_m(self) -> float:
        """CoG height relative to the body origin (axle height), in metres.

        Negative, because an F1 car's centre of gravity sits below axle centreline.
        """
        return self.centre_of_gravity_height_m - self.wheel_radius_m


def _wheel_body_xml(name: str, x: float, y: float, config: CarConfig, *, steerable: bool) -> str:
    """One wheel, optionally wrapped in a steering hinge body.

    The cylinder's axis is set with ``zaxis="0 1 0"`` so it rolls about the car's lateral
    axis. A default-oriented cylinder would stand on its edge like a drum.

    Front and rear wheels differ in width: the SF70H runs 305 mm fronts and 405 mm rears.
    """
    friction = " ".join(str(value) for value in config.wheel_friction)
    width = config.wheel_width_front_m if name.startswith("f") else config.wheel_width_rear_m
    wheel = f"""
        <body name="wheel_{name}">
          <joint name="roll_{name}" type="hinge" axis="0 1 0"
                 damping="{config.wheel_damping}" armature="{config.wheel_armature}"/>
          <geom name="wheel_{name}_geom" type="cylinder" zaxis="0 1 0"
                size="{config.wheel_radius_m} {width / 2}"
                mass="{config.wheel_mass_kg}"
                friction="{friction}"
                contype="{WHEEL_CONTYPE}" conaffinity="{WHEEL_CONAFFINITY}"
                rgba="0.08 0.08 0.09 1"/>
        </body>"""

    # Sprung, damped upright. A slide joint carries the coilover; the kingpin, where
    # present, sits inside it; the wheel rolls inside that. Every body with a joint needs
    # inertia of its own -- the child wheel's mass does not count -- hence the <inertial>.
    stiffness = (
        config.suspension_stiffness_front_n_m
        if name.startswith("f")
        else config.suspension_stiffness_rear_n_m
    )
    hub_inertia = config.hub_mass_kg * 0.01
    steering = ""
    if steerable:
        steering = f"""
        <joint name="steer_{name}" type="hinge" axis="0 0 1"
               range="{-config.max_steer_rad} {config.max_steer_rad}"
               damping="{config.steer_damping_nms}" armature="0.2"/>"""
    return f"""
      <body name="hub_{name}" pos="{x} {y} 0">
        <inertial pos="0 0 0" mass="{config.hub_mass_kg}"
                  diaginertia="{hub_inertia} {hub_inertia} {hub_inertia}"/>
        <joint name="susp_{name}" type="slide" axis="0 0 1"
               range="{-config.suspension_travel_m} {config.suspension_travel_m}"
               stiffness="{stiffness}" damping="{config.suspension_damping_ns_m}"/>{steering}{wheel}
      </body>"""


def car_assets_xml(config: CarConfig | None = None) -> str:
    """MJCF ``<asset>`` fragment for the car's materials."""
    del config  # Reserved: liveries or wheel textures would land here.
    return """
    <material name="car_body" rgba="0.75 0.12 0.12 1" specular="0.4" shininess="0.6"/>
    <material name="car_carbon" rgba="0.10 0.10 0.11 1" specular="0.3" shininess="0.4"/>
    <material name="car_accent" rgba="0.92 0.92 0.92 1" specular="0.3" shininess="0.3"/>"""


#: Visual-only geom attributes. No contact, no mass, and group 1 so the default viewer and
#: the head camera both draw them. The collision box lives in group 3, which neither draws.
_VISUAL = 'contype="0" conaffinity="0" group="1"'


def _bodywork_xml(config: CarConfig) -> str:
    """Low-poly F1 bodywork from primitives: tub, nose, wings, sidepods, airbox, cover.

    Everything here is **visual only**. Physics still runs on the single collision box, so
    this changes nothing about how the car drives; it changes what the fly sees and what a
    human watching the demo sees. Proportions are the SF70H's, laid out in the body frame
    (origin at the wheelbase midpoint at axle height, +x forward, +z up) and assume the
    3.6 m wheelbase.

    The one shape that matters for the science is the nose. It sits directly in the head
    camera's lower field, and a tall nose would put static bodywork where the eye needs
    moving road. It is kept low and narrow for that reason, and a test measures how much
    of the frame it takes.
    """
    ground = -config.wheel_radius_m  # road surface, in the body frame
    fw_z = ground + 0.075  # front wing main plane, just clear of the kerbs
    return f"""
      <!-- monocoque / tub -->
      <geom name="body_tub" type="box" material="car_body" {_VISUAL}
            pos="0.15 0 0.0" size="1.05 0.30 0.17"/>
      <!-- nose: low and narrow so it stays out of the fly's view -->
      <geom name="body_nose" type="box" material="car_body" {_VISUAL}
            pos="1.90 0 -0.10" size="0.65 0.13 0.08"/>
      <!-- front wing: main plane and endplates -->
      <geom name="body_front_wing" type="box" material="car_carbon" {_VISUAL}
            pos="2.45 0 {fw_z:.3f}" size="0.18 0.90 0.015"/>
      <geom name="body_front_endplate_l" type="box" material="car_accent" {_VISUAL}
            pos="2.45 0.90 {fw_z + 0.06:.3f}" size="0.18 0.012 0.075"/>
      <geom name="body_front_endplate_r" type="box" material="car_accent" {_VISUAL}
            pos="2.45 -0.90 {fw_z + 0.06:.3f}" size="0.18 0.012 0.075"/>
      <!-- sidepods -->
      <geom name="body_sidepod_l" type="box" material="car_body" {_VISUAL}
            pos="-0.55 0.46 -0.02" size="0.85 0.16 0.15"/>
      <geom name="body_sidepod_r" type="box" material="car_body" {_VISUAL}
            pos="-0.55 -0.46 -0.02" size="0.85 0.16 0.15"/>
      <!-- airbox / roll hoop behind the cockpit -->
      <geom name="body_airbox" type="box" material="car_body" {_VISUAL}
            pos="-0.55 0 0.36" size="0.22 0.17 0.20"/>
      <!-- engine cover -->
      <geom name="body_engine_cover" type="box" material="car_body" {_VISUAL}
            pos="-1.35 0 0.12" size="0.55 0.16 0.13"/>
      <!-- rear wing: pylon, main plane, endplates -->
      <geom name="body_rear_pylon" type="box" material="car_carbon" {_VISUAL}
            pos="-1.95 0 0.28" size="0.04 0.03 0.18"/>
      <geom name="body_rear_wing" type="box" material="car_carbon" {_VISUAL}
            pos="-2.15 0 0.45" size="0.16 0.50 0.015"/>
      <geom name="body_rear_endplate_l" type="box" material="car_accent" {_VISUAL}
            pos="-2.15 0.50 0.30" size="0.20 0.012 0.20"/>
      <geom name="body_rear_endplate_r" type="box" material="car_accent" {_VISUAL}
            pos="-2.15 -0.50 0.30" size="0.20 0.012 0.20"/>
      <!-- Where the fly goes. GH-21 mounts the tethered flybody here and GH-25/26 pose the
           cockpit mesh from it, so this is the one frame the body track builds against:
           origin on the tub floor of the cockpit opening, +x forward, +z up. Group 4 keeps
           the marker out of the head camera; toggle it on in the viewer to see it. -->
      <site name="fly_mount" type="sphere" size="0.04" rgba="0.1 0.9 0.2 0.8" group="4"
            pos="{config.fly_mount_x_m} 0 {config.fly_mount_z_m}"/>"""


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
    half_front = config.track_width_front_m / 2.0
    half_rear = config.track_width_rear_m / 2.0
    camera_z = config.fly_mount_z_m + config.camera_height_above_mount_m

    wheels = "".join(
        [
            _wheel_body_xml("fl", half_base, half_front, config, steerable=True),
            _wheel_body_xml("fr", half_base, -half_front, config, steerable=True),
            _wheel_body_xml("rl", -half_base, half_rear, config, steerable=False),
            _wheel_body_xml("rr", -half_base, -half_rear, config, steerable=False),
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
      <!-- Mass and inertia are stated explicitly rather than derived from the chassis box.
           MuJoCo would compute Izz from a uniform 5 m box as ~1500 kg m^2 against a real
           F1 car's ~750, which halves yaw response and makes the car handle like a bus.
           The <inertial> x offset places the centre of gravity at the real front/rear
           weight split, and its z at CoG height, so weight transfer under braking and
           roll in a corner behave. -->
      <inertial pos="{config.centre_of_gravity_x_m:.4f} 0 {config.centre_of_gravity_offset_z_m:.4f}"
                mass="{config.mass_kg}"
                diaginertia="{config.inertia_roll_kgm2} {config.inertia_pitch_kgm2}
                             {config.inertia_yaw_kgm2}"/>
      <!-- The collision box. Physics runs on this alone; it is in group 3 so neither the
           viewer nor the head camera draws it, and the bodywork below is what you see. -->
      <geom name="chassis" type="box" mass="0" group="3" rgba="0.5 0.5 0.5 0.3"
            size="{config.chassis_length_m / 2} {config.chassis_width_m / 2}
                  {config.chassis_height_m / 2}"
            contype="{CHASSIS_CONTYPE}" conaffinity="{CHASSIS_CONAFFINITY}"/>{_bodywork_xml(config)}
      <!-- The fly's eye: directly above the fly_mount site at helmet height, which is also
           where a real onboard T-cam sits. Earlier the camera floated over the front axle
           purely to clear the old box-shaped body; now that the body is shaped like a car
           the camera can sit where the fly actually is. -->
      <camera name="fly_head" mode="fixed"
              pos="{config.fly_mount_x_m} 0 {camera_z:.4f}"
              xyaxes="0 -1 0 0 0 1" fovy="{config.camera_fovy_deg}"/>{wheels}
    </body>"""


def car_actuators_xml(config: CarConfig | None = None) -> str:
    """MJCF ``<actuator>`` fragment: steering, drive, and brakes.

    Drive and brake actuators are plain ``motor`` elements with ``gear="1"``, so their
    control value *is* a torque in newton-metres. That is deliberate: the torque is worked
    out in Python by :class:`CarDynamics` from the engine curve, the engaged gear and the
    brake model, and MuJoCo is only asked to apply it. Encoding a fixed gear ratio in the
    MJCF would mean the transmission lived in two places.

    Brakes were previously ``damper`` actuators, whose force is proportional to wheel
    speed. That made braking fade exactly when trying to stop -- 1.37 g from 150 km/h but
    0.6 g from 64 km/h. See :mod:`fly_driver.envs.powertrain`.
    """
    config = config or CarConfig()
    steer = "".join(
        f"""
    <position name="steer_{side}" joint="steer_{side}" kp="{config.steer_gain}"
              ctrlrange="{-config.max_steer_rad} {config.max_steer_rad}"/>"""
        for side in ("fl", "fr")
    )
    drive = "".join(
        f"""
    <motor name="drive_{side}" joint="roll_{side}" gear="1"
           ctrlrange="{-config.max_actuator_torque_nm} {config.max_actuator_torque_nm}"/>"""
        for side in ("rl", "rr")
    )
    brakes = "".join(
        f"""
    <motor name="brake_{side}" joint="roll_{side}" gear="1"
           ctrlrange="{-config.max_actuator_torque_nm} {config.max_actuator_torque_nm}"/>"""
        for side in ("fl", "fr", "rl", "rr")
    )
    return steer + drive + brakes


def car_tendons_xml(config: CarConfig | None = None) -> str:
    """MJCF ``<tendon>`` fragment: the anti-roll bars.

    A fixed tendon whose length is ``q_left - q_right`` of an axle's two suspension
    slides, given a stiffness, is exactly an anti-roll bar: it resists the two sides
    compressing differently and does nothing in pure heave.
    """
    config = config or CarConfig()
    return "".join(
        f"""
    <fixed name="arb_{axle}" stiffness="{config.anti_roll_stiffness_n_m}">
      <joint joint="susp_{axle}l" coef="1"/>
      <joint joint="susp_{axle}r" coef="-1"/>
    </fixed>"""
        for axle in ("f", "r")
    )


def assemble_model_xml(
    centerline: Centerline,
    scene: SceneConfig | None = None,
    car: CarConfig | None = None,
) -> str:
    """The complete MJCF for the car on a track, with every car fragment wired in.

    The one place that knows the car needs assets, bodies, actuators *and* tendons.
    Before this existed each caller assembled the fragments by hand, and forgetting one
    -- the anti-roll bars are the easy one to miss -- compiled fine and quietly produced
    a different car. Everything that builds a model goes through here.
    """
    car = car or CarConfig()
    position, yaw = centerline.pose_at(0.0)
    return build_scene_xml(
        centerline,
        scene,
        extra_assets=car_assets_xml(car),
        extra_bodies=car_body_xml(position, yaw, car),
        extra_actuators=car_actuators_xml(car),
        extra_tendons=car_tendons_xml(car),
    )


def steering_angle_rad(steer: float, config: CarConfig | None = None) -> float:
    """Kingpin angle for a normalised steering input.

    Positive ``steer`` means right, and a right turn is a negative rotation about +z.
    Nothing downstream can detect this being inverted -- a policy would simply learn the
    mirror image -- so the sign is pinned by tests rather than trusted.
    """
    config = config or CarConfig()
    return -float(steer) * config.max_steer_rad


class CarDynamics:
    """Applies aerodynamics and powertrain forces to a compiled MuJoCo model.

    This class has to exist. Aerodynamic forces are not part of the MJCF -- MuJoCo knows
    nothing about wings -- so they must be written into ``data.xfrc_applied`` on **every
    physics substep**. If each caller did that itself, the env, the manual drive script and
    the tests would drift apart and only some of them would be simulating an F1 car.
    Everything goes through :meth:`step`.

    It also owns the one piece of genuinely stateful vehicle behaviour: the engaged gear.

    Args:
        model: A compiled model containing a body named ``car`` and the actuators in
            :data:`ACTUATOR_NAMES`.
        car: Vehicle parameters. Defaults to the SF70H :class:`CarConfig`.
        aero: Aerodynamic coefficients. Defaults to :data:`~fly_driver.envs.aero.SF70H_AERO`.
        powertrain: Engine, gearbox and brakes. Defaults to
            :data:`~fly_driver.envs.powertrain.SF70H_POWERTRAIN`.

    Raises:
        ValueError: If the model lacks the car body, a wheel joint, or any expected actuator.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        car: CarConfig | None = None,
        aero: AeroConfig | None = None,
        powertrain: PowertrainConfig | None = None,
    ) -> None:
        self._model = model
        self.car = car or CarConfig()
        self.aero = aero or SF70H_AERO
        self.powertrain = powertrain or SF70H_POWERTRAIN

        self._body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
        if self._body < 0:
            raise ValueError("model has no body named 'car'")

        self._actuator: dict[str, int] = {}
        for name in ACTUATOR_NAMES:
            index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if index < 0:
                raise ValueError(f"model has no actuator named {name!r}")
            self._actuator[name] = index

        # Spin inertia of each wheel including armature, read from the compiled model
        # rather than recomputed, so the brake's anti-reversal clamp stays correct if the
        # wheel geometry changes.
        self._wheel_dof: dict[str, int] = {}
        self._wheel_inertia: dict[str, float] = {}
        for side in ("fl", "fr", "rl", "rr"):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"roll_{side}")
            if joint < 0:
                raise ValueError(f"model has no joint named 'roll_{side}'")
            dof = int(model.jnt_dofadr[joint])
            self._wheel_dof[side] = dof
            self._wheel_inertia[side] = float(model.dof_M0[dof])

        self._gear = 0

    @property
    def gear(self) -> int:
        """Currently engaged gear, 0-indexed."""
        return self._gear

    def reset(self) -> None:
        """Return to first gear. Call at an episode boundary."""
        self._gear = 0

    def _world_velocity(self, data: mujoco.MjData) -> npt.NDArray[np.float64]:
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self._model, data, mujoco.mjtObj.mjOBJ_BODY, self._body, velocity, 0
        )
        return velocity[3:6]

    def speed_mps(self, data: mujoco.MjData) -> float:
        """Ground speed of the car, in m/s."""
        return float(np.linalg.norm(self._world_velocity(data)[:2]))

    def apply_aero(self, data: mujoco.MjData) -> None:
        """Write this step's aerodynamic force and moment into ``data.xfrc_applied``.

        Drag opposes the **velocity vector**, not the heading, which matters exactly when
        the car is sliding. Downforce acts along the car's own downward axis rather than
        world -z, so a rolled car is still pressed onto the road instead of sideways.

        Both act at the centre of pressure. ``xfrc_applied`` is applied at the body's
        centre of mass, so the offset between the two is converted into an explicit
        moment -- which is what makes ``balance_front`` do anything at all rather than
        being a decorative config field.
        """
        world_velocity = self._world_velocity(data)
        speed = float(np.linalg.norm(world_velocity))
        rotation = data.xmat[self._body].reshape(3, 3)

        force = np.zeros(3)
        if speed > 1e-6:
            force -= float(drag_n(speed, self.aero)) * (world_velocity / speed)
        force -= float(downforce_n(speed, self.aero)) * rotation[:, 2]

        offset_body = np.array(
            [self.car.centre_of_pressure_x_m - self.car.centre_of_gravity_x_m, 0.0, 0.0]
        )
        data.xfrc_applied[self._body, :3] = force
        data.xfrc_applied[self._body, 3:] = np.cross(rotation @ offset_body, force)

    def actuator_commands(self, control: ControlVector, data: mujoco.MjData) -> dict[str, float]:
        """Torques and steering angles for this control input.

        Gear selection uses ground speed; drive torque uses each wheel's true speed.
                The split is deliberate and both halves matter -- see the inline comments.
        """
        speed = self.speed_mps(data)
        # Gear selection is referenced to ground speed. A spinning wheel reports a huge
        # angular velocity, which would upshift straight to top and collapse the torque.
        self._gear = select_gear(speed / self.car.wheel_radius_m, self._gear, self.powertrain)

        angle = steering_angle_rad(control.steer, self.car)
        commands = {"steer_fl": angle, "steer_fr": angle}

        # Drive torque uses each wheel's *actual* speed, because the engine is geared to
        # the wheel: wheelspin revs it and the limiter cut is what bounds the spin.
        # Referencing this to ground speed instead let a wheel accelerate without limit.
        for side in ("rl", "rr"):
            commands[f"drive_{side}"] = drive_torque(
                control.throttle,
                float(data.qvel[self._wheel_dof[side]]),
                self._gear,
                self.powertrain,
            )

        dt = float(self._model.opt.timestep)
        for side in ("fl", "fr", "rl", "rr"):
            wheel_rads = float(data.qvel[self._wheel_dof[side]])
            torque = brake_torque(
                control.brake,
                wheel_rads,
                front=side.startswith("f"),
                config=self.powertrain,
                dt=dt,
                wheel_inertia=self._wheel_inertia[side],
            )
            # Slip limiter. Without it a 40% pedal at 150 km/h locked the fronts 88% of
            # the time, and a locked, steered wheel steers nothing.
            torque *= abs_factor(speed, wheel_rads, self.car.wheel_radius_m, self.powertrain)
            commands[f"brake_{side}"] = torque
        return commands

    def step(self, control: ControlVector, data: mujoco.MjData, n_substeps: int = 1) -> None:
        """Advance the simulation, applying aero and powertrain forces each substep.

        The only supported way to step the car. Calling ``mujoco.mj_step`` directly skips
        aerodynamics entirely, which quietly turns a Formula 1 car back into a shopping
        trolley -- no downforce means about 1.6 g of grip instead of 6.

        Args:
            control: Steering, throttle and brake.
            data: Simulation state to advance, modified in place.
            n_substeps: Physics steps to take.
        """
        if n_substeps < 1:
            raise ValueError(f"n_substeps must be at least 1, got {n_substeps}")
        for _ in range(n_substeps):
            for name, value in self.actuator_commands(control, data).items():
                data.ctrl[self._actuator[name]] = value
            self.apply_aero(data)
            mujoco.mj_step(self._model, data)
