"""Rigid-body car tests (GH-16).

`AGENTS.md` §11 asks for tests rather than eyeballing, and a vehicle is full of failures
that compile and run while being wrong: a car that barely moves, a steering input that
does nothing, a brake that shoves the car backwards, a camera pointing at the sky.

The steering-sign tests matter most. Nothing downstream can tell you that left and right
are swapped -- the policy would simply learn the mirror image and the lesion map (GH-30)
would report the wrong cell types.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fly_driver.envs.car import (
    ACTUATOR_NAMES,
    SF70H_REFERENCE,
    CarConfig,
    CarDynamics,
    assemble_model_xml,
    steering_angle_rad,
)
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.scene import SceneConfig
from fly_driver.interface import ControlVector

CONFIG = CarConfig()


@pytest.fixture(scope="module")
def track() -> Centerline:
    """A long straight loop, so a 3-second full-throttle run stays on it."""
    points = [(0.0, 0.0), (400.0, 0.0), (400.0, 200.0), (0.0, 200.0)]
    return Centerline(points=points, half_width_right=[8.0] * 4, half_width_left=[8.0] * 4)


@pytest.fixture(scope="module")
def model(track) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        assemble_model_xml(track, SceneConfig(mesh_spacing_m=50.0), CONFIG)
    )


@pytest.fixture
def data(model) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data


def _actuator_ids(model: mujoco.MjModel) -> dict[str, int]:
    return {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        for name in ACTUATOR_NAMES
    }


def _drive(model, data, steer, throttle, brake, seconds) -> None:
    """Drive through CarDynamics, the only path that applies aerodynamics."""
    dynamics = CarDynamics(model, CONFIG)
    control = ControlVector.clipped(steer=steer, throttle=throttle, brake=brake)
    dynamics.step(control, data, int(seconds / model.opt.timestep))


def _body_id(model) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")


def _speed(model, data) -> float:
    return float(np.linalg.norm(data.cvel[_body_id(model)][3:5]))


def _yaw_rate(model, data) -> float:
    """Angular velocity about the world z axis. Positive is counter-clockwise (left)."""
    return float(data.cvel[_body_id(model)][2])


class TestModelStructure:
    def test_all_actuators_exist(self, model):
        for name in ACTUATOR_NAMES:
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) != -1

    def test_actuator_count(self, model):
        assert model.nu == len(ACTUATOR_NAMES) == 8

    def test_car_body_and_free_joint_exist(self, model):
        assert _body_id(model) != -1
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "car_root") != -1

    def test_four_wheels(self, model):
        for side in ("fl", "fr", "rl", "rr"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"wheel_{side}_geom") != -1

    def test_only_front_wheels_steer(self, model):
        for side in ("fl", "fr"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"steer_{side}") != -1
        for side in ("rl", "rr"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"steer_{side}") == -1

    def test_wheels_cannot_collide_with_the_chassis(self, model):
        """The overlap this prevents made the car crawl and yaw with the wheels centred."""
        chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
        wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wheel_fl_geom")
        collides = (model.geom_contype[chassis] & model.geom_conaffinity[wheel]) or (
            model.geom_contype[wheel] & model.geom_conaffinity[chassis]
        )
        assert not collides

    def test_wheels_still_collide_with_the_ground(self, model):
        ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        wheel = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wheel_fl_geom")
        collides = (model.geom_contype[wheel] & model.geom_conaffinity[ground]) or (
            model.geom_contype[ground] & model.geom_conaffinity[wheel]
        )
        assert collides


class TestCamera:
    def test_camera_exists(self, model):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fly_head") != -1

    def test_camera_looks_forward_not_at_the_sky(self, model, data):
        """A camera pointing up or sideways renders plausibly and silently destroys every
        optic-flow measurement that GH-13 and GH-14 depend on."""
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fly_head")
        # MuJoCo cameras look down their own -z.
        forward = -data.cam_xmat[cam].reshape(3, 3)[:, 2]
        car_forward = data.xmat[_body_id(model)].reshape(3, 3)[:, 0]
        assert float(np.dot(forward, car_forward)) > 0.99
        assert abs(float(forward[2])) < 0.05

    def test_camera_is_above_the_ground(self, model, data):
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fly_head")
        assert data.cam_xpos[cam][2] > 0.5


class TestSteeringMapping:
    def test_steer_right_is_a_negative_kingpin_angle(self):
        """Positive steer means right, and right is a clockwise (negative) rotation."""
        assert steering_angle_rad(1.0, CONFIG) < 0
        assert steering_angle_rad(-1.0, CONFIG) > 0

    def test_centred_steering_is_zero(self):
        assert steering_angle_rad(0.0, CONFIG) == 0.0

    def test_full_lock_matches_the_configured_limit(self):
        assert abs(steering_angle_rad(1.0, CONFIG)) == pytest.approx(CONFIG.max_steer_rad)

    def test_scales_linearly(self):
        assert steering_angle_rad(0.5, CONFIG) == pytest.approx(
            steering_angle_rad(1.0, CONFIG) * 0.5
        )


class TestDynamicsCommands:
    def test_covers_every_actuator(self, model, data):
        dynamics = CarDynamics(model, CONFIG)
        commands = dynamics.actuator_commands(ControlVector.neutral(), data)
        assert set(commands) == set(ACTUATOR_NAMES)

    def test_both_front_wheels_get_the_same_steering_angle(self, model, data):
        dynamics = CarDynamics(model, CONFIG)
        commands = dynamics.actuator_commands(
            ControlVector(steer=0.4, throttle=0.0, brake=0.0), data
        )
        assert commands["steer_fl"] == commands["steer_fr"]

    def test_drive_is_rear_wheels_only(self, model, data):
        dynamics = CarDynamics(model, CONFIG)
        commands = dynamics.actuator_commands(
            ControlVector(steer=0.0, throttle=1.0, brake=0.0), data
        )
        assert commands["drive_rl"] > 0 and commands["drive_rr"] > 0
        assert "drive_fl" not in commands

    def test_starts_in_first_gear_and_resets_to_it(self, model, data):
        dynamics = CarDynamics(model, CONFIG)
        assert dynamics.gear == 0
        _drive(model, data, 0.0, 1.0, 0.0, 4.0)
        dynamics.actuator_commands(ControlVector.neutral(), data)
        dynamics.reset()
        assert dynamics.gear == 0

    def test_rejects_a_model_without_the_car(self):
        empty = mujoco.MjModel.from_xml_string(
            "<mujoco><worldbody><geom type='plane' size='1 1 1'/></worldbody></mujoco>"
        )
        with pytest.raises(ValueError, match="car"):
            CarDynamics(empty, CONFIG)

    def test_rejects_a_non_positive_substep_count(self, model, data):
        dynamics = CarDynamics(model, CONFIG)
        with pytest.raises(ValueError, match="n_substeps"):
            dynamics.step(ControlVector.neutral(), data, 0)


class TestAerodynamicsAreApplied:
    def test_no_aero_force_at_rest(self, model, data):
        dynamics = CarDynamics(model, CONFIG)
        dynamics.apply_aero(data)
        assert float(np.linalg.norm(data.xfrc_applied[_body_id(model), :3])) < 1.0

    def test_downforce_pushes_the_car_down_at_speed(self, model, data):
        dynamics = CarDynamics(model, CONFIG)
        _drive(model, data, 0.0, 1.0, 0.0, 6.0)
        dynamics.apply_aero(data)
        assert data.xfrc_applied[_body_id(model), 2] < -1000.0

    def test_drag_opposes_forward_motion(self, model, data):
        dynamics = CarDynamics(model, CONFIG)
        _drive(model, data, 0.0, 1.0, 0.0, 6.0)
        dynamics.apply_aero(data)
        forward = data.xmat[_body_id(model)].reshape(3, 3)[:2, 0]
        assert float(np.dot(data.xfrc_applied[_body_id(model), :2], forward)) < 0

    def test_aero_force_grows_with_speed(self, model, data):
        dynamics = CarDynamics(model, CONFIG)
        _drive(model, data, 0.0, 1.0, 0.0, 3.0)
        dynamics.apply_aero(data)
        slow = abs(float(data.xfrc_applied[_body_id(model), 2]))
        _drive(model, data, 0.0, 1.0, 0.0, 6.0)
        dynamics.apply_aero(data)
        assert abs(float(data.xfrc_applied[_body_id(model), 2])) > slow


class TestLongitudinalDynamics:
    def test_the_car_does_not_move_on_its_own(self, model, data):
        start = data.xpos[_body_id(model)][:2].copy()
        _drive(model, data, 0.0, 0.0, 0.0, 3.0)
        assert float(np.linalg.norm(data.xpos[_body_id(model)][:2] - start)) < 0.1

    def test_throttle_moves_the_car_forward(self, model, data):
        """Catches a car that compiles, runs, and silently does nothing."""
        start = data.xpos[_body_id(model)][:2].copy()
        _drive(model, data, 0.0, 1.0, 0.0, 3.0)
        displacement = data.xpos[_body_id(model)][:2] - start
        forward = data.xmat[_body_id(model)].reshape(3, 3)[:2, 0]
        assert float(np.linalg.norm(displacement)) > 10.0
        assert float(np.dot(displacement, forward)) > 0  # forward, not backward

    def test_reaches_a_plausible_speed(self, model, data):
        _drive(model, data, 0.0, 1.0, 0.0, 3.0)
        speed = _speed(model, data)
        assert 5.0 < speed < 100.0, f"{speed * 3.6:.0f} km/h after 3 s is not a racing car"

    def test_more_throttle_gives_more_speed(self, model, data):
        _drive(model, data, 0.0, 0.4, 0.0, 3.0)
        slow = _speed(model, data)
        mujoco.mj_resetData(model, data)
        _drive(model, data, 0.0, 1.0, 0.0, 3.0)
        assert _speed(model, data) > slow

    def test_brake_stops_the_car(self, model, data):
        _drive(model, data, 0.0, 1.0, 0.0, 3.0)
        moving = _speed(model, data)
        assert moving > 5.0
        _drive(model, data, 0.0, 0.0, 1.0, 3.0)
        assert _speed(model, data) < 0.5

    def test_brake_never_reverses_the_car(self, model, data):
        """Dampers oppose motion by construction; a hand-rolled opposing torque would
        chatter through zero and push the car backwards."""
        _drive(model, data, 0.0, 1.0, 0.0, 2.0)
        forward = data.xmat[_body_id(model)].reshape(3, 3)[:2, 0].copy()
        before = data.xpos[_body_id(model)][:2].copy()
        _drive(model, data, 0.0, 0.0, 1.0, 4.0)
        travelled = float(np.dot(data.xpos[_body_id(model)][:2] - before, forward))
        assert travelled > -0.05


class TestSteering:
    """Left/right must not be swapped. Nothing downstream can detect it."""

    def _yaw_rate_under_steer(self, model, data, steer) -> float:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        _drive(model, data, 0.0, 0.6, 0.0, 2.0)  # get rolling first
        _drive(model, data, steer, 0.6, 0.0, 2.0)
        return _yaw_rate(model, data)

    def test_left_steer_turns_left(self, model, data):
        assert self._yaw_rate_under_steer(model, data, -0.5) > 0.05

    def test_right_steer_turns_right(self, model, data):
        assert self._yaw_rate_under_steer(model, data, 0.5) < -0.05

    def test_centred_steering_goes_straight(self, model, data):
        assert abs(self._yaw_rate_under_steer(model, data, 0.0)) < 0.05

    def test_more_steering_turns_harder_in_the_linear_region(self, model, data):
        """Only checked below tyre saturation. Past about a fifth of lock the front tyres
        are at their limit and the car understeers, so more steering gives *less* yaw --
        measured 0.42 rad/s at 0.2 lock but 0.27 at 0.4. That is real behaviour for a car
        with this much grip, not a defect, so the monotonic check stays in the region where
        monotonicity is actually expected."""
        small = abs(self._yaw_rate_under_steer(model, data, -0.05))
        medium = abs(self._yaw_rate_under_steer(model, data, -0.1))
        large = abs(self._yaw_rate_under_steer(model, data, -0.2))
        assert large > medium > small


class TestStability:
    def test_car_stays_upright_under_hard_cornering(self, model, data):
        _drive(model, data, 0.0, 1.0, 0.0, 2.0)
        _drive(model, data, -1.0, 1.0, 0.0, 3.0)
        up = data.xmat[_body_id(model)].reshape(3, 3)[:, 2]
        assert float(up[2]) > 0.7, "the car rolled over"

    def test_no_nan_over_a_long_rollout(self, model, data):
        """Numerical-stability guard (`AGENTS.md` §11)."""
        rng = np.random.default_rng(0)
        for _ in range(40):
            _drive(model, data, float(rng.uniform(-1, 1)), float(rng.uniform(0, 1)), 0.0, 0.25)
            assert np.all(np.isfinite(data.qpos))
            assert np.all(np.isfinite(data.qvel))

    def test_ride_height_is_stable_when_stationary(self, model, data):
        _drive(model, data, 0.0, 0.0, 0.0, 2.0)
        assert data.xpos[_body_id(model)][2] == pytest.approx(CONFIG.ride_height_m, abs=0.05)


class TestSteeringDamping:
    def test_kingpin_is_near_critically_damped(self, model):
        """The steering damper. At the original 2.0 N m s/rad the kingpin's damping ratio
        was 0.011 -- a 20 Hz mode with nothing holding it -- and under braking with a
        touch of steering the front wheels rang lock-to-lock, 28 degrees peak to peak.

        Computed from the compiled model, not the config, so a heavier wheel or a stiffer
        actuator cannot quietly bring the shimmy back."""
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steer_fl")
        dof = model.jnt_dofadr[joint]
        actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "steer_fl")
        inertia = float(model.dof_M0[dof])
        stiffness = float(model.actuator_gainprm[actuator][0])
        damping = float(model.dof_damping[dof])
        ratio = damping / (2.0 * np.sqrt(stiffness * inertia))
        assert 0.5 <= ratio <= 1.2, f"kingpin damping ratio {ratio:.3f}"

    def test_steering_still_responds_quickly(self, model, data):
        """Damping must not make the wheels lag the command. Full lock in under 0.1 s."""
        dynamics = CarDynamics(model, CONFIG)
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "steer_fl")
        address = model.jnt_qposadr[joint]
        dynamics.step(
            ControlVector(steer=1.0, throttle=0.0, brake=0.0), data, int(0.1 / model.opt.timestep)
        )
        assert abs(float(data.qpos[address])) > 0.9 * CONFIG.max_steer_rad


class TestSuspension:
    """Coilovers on all four corners plus an anti-roll bar per axle."""

    def test_every_corner_has_a_slide_joint(self, model):
        for side in ("fl", "fr", "rl", "rr"):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"susp_{side}")
            assert joint != -1
            assert model.jnt_type[joint] == mujoco.mjtJoint.mjJNT_SLIDE

    def test_each_axle_has_its_own_anti_roll_bar(self, model):
        """Front and rear bars are separately adjustable, as on a real car. In this model
        the split moves balance very little -- measured slip angles barely changed from
        50/50 to 20/80 -- because MuJoCo's friction is exactly proportional to load, so
        moving load between wheels does not change an axle's total grip. It becomes a
        real setup knob once a load-sensitive tyre model lands with AC's data."""
        for axle in ("f", "r"):
            tendon = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, f"arb_{axle}")
            assert tendon != -1
        front = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "arb_f")
        rear = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, "arb_r")
        assert model.tendon_stiffness[front] == pytest.approx(CONFIG.anti_roll_stiffness_front_n_m)
        assert model.tendon_stiffness[rear] == pytest.approx(CONFIG.anti_roll_stiffness_rear_n_m)

    def test_the_front_springs_are_the_stiffer_pair(self, model):
        """This used to assert the opposite, on the general principle that the driven
        axle carries more weight so wants more spring. The SF70H's own data says the
        front is stiffer -- 40 kN/m per wheel against 30 at the rear, and 100 against 90
        once the heave springs are folded in. Stated as a fact about this car, not a rule
        about cars."""
        front = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "susp_fl")
        rear = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "susp_rl")
        assert model.jnt_stiffness[front] > model.jnt_stiffness[rear] > 0

    def test_anti_roll_bars_exist_per_axle(self, model):
        for axle in ("f", "r"):
            assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, f"arb_{axle}") != -1

    def test_static_sag_is_small_and_equal_left_to_right(self, model, data):
        """Unequal sag means an asymmetric car, which would pull under braking.

        The window used to be 3-15 mm, from springs guessed at 250/300 kN/m. The real car
        is far softer than that -- 100/90 kN/m once AC's heave springs are folded in --
        and sits 16 mm down at the front and 22 at the rear on its own weight. That is the
        point of a soft spring and a stiff heave spring: compliant over kerbs, rigid under
        aero load."""
        _drive(model, data, 0.0, 0.0, 0.0, 2.0)
        # The slide axis is +z on the hub, so under load the chassis drops relative to
        # the hub and the joint coordinate goes *positive*: compression = +q.
        sag = {}
        for side in ("fl", "fr", "rl", "rr"):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"susp_{side}")
            sag[side] = float(data.qpos[model.jnt_qposadr[joint]])
        assert all(0.010 < value < 0.030 for value in sag.values()), sag
        assert sag["fl"] == pytest.approx(sag["fr"], abs=5e-4)
        assert sag["rl"] == pytest.approx(sag["rr"], abs=5e-4)
        # Softer rear springs under a rear-biased weight distribution: the car sits
        # nose-up at rest, which is the rake a real one runs.
        assert sag["rl"] > sag["fl"]

    def test_car_rests_level(self, model, data):
        _drive(model, data, 0.0, 0.0, 0.0, 2.0)
        rotation = data.xmat[_body_id(model)].reshape(3, 3)
        assert abs(np.degrees(np.arcsin(-rotation[2, 0]))) < 0.3  # pitch
        assert abs(np.degrees(np.arctan2(rotation[2, 1], rotation[2, 2]))) < 0.3  # roll

    def test_wheel_hop_is_overdamped(self, model):
        """The unsprung corner is ~21 kg on a very stiff spring, a ~17 Hz mode; the damper
        must be well past critical for it or the wheels bounce on every contact impulse."""
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "susp_fl")
        dof = model.jnt_dofadr[joint]
        inertia = float(model.dof_M0[dof])
        ratio = float(model.dof_damping[dof]) / (
            2.0 * np.sqrt(model.jnt_stiffness[joint] * inertia)
        )
        assert ratio > 1.0, f"wheel-hop damping ratio {ratio:.2f}"


class TestCarConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mass_kg": 0.0},
            {"wheelbase_m": -1.0},
            {"wheel_radius_m": 0.0},
            {"max_actuator_torque_nm": 0.0},
            {"hub_mass_kg": 0.0},
            {"camera_fovy_deg": 0.0},
            {"camera_fovy_deg": 180.0},
            {"max_steer_rad": 2.0},
            {"anti_roll_stiffness_front_n_m": -1.0},
            {"anti_roll_stiffness_rear_n_m": -1.0},
        ],
    )
    def test_rejects_bad_values(self, kwargs):
        with pytest.raises(ValueError):
            CarConfig(**kwargs)

    def test_ride_height_matches_wheel_radius(self):
        assert CarConfig(wheel_radius_m=0.4).ride_height_m == pytest.approx(0.4)


class TestMatchesSF70HSpecification:
    """The config claims to describe a Ferrari SF70H. Check it against the published
    figures in SF70H_REFERENCE rather than trusting the docstring."""

    def test_mass_is_the_2017_minimum(self):
        assert CONFIG.mass_kg == pytest.approx(SF70H_REFERENCE["mass_kg"])

    def test_wheelbase(self):
        assert CONFIG.wheelbase_m == pytest.approx(SF70H_REFERENCE["wheelbase_m"])

    def test_wheel_diameter(self):
        assert CONFIG.wheel_radius_m * 2 == pytest.approx(SF70H_REFERENCE["wheel_diameter_m"])

    def test_overall_width_is_within_the_regulation(self):
        """2017 regs capped overall width at 2000 mm, measured across the wheels."""
        for track, tyre in (
            (CONFIG.track_width_front_m, CONFIG.wheel_width_front_m),
            (CONFIG.track_width_rear_m, CONFIG.wheel_width_rear_m),
        ):
            assert track + tyre <= SF70H_REFERENCE["overall_width_m"] + 1e-9

    def test_rears_are_wider_than_fronts(self):
        assert CONFIG.wheel_width_rear_m > CONFIG.wheel_width_front_m

    def test_weight_distribution_is_rear_biased_and_legal(self):
        """2017 regulations put a floor of 44% on the front axle."""
        assert 0.44 <= CONFIG.front_weight_fraction < 0.50

    def test_steering_lock_is_realistic(self):
        """F1 runs roughly 20 degrees at the road wheel, not a road car's 35+."""
        assert 15.0 <= np.degrees(CONFIG.max_steer_rad) <= 28.0

    def test_tyre_friction_is_in_the_published_slick_range(self):
        """The ceiling was 1.8, from published slick figures. It is 1.9 because this car
        is fitted to Assetto Corsa rather than to those figures, and AC's UltraSoft gives
        DY_REF 1.88 on both axles -- so a pair averaging 1.875 sits on AC's number while
        1.8 and below sits under it. Above 1.9 the pair would average past AC's."""
        assert 1.5 <= CONFIG.wheel_friction[0] <= 1.9

    def test_the_rear_tyres_grip_harder_than_the_fronts(self):
        """Payton asked for oversteer to stop being a constant problem. With one mu
        everywhere the axles are equally grippy while the rears also carry the drive
        torque, so the car spun on the throttle. The wider rear tyre is expressed as
        more friction because MuJoCo has no tyre load sensitivity."""
        assert CONFIG.wheel_friction_rear[0] > CONFIG.wheel_friction[0]

    def test_the_rear_grip_advantage_is_small(self):
        """Too much and the car cannot be made to oversteer at all, which is not what
        was asked for. 1.85 already took away the last provokable slide."""
        assert CONFIG.wheel_friction_rear[0] / CONFIG.wheel_friction[0] <= 1.10

    def test_rear_tyre_friction_is_still_in_the_published_slick_range(self):
        assert 1.5 <= CONFIG.wheel_friction_rear[0] <= 2.0

    def test_the_axles_average_near_assetto_corsas_reference_grip(self):
        """What actually pins the pair. AC gives DY_REF 1.88 front and rear and carries
        the axle difference in load sensitivity, which MuJoCo cannot express -- so the
        spread stands in for it and the average is the part that has to stay honest."""
        average = (CONFIG.wheel_friction[0] + CONFIG.wheel_friction_rear[0]) / 2.0
        assert average == pytest.approx(1.88, abs=0.03)


class TestMassProperties:
    def test_centre_of_gravity_is_behind_the_midpoint(self):
        """Rear weight bias means the CoG sits behind the wheelbase centre."""
        assert CONFIG.centre_of_gravity_x_m < 0

    def test_centre_of_gravity_is_below_the_axle_line(self):
        assert CONFIG.centre_of_gravity_offset_z_m < 0

    def test_model_mass_matches_the_configured_mass(self, model):
        """Chassis geom carries mass=0 and the explicit <inertial> holds it all, so a
        mistake here would silently double-count the car's weight."""
        total = float(model.body_mass.sum())
        wheels_and_hubs = 4 * CONFIG.wheel_mass_kg + 4 * CONFIG.hub_mass_kg
        assert total == pytest.approx(CONFIG.mass_kg + wheels_and_hubs, rel=0.01)

    def test_yaw_inertia_is_not_the_box_default(self, model):
        """MuJoCo would derive ~1500 kg m^2 from a uniform 5 m box against a real car's
        ~750. Halved yaw response is invisible in any test that does not look for it."""
        car = _body_id(model)
        assert float(model.body_inertia[car][2]) == pytest.approx(CONFIG.inertia_yaw_kgm2, rel=0.05)

    def test_static_front_axle_load_matches_the_weight_split(self, model, data):
        """Settle the car and check the contact forces land where the config says."""
        _drive(model, data, 0.0, 0.0, 0.0, 2.0)
        loads = {"f": 0.0, "r": 0.0}
        for i in range(data.ncon):
            contact = data.contact[i]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
            if not name.startswith("wheel_"):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
            if not name.startswith("wheel_"):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            loads[name[6]] += abs(float(force[0]))
        total = loads["f"] + loads["r"]
        assert total > 0, "car is not resting on its wheels"
        assert loads["f"] / total == pytest.approx(CONFIG.front_weight_fraction, abs=0.06)
