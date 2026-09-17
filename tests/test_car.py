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
    CarConfig,
    car_actuators_xml,
    car_assets_xml,
    car_body_xml,
    control_to_ctrl,
)
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.scene import SceneConfig, build_scene_xml

CONFIG = CarConfig()


@pytest.fixture(scope="module")
def track() -> Centerline:
    """A long straight loop, so a 3-second full-throttle run stays on it."""
    points = [(0.0, 0.0), (400.0, 0.0), (400.0, 200.0), (0.0, 200.0)]
    return Centerline(points=points, half_width_right=[8.0] * 4, half_width_left=[8.0] * 4)


@pytest.fixture(scope="module")
def model(track) -> mujoco.MjModel:
    position, yaw = track.pose_at(0.0)
    xml = build_scene_xml(
        track,
        SceneConfig(mesh_spacing_m=50.0),
        extra_assets=car_assets_xml(CONFIG),
        extra_bodies=car_body_xml(position, yaw, CONFIG),
        extra_actuators=car_actuators_xml(CONFIG),
    )
    return mujoco.MjModel.from_xml_string(xml)


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
    ids = _actuator_ids(model)
    for name, value in control_to_ctrl(steer, throttle, brake, CONFIG).items():
        data.ctrl[ids[name]] = value
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


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


class TestControlMapping:
    def test_covers_every_actuator(self):
        assert set(control_to_ctrl(0.0, 0.0, 0.0).keys()) == set(ACTUATOR_NAMES)

    def test_steer_right_is_a_negative_kingpin_angle(self):
        """Positive steer means right, and right is a clockwise (negative) rotation."""
        assert control_to_ctrl(1.0, 0.0, 0.0)["steer_fl"] < 0
        assert control_to_ctrl(-1.0, 0.0, 0.0)["steer_fl"] > 0

    def test_both_front_wheels_get_the_same_angle(self):
        commands = control_to_ctrl(0.4, 0.0, 0.0)
        assert commands["steer_fl"] == commands["steer_fr"]

    def test_full_lock_matches_the_configured_limit(self):
        assert abs(control_to_ctrl(1.0, 0.0, 0.0)["steer_fl"]) == pytest.approx(
            CONFIG.max_steer_rad
        )

    def test_drive_is_rear_wheels_only(self):
        commands = control_to_ctrl(0.0, 0.7, 0.0)
        assert commands["drive_rl"] == pytest.approx(0.7)
        assert commands["drive_rr"] == pytest.approx(0.7)
        assert "drive_fl" not in commands

    def test_brake_is_applied_to_all_four_wheels(self):
        commands = control_to_ctrl(0.0, 0.0, 0.5)
        assert all(
            commands[f"brake_{side}"] == pytest.approx(0.5) for side in ("fl", "fr", "rl", "rr")
        )


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

    def test_more_steering_turns_harder(self, model, data):
        gentle = abs(self._yaw_rate_under_steer(model, data, -0.2))
        hard = abs(self._yaw_rate_under_steer(model, data, -0.5))
        assert hard > gentle


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


class TestCarConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mass_kg": 0.0},
            {"wheelbase_m": -1.0},
            {"wheel_radius_m": 0.0},
            {"drive_gear": -5.0},
            {"brake_gain": 0.0},
            {"hub_mass_kg": 0.0},
            {"camera_fovy_deg": 0.0},
            {"camera_fovy_deg": 180.0},
            {"max_steer_rad": 2.0},
        ],
    )
    def test_rejects_bad_values(self, kwargs):
        with pytest.raises(ValueError):
            CarConfig(**kwargs)

    def test_ride_height_matches_wheel_radius(self):
        assert CarConfig(wheel_radius_m=0.4).ride_height_m == pytest.approx(0.4)
