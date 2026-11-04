"""The optional limit on how fast the steering can move (GH-104)."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fly_driver.envs.car import CarConfig, CarDynamics, assemble_model_xml, steering_angle_rad
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.envs.scene import SceneConfig
from fly_driver.interface import ControlVector

#: Lock is 0.3142 rad, so 0.5 rad/s takes 0.6284 s to reach it from centre.
RATE_RAD_S = 0.5
LIMITED = CarConfig(max_steer_rate_rad_s=RATE_RAD_S)


def _steer(steer: float) -> ControlVector:
    return ControlVector(steer=steer, throttle=0.0, brake=0.0)


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    points = [(0.0, 0.0), (400.0, 0.0), (400.0, 200.0), (0.0, 200.0)]
    track = Centerline(points=points, half_width_right=[8.0] * 4, half_width_left=[8.0] * 4)
    return mujoco.MjModel.from_xml_string(
        assemble_model_xml(track, SceneConfig(mesh_spacing_m=50.0), LIMITED)
    )


@pytest.fixture
def data(model: mujoco.MjModel) -> mujoco.MjData:
    state = mujoco.MjData(model)
    mujoco.mj_forward(model, state)
    return state


def test_with_no_limit_the_angle_follows_the_command_at_once(model, data) -> None:
    dynamics = CarDynamics(model, CarConfig())
    commands = dynamics.actuator_commands(_steer(1.0), data)
    assert commands["steer_fl"] == steering_angle_rad(1.0)
    commands = dynamics.actuator_commands(_steer(-1.0), data)
    assert commands["steer_fl"] == steering_angle_rad(-1.0)


def test_each_substep_moves_at_most_one_substep_of_the_rate(model, data) -> None:
    dynamics = CarDynamics(model, LIMITED)
    step = RATE_RAD_S * model.opt.timestep
    angles = [dynamics.actuator_commands(_steer(1.0), data)["steer_fl"] for _ in range(3)]
    # Steering right is a negative kingpin angle, so it moves down from zero.
    assert angles == pytest.approx([-step, -2 * step, -3 * step])


def test_both_front_wheels_stay_together_under_the_limit(model, data) -> None:
    commands = CarDynamics(model, LIMITED).actuator_commands(_steer(1.0), data)
    assert commands["steer_fl"] == commands["steer_fr"]


def test_it_stops_at_the_command_without_overshooting(model, data) -> None:
    dynamics = CarDynamics(model, LIMITED)
    target = steering_angle_rad(0.1)
    steps_needed = int(np.ceil(abs(target) / (RATE_RAD_S * model.opt.timestep)))
    for _ in range(steps_needed + 5):
        angle = dynamics.actuator_commands(_steer(0.1), data)["steer_fl"]
    assert angle == pytest.approx(target)


def test_lock_to_lock_takes_the_hand_computed_time(model, data) -> None:
    dynamics = CarDynamics(model, LIMITED)
    lock = LIMITED.max_steer_rad
    to_lock = int(np.ceil(lock / (RATE_RAD_S * model.opt.timestep)))
    for _ in range(to_lock):
        dynamics.actuator_commands(_steer(1.0), data)
    # Across the full 2 * lock, at the rate: one substep short of it hasn't arrived.
    across = int(np.ceil(2 * lock / (RATE_RAD_S * model.opt.timestep)))
    for _ in range(across - 1):
        angle = dynamics.actuator_commands(_steer(-1.0), data)["steer_fl"]
    assert angle < lock
    assert dynamics.actuator_commands(_steer(-1.0), data)["steer_fl"] == pytest.approx(lock)


def test_reset_recentres_the_steering(model, data) -> None:
    dynamics = CarDynamics(model, LIMITED)
    for _ in range(50):
        dynamics.actuator_commands(_steer(1.0), data)
    dynamics.reset()
    step = RATE_RAD_S * model.opt.timestep
    assert dynamics.actuator_commands(_steer(1.0), data)["steer_fl"] == pytest.approx(-step)


@pytest.mark.parametrize("rate", [0.0, -1.0])
def test_a_rate_that_isnt_positive_is_refused(rate: float) -> None:
    with pytest.raises(ValueError, match="max_steer_rate_rad_s"):
        CarConfig(max_steer_rate_rad_s=rate)


def test_the_default_car_has_no_limit() -> None:
    assert CarConfig().max_steer_rate_rad_s is None


def _flip_kingpin(car: CarConfig | None, frames: int = 25) -> list[float]:
    """Left front kingpin angle over a lock-to-lock flip every frame."""
    env = PracticeTrack(terminate_off_track=False, car=car)
    try:
        try:
            env.reset(seed=0)
        except Exception as exc:  # pragma: no cover - depends on the machine
            env.close()
            pytest.skip(f"no offscreen GL context: {exc}")
        joint = env.model.joint("steer_fl").qposadr[0]
        angles = []
        for frame in range(frames):
            env.step(_steer(1.0 if frame % 2 == 0 else -1.0))
            angles.append(float(env.data.qpos[joint]))
        return angles
    finally:
        env.close()


def test_the_limit_stops_a_flip_every_frame_reaching_the_wheels() -> None:
    free = max(abs(angle) for angle in _flip_kingpin(None))
    limited = max(abs(angle) for angle in _flip_kingpin(CarConfig(max_steer_rate_rad_s=1.26)))
    # 1.26 rad/s moves 0.0252 rad in a 20 ms frame, about 8% of lock.
    assert free > 0.1
    assert limited < 0.05
