"""Every control level, from 0 to 100%, has to survive the whole pipeline (GH-104).

A driver needs fine control of steering and pedals, not just on and off. Each stage here takes
the same 101 levels: the shared control type, the Assetto Corsa wire and action mapping, the car's
actuator commands, and the car's real response.

The last group records a physics fact that broke this for a long time and looked like a policy
problem: on the default car, small steering inputs barely move the wheels.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fly_driver.analysis.steering_wobble import _FixedStart, longest_straights
from fly_driver.bridge.assetto import to_ac_action
from fly_driver.bridge.wire import CONTROL, decode_control
from fly_driver.envs.car import CarConfig, CarDynamics, assemble_model_xml, steering_angle_rad
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.envs.scene import SceneConfig
from fly_driver.envs.segments import find_segments
from fly_driver.interface import ControlVector

PERCENT = np.linspace(0.0, 1.0, 101)
STEER = np.linspace(-1.0, 1.0, 201)

#: The steering motor that follows small inputs at every speed tested. See docs/steering_chatter.md.
STIFF_STEERING = CarConfig(steer_gain=192000.0, steer_damping_nms=800.0)


def test_the_control_type_keeps_every_level() -> None:
    for level in PERCENT:
        control = ControlVector(steer=2.0 * level - 1.0, throttle=level, brake=level)
        assert control.steer == 2.0 * level - 1.0
        assert control.throttle == level
        assert control.brake == level


def test_the_float32_array_keeps_every_level_distinct() -> None:
    throttles = {ControlVector(0.0, level, 0.0).to_array()[1] for level in PERCENT}
    steers = {ControlVector(level, 0.0, 0.0).to_array()[0] for level in STEER}
    assert len(throttles) == PERCENT.size
    assert len(steers) == STEER.size


def test_clipping_leaves_in_range_values_alone() -> None:
    for level in PERCENT:
        clipped = ControlVector.clipped(2.0 * level - 1.0, level, level)
        assert clipped == ControlVector(2.0 * level - 1.0, level, level)


def test_the_assetto_corsa_wire_carries_every_level_exactly() -> None:
    for level in PERCENT:
        record = CONTROL.pack(7, 2.0 * level - 1.0, level, level, 1.5)
        _, control, _ = decode_control(record)
        assert control == ControlVector(2.0 * level - 1.0, level, level)


def test_the_assetto_corsa_action_is_linear_in_every_channel() -> None:
    max_steer = 0.4
    for level in PERCENT:
        control = ControlVector(2.0 * level - 1.0, level, level)
        steer, pedal, brake = to_ac_action(control, max_steer)
        assert steer == pytest.approx((2.0 * level - 1.0) * max_steer)
        # AssettoCorsaGym's pedals run from -1 released to +1 full.
        assert pedal == pytest.approx(2.0 * level - 1.0)
        assert brake == pytest.approx(2.0 * level - 1.0)


@pytest.fixture(scope="module")
def small_track_model() -> mujoco.MjModel:
    points = [(0.0, 0.0), (400.0, 0.0), (400.0, 200.0), (0.0, 200.0)]
    track = Centerline(points=points, half_width_right=[8.0] * 4, half_width_left=[8.0] * 4)
    return mujoco.MjModel.from_xml_string(
        assemble_model_xml(track, SceneConfig(mesh_spacing_m=50.0), CarConfig())
    )


def _commands(model: mujoco.MjModel, control: ControlVector) -> dict[str, float]:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    data.qvel[model.joint("roll_rl").dofadr[0]] = 40.0
    data.qvel[model.joint("roll_rr").dofadr[0]] = 40.0
    data.qvel[model.joint("roll_fl").dofadr[0]] = 40.0
    data.qvel[model.joint("roll_fr").dofadr[0]] = 40.0
    return CarDynamics(model, CarConfig()).actuator_commands(control, data)


def test_steering_commands_are_linear_at_the_actuator(small_track_model) -> None:
    for level in STEER:
        commands = _commands(small_track_model, ControlVector(level, 0.0, 0.0))
        assert commands["steer_fl"] == pytest.approx(steering_angle_rad(level))
        assert commands["steer_fr"] == pytest.approx(steering_angle_rad(level))


def test_throttle_commands_scale_with_the_pedal_at_the_actuator(small_track_model) -> None:
    full = _commands(small_track_model, ControlVector(0.0, 1.0, 0.0))["drive_rl"]
    assert full > 0.0
    for level in PERCENT:
        torque = _commands(small_track_model, ControlVector(0.0, level, 0.0))["drive_rl"]
        assert torque == pytest.approx(level * full)


def test_brake_commands_scale_with_the_pedal_at_the_actuator(small_track_model) -> None:
    full = _commands(small_track_model, ControlVector(0.0, 0.0, 1.0))["brake_fl"]
    assert full < 0.0
    for level in PERCENT:
        torque = _commands(small_track_model, ControlVector(0.0, 0.0, level))["brake_fl"]
        assert torque == pytest.approx(level * full)


def test_the_motor_ceiling_never_clips_a_drive_or_brake_command(small_track_model) -> None:
    ceiling = CarConfig().max_actuator_torque_nm
    commands = _commands(small_track_model, ControlVector(0.0, 1.0, 1.0))
    assert abs(commands["drive_rl"]) < ceiling
    assert abs(commands["brake_fl"]) < ceiling


@pytest.fixture
def straight_track():
    """A factory for a track, and a reset to the start of its longest straight."""
    made: list[PracticeTrack] = []

    def build(car: CarConfig | None = None) -> tuple[PracticeTrack, object]:
        env = PracticeTrack(random_start=True, terminate_off_track=False, max_steps=20000, car=car)
        made.append(env)
        try:
            env.reset(seed=0)
        except Exception as exc:  # pragma: no cover - depends on the machine
            pytest.skip(f"no offscreen GL context: {exc}")
        return env, longest_straights(find_segments(env.centerline), 1)[0]

    yield build
    for env in made:
        env.close()


def _start(env: PracticeTrack, straight) -> dict:
    env.rng = _FixedStart(straight.start_m + 5.0)
    return env.reset()[1]


def _wheel_share(env: PracticeTrack, straight, steer: float, kmh: float) -> float:
    """Wheel angle as a share of lock, after holding ``steer`` for 0.5 s at about ``kmh``."""
    info = _start(env, straight)
    while info.get("speed_mps", 0.0) < kmh / 3.6:
        info = env.step((0.0, 1.0, 0.0))[4]
    for _ in range(25):
        throttle = 1.0 if info["speed_mps"] < kmh / 3.6 else 0.0
        info = env.step((steer, throttle, 0.0))[4]
    joint = env.model.joint("steer_fl").qposadr[0]
    return abs(float(env.data.qpos[joint])) / env.car.max_steer_rad


@pytest.mark.slow
def test_more_throttle_accelerates_harder(straight_track) -> None:
    env, straight = straight_track()
    speeds = []
    for level in (0.1, 0.25, 0.5, 0.75, 1.0):
        info = _start(env, straight)
        for _ in range(150):
            info = env.step((0.0, level, 0.0))[4]
        speeds.append(info["speed_mps"])
    assert speeds == sorted(speeds)
    assert speeds[0] > 0.0
    # Half throttle is really half: well short of full, and well past a quarter.
    assert speeds[1] < speeds[2] < 0.8 * speeds[4]


@pytest.mark.slow
def test_more_brake_stops_harder_until_the_tyres_run_out(straight_track) -> None:
    env, straight = straight_track()
    seconds = []
    for level in (0.05, 0.1, 0.25, 0.5, 1.0):
        info = _start(env, straight)
        while info.get("speed_mps", 0.0) < 200.0 / 3.6:
            info = env.step((0.0, 1.0, 0.0))[4]
        frames = 0
        while info["speed_mps"] > 100.0 / 3.6:
            info = env.step((0.0, 0.0, level))[4]
            frames += 1
        seconds.append(frames * env.dt)
    assert seconds == sorted(seconds, reverse=True)
    # Measured 6.4 s at 5% and 0.96 s at full. Above about half, the limit is the tyres' grip.
    assert seconds[0] > 3.0 * seconds[3]


@pytest.mark.slow
def test_the_default_car_has_a_steering_dead_zone_that_grows_with_speed(straight_track) -> None:
    """Records a known fault. If this starts failing the default car has changed: good news,
    but anything measured on the old car (every lab run so far) is no longer comparable."""
    env, straight = straight_track()
    assert _wheel_share(env, straight, 0.1, 108.0) < 0.02
    assert _wheel_share(env, straight, 0.2, 250.0) < 0.02
    assert _wheel_share(env, straight, 0.5, 108.0) == pytest.approx(0.5, abs=0.02)


@pytest.mark.slow
@pytest.mark.parametrize("kmh", [108.0, 200.0, 250.0])
@pytest.mark.parametrize("steer", [0.1, 0.2, 0.3, 0.5])
def test_stiff_steering_follows_the_input_at_speed(straight_track, kmh, steer) -> None:
    env, straight = straight_track(STIFF_STEERING)
    assert _wheel_share(env, straight, steer, kmh) == pytest.approx(steer, abs=0.03)


@pytest.mark.slow
def test_stiff_steering_turns_harder_the_more_you_ask(straight_track) -> None:
    env, straight = straight_track(STIFF_STEERING)
    yaw_dof = env.model.joint("car_root").dofadr[0] + 5
    rates = []
    for steer in (0.02, 0.05, 0.1, 0.2, 0.3):
        info = _start(env, straight)
        while info.get("speed_mps", 0.0) < 30.0:
            info = env.step((0.0, 1.0, 0.0))[4]
        for _ in range(25):
            throttle = 1.0 if info["speed_mps"] < 30.0 else 0.0
            info = env.step((steer, throttle, 0.0))[4]
        rates.append(abs(float(env.data.qvel[yaw_dof])))
    assert rates == sorted(rates)
    # The default car gives 0.5 deg/s at 10% here. A dead zone would fail this.
    assert np.degrees(rates[2]) > 8.0


@pytest.mark.slow
@pytest.mark.parametrize("start_kmh", [0.0, 280.0])
def test_stiff_steering_stays_stable_under_a_flip_every_frame(straight_track, start_kmh) -> None:
    env, straight = straight_track(STIFF_STEERING)
    info = _start(env, straight)
    while info.get("speed_mps", 0.0) < start_kmh / 3.6:
        info = env.step((0.0, 1.0, 0.0))[4]
    for frame in range(200):
        info = env.step((1.0 if frame % 2 else -1.0, 1.0, 0.0))[4]
        assert not info["diverged"]
        assert np.isfinite(env.data.qpos).all()
