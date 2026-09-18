"""Tests for `TrackballLegBody` (GH-21). Same split as `test_flybody_wing.py`."""

from __future__ import annotations

import sys

import numpy as np
import pytest

import fly_driver.body
from fly_driver.body._flybody_common import FlybodyNotInstalledError, _action_index_map
from fly_driver.body.flybody_legs import (
    _FEMUR_A,
    _FEMUR_B,
    _TRIPOD_A,
    _TRIPOD_B,
    TrackballLegBody,
)
from fly_driver.interface import ControlVector


class _FakeActionSpec:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.name = "\t".join(names)
        self.shape = (len(names),)


class _FakeTimeStep:
    def __init__(self, observation: dict[str, np.ndarray]) -> None:
        self.observation = observation


class _FakeBallEnv:
    """A `Body`-testable stand-in for `flybody.fly_envs.walk_on_ball(disable_wings=True)`."""

    _ACTION_NAMES = (*_TRIPOD_A, *_TRIPOD_B, *_FEMUR_A, *_FEMUR_B)

    def __init__(self, ball_qvel: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.last_action: np.ndarray | None = None
        self._ball_qvel = np.array(ball_qvel)

    def action_spec(self) -> _FakeActionSpec:
        return _FakeActionSpec(self._ACTION_NAMES)

    def reset(self) -> _FakeTimeStep:
        return self.step(np.zeros(len(self._ACTION_NAMES)))

    def step(self, action: np.ndarray) -> _FakeTimeStep:
        self.last_action = np.asarray(action, dtype=np.float64).copy()
        return _FakeTimeStep({"walker/ball_qvel": self._ball_qvel})


def test_package_exports_trackball_leg_body() -> None:
    assert fly_driver.body.TrackballLegBody is TrackballLegBody


def test_missing_flybody_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "flybody", None)
    monkeypatch.setitem(sys.modules, "flybody.fly_envs", None)

    with pytest.raises(FlybodyNotInstalledError, match="running-the-stacks"):
        TrackballLegBody().reset()


class TestActuateMapping:
    def test_zero_drive_holds_both_tripods_at_zero(self) -> None:
        env = _FakeBallEnv()
        body = TrackballLegBody(env=env)
        body.reset()

        body.actuate(ControlVector.neutral())

        assert env.last_action is not None
        index = _action_index_map(env.action_spec().name)
        for name in (*_TRIPOD_A, *_TRIPOD_B, *_FEMUR_A, *_FEMUR_B):
            assert env.last_action[index[name]] == pytest.approx(0.0)

    def test_positive_drive_puts_the_two_tripods_in_antiphase(self) -> None:
        env = _FakeBallEnv()
        body = TrackballLegBody(env=env)
        body.reset()

        body.actuate(ControlVector(steer=0.0, throttle=1.0, brake=0.0))

        index = _action_index_map(env.action_spec().name)
        assert env.last_action is not None
        a_value = env.last_action[index[_TRIPOD_A[0]]]
        b_value = env.last_action[index[_TRIPOD_B[0]]]
        # Both tripods share one leg name (coxa_T1_left in A, coxa_T1_right in B) driven
        # oppositely: the alternating-tripod gait this module documents.
        assert a_value == pytest.approx(-b_value)
        for name in _TRIPOD_A:
            assert env.last_action[index[name]] == pytest.approx(a_value)
        for name in _TRIPOD_B:
            assert env.last_action[index[name]] == pytest.approx(b_value)

    def test_femur_swing_is_half_coxa_swing(self) -> None:
        env = _FakeBallEnv()
        body = TrackballLegBody(env=env)
        body.reset()

        body.actuate(ControlVector(steer=0.0, throttle=1.0, brake=0.0))

        index = _action_index_map(env.action_spec().name)
        assert env.last_action is not None
        coxa_value = env.last_action[index[_TRIPOD_A[0]]]
        femur_value = env.last_action[index[_FEMUR_A[0]]]
        assert femur_value == pytest.approx(0.5 * coxa_value)

    def test_brake_opposes_throttle_in_the_drive_signal(self) -> None:
        # Phase starts at 0 on reset, so sin(2*pi*phase) is 0 on the very first step
        # regardless of drive -- step twice so the comparison is past that zero-crossing.
        index = _action_index_map(_FakeBallEnv().action_spec().name)

        forward_env = _FakeBallEnv()
        forward_body = TrackballLegBody(env=forward_env)
        forward_body.reset()
        for _ in range(2):
            forward_body.actuate(ControlVector(steer=0.0, throttle=1.0, brake=0.0))

        braked_env = _FakeBallEnv()
        braked_body = TrackballLegBody(env=braked_env)
        braked_body.reset()
        for _ in range(2):
            braked_body.actuate(ControlVector(steer=0.0, throttle=0.2, brake=1.0))

        assert forward_env.last_action is not None
        assert braked_env.last_action is not None
        forward_value = forward_env.last_action[index[_TRIPOD_A[0]]]
        braked_value = braked_env.last_action[index[_TRIPOD_A[0]]]
        assert forward_value != 0.0
        # throttle=1,brake=0 -> drive=+1; throttle=0.2,brake=1 -> drive=-0.8: opposite sign.
        assert forward_value * braked_value < 0.0


class TestActuateReadout:
    def test_steer_passes_through_unchanged(self) -> None:
        env = _FakeBallEnv()
        body = TrackballLegBody(env=env)
        body.reset()

        realised = body.actuate(ControlVector(steer=0.37, throttle=0.0, brake=0.0))

        assert realised.steer == pytest.approx(0.37)

    def test_throttle_reads_from_ball_speed_scaled_by_max_speed(self) -> None:
        env = _FakeBallEnv(ball_qvel=(3.0, 0.0, 4.0))  # norm = 5.0
        body = TrackballLegBody(env=env, max_ball_speed=10.0)
        body.reset()

        realised = body.actuate(ControlVector.neutral())

        assert realised.throttle == pytest.approx(0.5)

    def test_ball_speed_beyond_max_is_clipped_not_raised(self) -> None:
        env = _FakeBallEnv(ball_qvel=(100.0, 0.0, 0.0))
        body = TrackballLegBody(env=env, max_ball_speed=1.0)
        body.reset()

        realised = body.actuate(ControlVector.neutral())

        assert realised.throttle == 1.0

    def test_brake_is_always_zero(self) -> None:
        env = _FakeBallEnv()
        body = TrackballLegBody(env=env)
        body.reset()

        realised = body.actuate(ControlVector(steer=0.0, throttle=0.0, brake=1.0))

        assert realised.brake == 0.0


def test_real_flybody_ball_env_has_the_names_this_module_assumes() -> None:
    """The one test that touches real flybody: assumptions checked against upstream.

    Also confirms a driven gait actually moves the ball -- checked by running one, not
    assumed from the leg names alone.
    """
    pytest.importorskip("flybody")
    import flybody.fly_envs as fly_envs

    env = fly_envs.walk_on_ball(disable_wings=True)
    index = _action_index_map(env.action_spec().name)
    for name in (*_TRIPOD_A, *_TRIPOD_B, *_FEMUR_A, *_FEMUR_B):
        assert name in index

    body = TrackballLegBody(env=env)
    body.reset()
    saw_nonzero_ball_speed = False
    for _ in range(60):
        realised = body.actuate(ControlVector(steer=0.0, throttle=1.0, brake=0.0))
        if realised.throttle > 0.0:
            saw_nonzero_ball_speed = True
    assert saw_nonzero_ball_speed
