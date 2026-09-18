"""Tests for `FlybodyWingBody` (GH-21).

Split the way `tests/eyes/test_flyvis_eye.py` splits: contract tests run without the
optional stack (flybody imports at module scope nowhere here, so nothing needs skipping
at collection time); the one test that touches real flybody skips itself via
``pytest.importorskip`` instead of a `conftest.py` collection guard, since only a single
file is involved.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import fly_driver.body
from fly_driver.body.flybody_wing import (
    _LEFT_WING_NAMES,
    _RIGHT_WING_NAMES,
    _USER_NAME,
    DEFAULT_STEER_GAIN,
    FlybodyNotInstalledError,
    FlybodyWingBody,
    _action_index_map,
)
from fly_driver.interface import FRAME_RATE_HZ, ControlVector


class _FakeActionSpec:
    """dm_control's `BoundedArray`-enough for `FlybodyWingBody`: a tab-joined `.name`."""

    def __init__(self, names: tuple[str, ...]) -> None:
        self.name = "\t".join(names)
        self.shape = (len(names),)


class _FakeTimeStep:
    def __init__(self, observation: dict[str, np.ndarray]) -> None:
        self.observation = observation


class _FakeFlyEnv:
    """A `Body`-testable stand-in for `flybody.fly_envs.vision_guided_flight()`.

    Records the action `actuate` sends so the mapping (index, sign, gain, clipping) is
    checked directly, rather than inferred from simulated physics that also depends on
    flybody's own dynamics.
    """

    _ACTION_NAMES = (
        "head_abduct",
        "head_twist",
        "head",
        *_LEFT_WING_NAMES,
        *_RIGHT_WING_NAMES,
        "abdomen_abduct",
        "abdomen",
        _USER_NAME,
    )

    def __init__(
        self,
        gyro_yaw: float | list[float] = 0.0,
        forward_speed: float = 0.0,
        control_timestep: float = 1.0 / FRAME_RATE_HZ,
    ) -> None:
        self.last_action: np.ndarray | None = None
        self.reset_count = 0
        self.step_count = 0
        # A single float repeats every call, so most tests don't have to care that
        # actuate() now takes multiple substeps per frame; a list is consumed one value
        # per call, for the tests that specifically check the substeps get averaged.
        self._gyro_yaw = gyro_yaw
        self._forward_speed = forward_speed
        self._control_timestep = control_timestep

    def control_timestep(self) -> float:
        return self._control_timestep

    def action_spec(self) -> _FakeActionSpec:
        return _FakeActionSpec(self._ACTION_NAMES)

    def reset(self) -> _FakeTimeStep:
        self.reset_count += 1
        return self.step(np.zeros(len(self._ACTION_NAMES)))

    def step(self, action: np.ndarray) -> _FakeTimeStep:
        self.last_action = np.asarray(action, dtype=np.float64).copy()
        if isinstance(self._gyro_yaw, list):
            yaw = self._gyro_yaw[min(self.step_count, len(self._gyro_yaw) - 1)]
        else:
            yaw = self._gyro_yaw
        self.step_count += 1
        return _FakeTimeStep(
            {
                "walker/gyro": np.array([0.0, 0.0, yaw]),
                "walker/velocimeter": np.array([self._forward_speed, 0.0, 0.0]),
            }
        )


def test_package_exports_body_without_importing_flybody() -> None:
    """Keep flybody a lazy dependency of the body package, like the eye's flyvis."""
    assert fly_driver.body.FlybodyWingBody is FlybodyWingBody


def test_missing_flybody_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain how to install flybody instead of failing with a bare ImportError."""
    monkeypatch.setitem(sys.modules, "flybody", None)
    monkeypatch.setitem(sys.modules, "flybody.fly_envs", None)

    with pytest.raises(FlybodyNotInstalledError, match="running-the-stacks"):
        FlybodyWingBody().reset()


def test_action_index_map_reads_tab_joined_names() -> None:
    """The lookup this module trusts instead of hardcoded positions."""
    spec_name = "a\tb\tc"
    assert _action_index_map(spec_name) == {"a": 0, "b": 1, "c": 2}


class TestActuateMapping:
    """`ControlVector` in, the exact action flybody would receive, out."""

    def test_positive_steer_biases_left_positive_right_negative(self) -> None:
        env = _FakeFlyEnv()
        body = FlybodyWingBody(env=env)
        body.reset()

        body.actuate(ControlVector(steer=1.0, throttle=0.0, brake=0.0))

        assert env.last_action is not None
        index = _action_index_map(env.action_spec().name)
        for name in _LEFT_WING_NAMES:
            assert env.last_action[index[name]] == pytest.approx(DEFAULT_STEER_GAIN)
        for name in _RIGHT_WING_NAMES:
            assert env.last_action[index[name]] == pytest.approx(-DEFAULT_STEER_GAIN)

    def test_negative_steer_flips_the_asymmetry(self) -> None:
        env = _FakeFlyEnv()
        body = FlybodyWingBody(env=env)
        body.reset()

        body.actuate(ControlVector(steer=-1.0, throttle=0.0, brake=0.0))

        index = _action_index_map(env.action_spec().name)
        assert env.last_action is not None
        assert env.last_action[index["wing_yaw_left"]] < 0.0
        assert env.last_action[index["wing_yaw_right"]] > 0.0

    def test_zero_steer_is_symmetric(self) -> None:
        env = _FakeFlyEnv()
        body = FlybodyWingBody(env=env)
        body.reset()

        body.actuate(ControlVector(steer=0.0, throttle=0.5, brake=0.0))

        index = _action_index_map(env.action_spec().name)
        assert env.last_action is not None
        for left, right in zip(_LEFT_WING_NAMES, _RIGHT_WING_NAMES, strict=True):
            assert env.last_action[index[left]] == pytest.approx(-env.last_action[index[right]])

    def test_throttle_minus_brake_drives_user_0_clipped(self) -> None:
        env = _FakeFlyEnv()
        body = FlybodyWingBody(env=env)
        body.reset()

        body.actuate(ControlVector(steer=0.0, throttle=1.0, brake=0.0))
        index = _action_index_map(env.action_spec().name)
        assert env.last_action is not None
        assert env.last_action[index[_USER_NAME]] == pytest.approx(1.0)

        body.actuate(ControlVector(steer=0.0, throttle=0.2, brake=1.0))
        assert env.last_action[index[_USER_NAME]] == pytest.approx(-0.8)

    def test_steer_gain_is_configurable(self) -> None:
        env = _FakeFlyEnv()
        body = FlybodyWingBody(steer_gain=0.7, env=env)
        body.reset()

        body.actuate(ControlVector(steer=1.0, throttle=0.0, brake=0.0))

        index = _action_index_map(env.action_spec().name)
        assert env.last_action is not None
        assert env.last_action[index["wing_yaw_left"]] == pytest.approx(0.7)


class TestActuateReadout:
    """What flybody's physics reports back becomes the realised `ControlVector`."""

    def test_readout_reflects_gyro_and_velocimeter(self) -> None:
        env = _FakeFlyEnv(gyro_yaw=0.42, forward_speed=0.6)
        body = FlybodyWingBody(max_yaw_rate=1.0, env=env)
        body.reset()

        realised = body.actuate(ControlVector.neutral())

        assert isinstance(realised, ControlVector)
        assert realised.steer == pytest.approx(0.42)
        assert realised.throttle == pytest.approx(0.6)
        assert realised.brake == 0.0

    def test_max_yaw_rate_scales_the_averaged_reading(self) -> None:
        env = _FakeFlyEnv(gyro_yaw=3.0)
        body = FlybodyWingBody(max_yaw_rate=6.0, env=env)
        body.reset()

        realised = body.actuate(ControlVector.neutral())

        assert realised.steer == pytest.approx(0.5)

    def test_forward_speed_beyond_control_range_is_clipped_not_raised(self) -> None:
        """`ControlVector.throttle` is `[0, 1]`; the readout must clip, not raise."""
        env = _FakeFlyEnv(gyro_yaw=0.0, forward_speed=1.5)
        body = FlybodyWingBody(env=env)
        body.reset()

        realised = body.actuate(ControlVector.neutral())

        assert realised.throttle == 1.0

    def test_negative_forward_speed_reads_as_zero_throttle_not_negative(self) -> None:
        """`ControlVector.throttle` is `[0, 1]`; a backward gust must not raise."""
        env = _FakeFlyEnv(gyro_yaw=0.0, forward_speed=-3.0)
        body = FlybodyWingBody(env=env)
        body.reset()

        realised = body.actuate(ControlVector.neutral())

        assert realised.throttle == 0.0

    def test_yaw_rate_beyond_control_range_is_clipped_not_raised(self) -> None:
        """`ControlVector` rejects out-of-range values; the readout must clip, not raise."""
        env = _FakeFlyEnv(gyro_yaw=50.0, forward_speed=0.0)
        body = FlybodyWingBody(max_yaw_rate=1.0, env=env)
        body.reset()

        realised = body.actuate(ControlVector.neutral())

        assert realised.steer == 1.0


class TestFrameAveraging:
    """One `actuate()` call is a whole frame, not one flybody `env.step()`."""

    def test_actuate_calls_step_once_per_substep(self) -> None:
        env = _FakeFlyEnv(gyro_yaw=0.0, control_timestep=1.0 / FRAME_RATE_HZ / 4)
        body = FlybodyWingBody(env=env)
        body.reset()  # one reset() step, counted separately below

        body.actuate(ControlVector.neutral())

        assert env.step_count == 1 + 4  # the reset's own step, then 4 substeps

    def test_readout_is_the_mean_over_the_frame_not_the_last_substep(self) -> None:
        # Index 0 is consumed by reset()'s own step; the 4 substeps then see 0, 2, 4, 6.
        env = _FakeFlyEnv(
            gyro_yaw=[0.0, 0.0, 2.0, 4.0, 6.0], control_timestep=1.0 / FRAME_RATE_HZ / 4
        )
        body = FlybodyWingBody(max_yaw_rate=1.0, env=env)
        body.reset()

        realised = body.actuate(ControlVector.neutral())

        assert realised.steer == pytest.approx(1.0)  # mean(0,2,4,6) = 3, clipped to 1.0


def test_real_flybody_action_spec_has_the_names_this_module_assumes() -> None:
    """The one test that touches real flybody: assumptions checked against upstream.

    Everything else in this file uses `_FakeFlyEnv` so the mapping logic is tested
    without needing flybody installed. This test is what stops that fake from silently
    drifting away from what `vision_guided_flight()` actually exposes.
    """
    pytest.importorskip("flybody")
    import flybody.fly_envs as fly_envs

    env = fly_envs.vision_guided_flight()
    index = _action_index_map(env.action_spec().name)

    for name in (*_LEFT_WING_NAMES, *_RIGHT_WING_NAMES, _USER_NAME):
        assert name in index

    body = FlybodyWingBody(env=env)
    body.reset()
    realised = body.actuate(ControlVector(steer=0.3, throttle=0.4, brake=0.0))
    assert isinstance(realised, ControlVector)
