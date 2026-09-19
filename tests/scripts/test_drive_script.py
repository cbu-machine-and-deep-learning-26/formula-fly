"""Tests for the manual driving tool's input layer (GH-16).

``scripts/drive.py`` is a dev tool, but its input mapping is real logic and it has been
wrong before: a first version put the brake on SPACE, which is the viewer's pause key, so
braking silently paused the physics; later versions stepped and decayed inputs with time
constants that were wrong in both directions.

The current design is simple enough to pin exactly. A held keyboard key is all or nothing.
A gamepad axis is analog. Both produce the same ControlVector, and the car never knows
which. The keyboard mapping is tested by setting the held-key set directly, so nothing here
needs pynput or a real controller.

Loaded by path because ``scripts/`` is not an installed package.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

from fly_driver.interface import ControlVector

_SPEC = importlib.util.spec_from_file_location(
    "drive_script", Path(__file__).resolve().parents[2] / "scripts" / "drive.py"
)
assert _SPEC and _SPEC.loader
drive = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drive)


@pytest.fixture
def keyboard():
    return drive.KeyboardInput()


class TestKeyboardBindings:
    def test_arrows_only(self):
        """The viewer owns letters and digits and handles them as well as passing them on."""
        assert set(drive.KEYS) == {"up", "down", "left", "right"}

    def test_control_does_not_need_pynput(self, keyboard):
        """The mapping is pure; only start() touches the listener."""
        assert keyboard.control(0.0) == ControlVector.neutral()


class TestKeyboardIsAllOrNothing:
    def test_nothing_held_is_neutral(self, keyboard):
        assert keyboard.control(20.0) == ControlVector.neutral()

    def test_up_is_full_throttle(self, keyboard):
        keyboard.held.add("up")
        assert keyboard.control(0.0).throttle == 1.0

    def test_releasing_up_is_zero_throttle(self, keyboard):
        """Held, not toggled: release means off. This is the change Payton asked for."""
        keyboard.held.add("up")
        keyboard.held.discard("up")
        assert keyboard.control(0.0).throttle == 0.0

    def test_down_is_full_brake(self, keyboard):
        keyboard.held.add("down")
        assert keyboard.control(0.0).brake == 1.0

    def test_releasing_down_is_zero_brake(self, keyboard):
        keyboard.held.add("down")
        keyboard.held.discard("down")
        assert keyboard.control(0.0).brake == 0.0

    def test_left_is_full_lock_left(self, keyboard):
        keyboard.held.add("left")
        assert keyboard.control(0.0).steer == -1.0

    def test_right_is_full_lock_right(self, keyboard):
        keyboard.held.add("right")
        assert keyboard.control(0.0).steer == 1.0

    def test_both_directions_cancel(self, keyboard):
        keyboard.held.update({"left", "right"})
        assert keyboard.control(0.0).steer == 0.0

    def test_throttle_and_brake_together_are_both_full(self, keyboard):
        """The contract allows it and a real driver can do it (left-foot braking)."""
        keyboard.held.update({"up", "down"})
        control = keyboard.control(0.0)
        assert control.throttle == 1.0 and control.brake == 1.0

    def test_no_intermediate_values_exist(self, keyboard):
        """Every reachable keyboard output is 0 or full on throttle and brake."""
        for combo in [set(), {"up"}, {"down"}, {"up", "down"}]:
            keyboard.held = set(combo)
            control = keyboard.control(0.0)
            assert control.throttle in (0.0, 1.0)
            assert control.brake in (0.0, 1.0)


class TestKeyboardSteeringAssist:
    def test_full_lock_at_rest(self, keyboard):
        keyboard.held.add("left")
        assert keyboard.control(0.0).steer == -1.0

    def test_scaled_down_at_speed(self, keyboard):
        keyboard.held.add("left")
        assert -0.3 < keyboard.control(300.0 / 3.6).steer < 0.0

    def test_raw_steer_ignores_speed(self):
        """What the fly gets for steer=1: literal full lock, whatever the speed."""
        keyboard = drive.KeyboardInput(raw_steer=True)
        keyboard.held.add("right")
        assert keyboard.control(300.0 / 3.6).steer == 1.0

    def test_assist_never_touches_throttle_or_brake(self, keyboard):
        keyboard.held.update({"up", "down"})
        control = keyboard.control(300.0 / 3.6)
        assert control.throttle == 1.0 and control.brake == 1.0


class TestSteeringGain:
    def test_one_at_rest(self):
        assert drive.steering_gain(0.0) == 1.0

    def test_falls_with_speed(self):
        gains = [drive.steering_gain(v) for v in (0.0, 10.0, 20.0, 40.0, 80.0)]
        assert all(later < earlier for earlier, later in zip(gains, gains[1:], strict=False))

    def test_never_below_the_floor(self):
        assert drive.steering_gain(1000.0) == drive.STEER_GAIN_FLOOR

    def test_reversing_counts_as_slow(self):
        assert drive.steering_gain(-5.0) == 1.0


class TestGamepadMapping:
    """Analog: half a trigger is half the throttle. Same ControlVector the fly produces."""

    @staticmethod
    def axes(stick_x=0.0, left_trigger=-1.0, right_trigger=-1.0):
        """Six GLFW axes with triggers released (-1) unless stated."""
        values = [0.0] * 6
        values[drive.AXIS_STEER] = stick_x
        values[drive.AXIS_LEFT_TRIGGER] = left_trigger
        values[drive.AXIS_RIGHT_TRIGGER] = right_trigger
        return values

    def test_axis_indices_match_glfw(self):
        """The mapping hardcodes indices so it is testable without GLFW; check them."""
        glfw = pytest.importorskip("glfw")
        assert drive.AXIS_STEER == glfw.GAMEPAD_AXIS_LEFT_X
        assert drive.AXIS_LEFT_TRIGGER == glfw.GAMEPAD_AXIS_LEFT_TRIGGER
        assert drive.AXIS_RIGHT_TRIGGER == glfw.GAMEPAD_AXIS_RIGHT_TRIGGER

    def test_resting_controller_is_neutral(self):
        assert drive.gamepad_axes_to_control(self.axes()) == ControlVector.neutral()

    def test_half_trigger_is_half_throttle(self):
        """GLFW triggers read -1 released to +1 pressed, so 0 is halfway."""
        assert drive.gamepad_axes_to_control(self.axes(right_trigger=0.0)).throttle == 0.5

    def test_full_trigger_is_full_throttle(self):
        assert drive.gamepad_axes_to_control(self.axes(right_trigger=1.0)).throttle == 1.0

    def test_left_trigger_is_brake(self):
        control = drive.gamepad_axes_to_control(self.axes(left_trigger=0.0))
        assert control.brake == 0.5 and control.throttle == 0.0

    def test_stick_inside_the_deadzone_is_centred(self):
        assert drive.gamepad_axes_to_control(self.axes(stick_x=0.05)).steer == 0.0

    def test_stick_is_rescaled_past_the_deadzone(self):
        """A plain cut would leave the first 8% dead and full deflection unreachable."""
        edge = drive.GAMEPAD_DEADZONE
        assert drive.gamepad_axes_to_control(self.axes(stick_x=edge)).steer == pytest.approx(0.0)
        assert drive.gamepad_axes_to_control(self.axes(stick_x=1.0)).steer == pytest.approx(1.0)
        assert drive.gamepad_axes_to_control(self.axes(stick_x=-1.0)).steer == pytest.approx(-1.0)

    def test_half_stick_is_roughly_half_lock(self):
        steer = drive.gamepad_axes_to_control(self.axes(stick_x=0.5)).steer
        assert 0.4 < steer < 0.5

    def test_steering_is_monotonic(self):
        values = [
            drive.gamepad_axes_to_control(self.axes(stick_x=x)).steer
            for x in (-1, -0.5, -0.1, 0, 0.1, 0.5, 1)
        ]
        assert values == sorted(values)

    def test_no_speed_assist_on_analog_input(self, monkeypatch):
        """A stick can be gentle on its own; the keyboard-only gain must not apply."""
        glfw = pytest.importorskip("glfw")

        class State:
            axes = self.axes(stick_x=0.5)

        monkeypatch.setattr(glfw, "get_gamepad_state", lambda jid: State())
        pad = drive.GamepadInput(0, "test")
        assert pad.control(0.0).steer == pad.control(300.0 / 3.6).steer

    def test_out_of_range_axes_are_clipped_not_rejected(self):
        control = drive.gamepad_axes_to_control(self.axes(stick_x=1.7, right_trigger=3.0))
        assert control.steer == 1.0 and control.throttle == 1.0

    def test_too_few_axes_is_an_error(self):
        with pytest.raises(ValueError, match="axes"):
            drive.gamepad_axes_to_control([0.0, 0.0])

    def test_bad_deadzone_is_an_error(self):
        with pytest.raises(ValueError, match="deadzone"):
            drive.gamepad_axes_to_control(self.axes(), deadzone=1.0)


class TestBothPathsSatisfyTheContract:
    def test_any_held_key_combination_is_valid(self, keyboard):
        rng = random.Random(0)
        for _ in range(200):
            keyboard.held = {k for k in drive.KEYS if rng.random() < 0.5}
            control = keyboard.control(rng.uniform(0.0, 100.0))
            ControlVector(steer=control.steer, throttle=control.throttle, brake=control.brake)

    def test_any_gamepad_axes_are_valid(self):
        rng = random.Random(1)
        for _ in range(200):
            axes = [rng.uniform(-1.0, 1.0) for _ in range(6)]
            control = drive.gamepad_axes_to_control(axes)
            ControlVector(steer=control.steer, throttle=control.throttle, brake=control.brake)


class TestTheViewerResetIsDetectable:
    """BACKSPACE is MuJoCo's own binding, not ours.

    It calls ``mj_resetData`` behind the driving loop, which never sees the key. The only
    signal that reaches us is the simulation clock going backwards, and everything the lap
    has accumulated -- the clocks, the penalty, the per-wheel grip -- is cleared off the
    back of it. If MuJoCo ever stopped zeroing the clock there, nothing would raise: the
    car would jump to the grid still owing whatever penalty it had built up, and the next
    lap would silently start dirty. So the assumption is pinned here rather than trusted.
    """

    def test_resetting_puts_the_clock_back_to_zero(self):
        import mujoco

        from fly_driver.envs.car import CarConfig, CarDynamics, assemble_model_xml
        from fly_driver.envs.centerline import Centerline
        from fly_driver.envs.scene import SceneConfig

        square = Centerline(
            points=[(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)],
            half_width_right=[6.0] * 4,
            half_width_left=[6.0] * 4,
        )
        model = mujoco.MjModel.from_xml_string(
            assemble_model_xml(square, SceneConfig(mesh_spacing_m=100.0), CarConfig())
        )
        data = mujoco.MjData(model)
        dynamics = CarDynamics(model)
        dynamics.step(ControlVector(steer=0.0, throttle=1.0, brake=0.0), data, 50)

        ran_to = float(data.time)
        assert ran_to > 0.0, "the clock never advanced, so the check proves nothing"

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        assert float(data.time) == 0.0
        assert float(data.time) < ran_to, "a reset is no longer detectable from the clock"
