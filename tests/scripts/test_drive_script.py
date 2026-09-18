"""Tests for the manual driving tool's input layer (GH-16).

``scripts/drive.py`` is a dev tool, but its input mapping is real logic and it has been
wrong before: a first version put the brake on SPACE, which is the viewer's pause key, so
braking silently paused the physics; later versions stepped and decayed inputs with time
constants that were wrong in both directions.

The current design is simple enough to pin exactly. A held keyboard key is all or nothing.
A gamepad axis is analog. Both produce the same ControlVector, and the car never knows
which. The keyboard mapping is tested by setting the held-key set directly, so nothing here
needs pynput or a real controller. ``--agent`` is tested without launching the viewer or
importing flyvis: the constant policy is a real ``Policy``, and loading the script must
not import torch.

Loaded by path because ``scripts/`` is not an installed package.
"""

from __future__ import annotations

import importlib.util
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fly_driver.interface import FEATURE_DTYPE, ControlVector, Policy

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


REPO = Path(__file__).resolve().parents[2]


class TestConstantPolicy:
    def test_satisfies_policy_protocol(self):
        policy = drive.ConstantPolicy(8, throttle=0.4)
        assert isinstance(policy, Policy)

    def test_holds_the_control_and_ignores_features(self):
        policy = drive.ConstantPolicy(4, steer=-0.25, throttle=0.4, brake=0.1)
        features = np.ones(4, dtype=FEATURE_DTYPE)
        assert policy.act(features) == ControlVector(steer=-0.25, throttle=0.4, brake=0.1)
        assert policy.act(np.zeros(4, dtype=FEATURE_DTYPE)) == policy.act(features)

    def test_rejects_empty_features(self):
        with pytest.raises(ValueError, match="feature_dim"):
            drive.ConstantPolicy(0)

    def test_rejects_out_of_range_throttle(self):
        with pytest.raises(ValueError, match="throttle"):
            drive.ConstantPolicy(1, throttle=1.5)

    def test_plugs_into_direct_drive_without_flyvis(self):
        """The watch policy is a real Policy; DirectDriveAgent must accept it."""
        from fly_driver.drivers import DirectDriveAgent

        class FakeEye:
            frame_shape = (2, 2, 3)
            feature_dim = 4

            def reset(self) -> None:
                return None

            def encode(self, frame: object) -> np.ndarray:
                del frame
                return np.zeros(4, dtype=FEATURE_DTYPE)

        agent = DirectDriveAgent(FakeEye(), drive.ConstantPolicy(4, throttle=0.4))
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        agent.reset()
        assert agent.act(frame) == ControlVector(steer=0.0, throttle=0.4, brake=0.0)


class TestAgentFlag:
    def test_default_is_human_keyboard(self):
        args = drive.parse_args([])
        assert args.agent is False
        assert args.input == "auto"
        assert args.throttle == 0.4

    def test_agent_defaults_to_straight_throttle(self):
        args = drive.parse_args(["--agent"])
        assert args.agent is True
        assert args.throttle == pytest.approx(0.4)

    def test_agent_rejects_input_flag(self):
        with pytest.raises(SystemExit):
            drive.parse_args(["--agent", "--input", "keyboard"])

    def test_agent_rejects_raw_steer(self):
        with pytest.raises(SystemExit):
            drive.parse_args(["--agent", "--raw-steer"])

    def test_throttle_must_be_in_range(self):
        with pytest.raises(SystemExit):
            drive.parse_args(["--agent", "--throttle", "1.2"])

    def test_help_mentions_agent(self):
        result = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "drive.py"), "--help"],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        assert result.returncode == 0, result.stderr
        assert "--agent" in result.stdout
        assert "DirectDriveAgent" in result.stdout


def test_drive_import_stays_flyvis_free() -> None:
    """Loading the script must not import torch/flyvis; only --agent does."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util, sys\n"
            "from pathlib import Path\n"
            "spec = importlib.util.spec_from_file_location(\n"
            "    'drive_script', Path('scripts/drive.py')\n"
            ")\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "assert 'flyvis' not in sys.modules\n"
            "assert 'fly_driver.eyes.flyvis_eye' not in sys.modules\n"
            "assert 'torch' not in sys.modules\n",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stdout + result.stderr
