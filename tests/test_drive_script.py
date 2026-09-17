"""Tests for the manual driving tool's input handling (GH-16).

``scripts/drive.py`` is a dev tool, but its control state is real logic: press-only key
events plus per-step decay, and a single signed pedal axis split into throttle and brake.
Getting the decay wrong makes the car feel undriveable, and "it feels wrong" is a
miserable thing to debug through a GUI.

The binding choice is also pinned here. The viewer owns every letter and digit key --
``W`` toggles wireframe, digits toggle geom groups, ``SPACE`` pauses the physics -- and it
handles them *as well as* passing them to our callback. A first version bound throttle to
``W`` and the brake to ``SPACE``, so braking silently paused the simulation. Arrows only.

Loaded by path because ``scripts/`` is not an installed package.
"""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "drive_script", Path(__file__).resolve().parents[1] / "scripts" / "drive.py"
)
assert _SPEC and _SPEC.loader
drive = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drive)

ARROWS = (drive.KEY_UP, drive.KEY_DOWN, drive.KEY_LEFT, drive.KEY_RIGHT)


@pytest.fixture
def state():
    return drive.DriverState()


class TestBindingsAvoidViewerCollisions:
    """The viewer wins on any key it already owns, so we must not use those."""

    def test_only_arrow_keys_are_bound(self):
        bound = {
            name: value
            for name, value in vars(drive).items()
            if name.startswith("KEY_") and isinstance(value, int)
        }
        assert set(bound.values()) == set(ARROWS), f"non-arrow binding present: {bound}"

    def test_no_letter_or_digit_keys(self):
        """Letters toggle render flags, digits toggle geom-group visibility."""
        for name, value in vars(drive).items():
            if name.startswith("KEY_") and isinstance(value, int):
                assert not (48 <= value <= 57), f"{name} is a digit key"
                assert not (65 <= value <= 90), f"{name} is a letter key"

    def test_space_is_not_bound(self, state):
        """SPACE is the viewer's pause/resume. Braking must not pause the sim."""
        state.on_key(32)
        assert state.brake == 0.0
        assert state.throttle == 0.0


class TestPedalAxis:
    def test_starts_coasting(self, state):
        assert state.throttle == 0.0 and state.brake == 0.0

    def test_up_adds_throttle(self, state):
        state.on_key(drive.KEY_UP)
        assert state.throttle == pytest.approx(drive.PEDAL_STEP)
        assert state.brake == 0.0

    def test_down_from_neutral_brakes(self, state):
        state.on_key(drive.KEY_DOWN)
        assert state.brake == pytest.approx(drive.PEDAL_STEP)
        assert state.throttle == 0.0

    def test_down_backs_off_throttle_before_braking(self, state):
        """One axis: you have to come off the throttle before the brakes bite."""
        for _ in range(4):
            state.on_key(drive.KEY_UP)
        assert state.throttle > 0
        for _ in range(4):
            state.on_key(drive.KEY_DOWN)
        assert state.throttle == pytest.approx(0.0)
        assert state.brake == pytest.approx(0.0)
        state.on_key(drive.KEY_DOWN)
        assert state.brake > 0

    def test_throttle_and_brake_are_never_both_applied(self, state):
        rng = random.Random(1)
        for _ in range(300):
            state.on_key(rng.choice([drive.KEY_UP, drive.KEY_DOWN]))
            assert state.throttle == 0.0 or state.brake == 0.0

    def test_pedal_holds_like_cruise_control(self, state):
        """Press-only input means it must persist, or the car would move for one frame."""
        state.on_key(drive.KEY_UP)
        for _ in range(50):
            state.settle()
        assert state.throttle == pytest.approx(drive.PEDAL_STEP)

    def test_clamps_at_full_throttle(self, state):
        for _ in range(50):
            state.on_key(drive.KEY_UP)
        assert state.throttle == 1.0

    def test_clamps_at_full_brake(self, state):
        for _ in range(50):
            state.on_key(drive.KEY_DOWN)
        assert state.brake == 1.0


class TestSteering:
    def test_left_arrow_steers_left_negative(self, state):
        state.on_key(drive.KEY_LEFT)
        assert state.steer < 0

    def test_right_arrow_steers_right_positive(self, state):
        state.on_key(drive.KEY_RIGHT)
        assert state.steer > 0

    def test_steering_recentres_when_left_alone(self, state):
        state.on_key(drive.KEY_RIGHT)
        for _ in range(200):
            state.settle()
        assert state.steer == 0.0

    def test_steering_does_not_snap_straight_to_centre(self, state):
        """One settle should decay, not zero it -- otherwise steering is unusable."""
        state.on_key(drive.KEY_RIGHT)
        held = state.steer
        state.settle()
        assert 0 < state.steer < held

    def test_steering_clamps(self, state):
        for _ in range(50):
            state.on_key(drive.KEY_LEFT)
        assert state.steer == -1.0

    def test_unknown_keys_are_ignored(self, state):
        state.on_key(9999)
        assert state.steer == 0.0 and state.throttle == 0.0 and state.brake == 0.0


class TestControlsStayInContractRange:
    def test_any_key_sequence_produces_valid_controls(self, state):
        """Whatever the driver mashes, the result must satisfy ControlVector's ranges."""
        from fly_driver.interface import ControlVector

        rng = random.Random(0)
        for _ in range(500):
            state.on_key(rng.choice(ARROWS))
            state.settle()
            ControlVector(steer=state.steer, throttle=state.throttle, brake=state.brake)
