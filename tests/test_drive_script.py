"""Tests for the manual driving tool's input handling (GH-16).

``scripts/drive.py`` is a dev tool, but its control state is real logic: press-only key
events plus per-step decay. Getting the decay wrong makes the car feel undriveable, and
"it feels wrong" is a miserable thing to debug through a GUI. The logic is pure, so it is
cheap to pin here.

Loaded by path because ``scripts/`` is not an installed package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "drive_script", Path(__file__).resolve().parents[1] / "scripts" / "drive.py"
)
assert _SPEC and _SPEC.loader
drive = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(drive)


@pytest.fixture
def state():
    return drive.DriverState()


class TestThrottle:
    def test_starts_at_zero(self, state):
        assert state.throttle == 0.0

    def test_w_increases_throttle(self, state):
        state.on_key(drive.KEY_W)
        assert state.throttle == pytest.approx(drive.THROTTLE_STEP)

    def test_throttle_holds_like_cruise_control(self, state):
        """Press-only input means throttle must persist across steps, or the car would
        only move on the exact frame a key was pressed."""
        state.on_key(drive.KEY_W)
        for _ in range(50):
            state.settle()
        assert state.throttle == pytest.approx(drive.THROTTLE_STEP)

    def test_throttle_clamps_at_one(self, state):
        for _ in range(50):
            state.on_key(drive.KEY_W)
        assert state.throttle == 1.0

    def test_throttle_clamps_at_zero(self, state):
        for _ in range(50):
            state.on_key(drive.KEY_S)
        assert state.throttle == 0.0

    def test_arrow_keys_match_wasd(self, state):
        state.on_key(drive.KEY_UP)
        assert state.throttle == pytest.approx(drive.THROTTLE_STEP)
        state.on_key(drive.KEY_DOWN)
        assert state.throttle == pytest.approx(0.0)


class TestSteering:
    def test_a_steers_left_negative(self, state):
        state.on_key(drive.KEY_A)
        assert state.steer < 0

    def test_d_steers_right_positive(self, state):
        state.on_key(drive.KEY_D)
        assert state.steer > 0

    def test_steering_recentres_when_left_alone(self, state):
        state.on_key(drive.KEY_D)
        for _ in range(200):
            state.settle()
        assert state.steer == 0.0

    def test_steering_does_not_snap_straight_to_centre(self, state):
        """One settle should decay, not zero it -- otherwise steering is unusable."""
        state.on_key(drive.KEY_D)
        held = state.steer
        state.settle()
        assert 0 < state.steer < held

    def test_steering_clamps(self, state):
        for _ in range(50):
            state.on_key(drive.KEY_A)
        assert state.steer == -1.0


class TestBrake:
    def test_no_brake_by_default(self, state):
        assert state.brake == 0.0

    def test_space_applies_the_brake(self, state):
        state.on_key(drive.KEY_SPACE)
        assert state.brake == 1.0

    def test_brake_releases_after_the_pulse(self, state):
        state.on_key(drive.KEY_SPACE)
        for _ in range(drive.BRAKE_PULSE_STEPS):
            state.settle()
        assert state.brake == 0.0

    def test_brake_holds_for_the_whole_pulse(self, state):
        state.on_key(drive.KEY_SPACE)
        for _ in range(drive.BRAKE_PULSE_STEPS - 1):
            state.settle()
            assert state.brake == 1.0


class TestCommands:
    def test_r_requests_reset(self, state):
        state.on_key(drive.KEY_R)
        assert state.reset_requested

    def test_escape_requests_quit(self, state):
        state.on_key(drive.KEY_ESCAPE)
        assert state.quit_requested

    def test_unknown_keys_are_ignored(self, state):
        state.on_key(9999)
        assert state.throttle == 0.0 and state.steer == 0.0 and not state.quit_requested


class TestControlsStayInContractRange:
    def test_any_key_sequence_produces_valid_controls(self, state):
        """Whatever the driver mashes, the result must satisfy ControlVector's ranges."""
        from fly_driver.interface import ControlVector

        keys = [drive.KEY_W, drive.KEY_S, drive.KEY_A, drive.KEY_D, drive.KEY_SPACE]
        import random

        rng = random.Random(0)
        for _ in range(500):
            state.on_key(rng.choice(keys))
            state.settle()
            ControlVector(steer=state.steer, throttle=state.throttle, brake=state.brake)
