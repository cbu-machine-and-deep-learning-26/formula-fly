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


class TestTheRacingLineFlag:
    """Off by default for a human driver, on with ``--racing-line``.

    Only here. SceneConfig still defaults the line on, because in the env it is a cue the
    fly is meant to see -- so the test checks both, or flipping the env's default to match
    the drive tool would pass unnoticed and take the cue away from the fly.
    """

    def _exported(self, tmp_path, *flags: str) -> str:
        path = tmp_path / "model.xml"
        assert drive.main(["--export", str(path), *flags]) == 0
        return path.read_text(encoding="utf-8")

    def test_the_drive_tool_leaves_it_off_by_default(self, tmp_path):
        assert "racing_line_geom" not in self._exported(tmp_path)

    def test_the_flag_paints_it(self, tmp_path):
        assert "racing_line_geom" in self._exported(tmp_path, "--racing-line")

    def test_the_env_still_shows_it_to_the_fly_by_default(self):
        from fly_driver.envs.scene import SceneConfig

        assert SceneConfig().racing_line is True


class TestWhatDriveFlyNeedsFromIt:
    """``scripts/drive_fly.py`` (GH-21) imports from this script:
    ``from scripts.drive import build, reset_to_start``.

    Nothing else tests that, so renaming either function here would break the fly-body
    driver without a single test failing (#70). These read drive_fly.py itself, so a new
    import added there later is covered without anyone remembering to come back here.
    """

    _DRIVE_FLY = Path(__file__).resolve().parents[2] / "scripts" / "drive_fly.py"

    def _imported_from_drive(self) -> set[str]:
        import ast

        tree = ast.parse(self._DRIVE_FLY.read_text(encoding="utf-8"))
        return {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "scripts.drive"
            for alias in node.names
        }

    def test_every_name_drive_fly_imports_still_exists(self):
        names = self._imported_from_drive()
        assert names, "drive_fly.py no longer imports from drive.py; this test can go"
        missing = sorted(name for name in names if not hasattr(drive, name))
        assert not missing, f"drive_fly.py imports {missing} from drive.py, which is gone"

    def test_they_still_work_the_way_drive_fly_calls_them(self):
        """``build(car, scene)`` then ``reset_to_start(model, data)``, as its render phase does."""
        import mujoco

        from fly_driver.envs.car import CarConfig
        from fly_driver.envs.scene import SceneConfig

        centerline, model = drive.build(CarConfig(), SceneConfig())
        data = mujoco.MjData(model)
        data.time = 12.0
        drive.reset_to_start(model, data)
        assert data.time == 0.0, "reset_to_start no longer puts the car back on the grid"
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car") >= 0
        assert centerline.length > 5000.0, "build no longer loads Silverstone"


class TestAgentModeOnTheTrack:
    """``--agent``'s loop, run without flyvis or a window.

    The real eye needs flyvis and the viewer needs a desktop, so neither the default venv
    nor CI ever runs this loop, and GH-16 changed three things under it without a single
    test failing: ``--track-limit``'s default became 0, which the env reads as "never end
    the episode"; ``lap_time`` started including the penalty; and the drive tool's scene
    stopped painting the racing line the policy is trained on. The env, the viewer and the
    agent are stubbed here so the loop itself runs. What reaches the env, and what reaches
    the record book, is real.
    """

    _CAR_ONLY = (
        '<mujoco><worldbody><body name="car"><freejoint/><geom size="0.1"/>'
        "</body></worldbody></mujoco>"
    )

    @staticmethod
    def _info(**overrides: object) -> dict[str, object]:
        info: dict[str, object] = {
            "lap_complete": False,
            "lap_time": None,
            "lap_time_raw": None,
            "lap_penalty_s": None,
            "speed_mps": 10.0,
            "on_track": True,
            "lap_fraction": 0.1,
            "lateral_m": 0.0,
            "gear": 1,
            "penalty_s": 0.0,
            "segment": "S1",
            "segment_elapsed_s": 1.0,
        }
        info.update(overrides)
        return info

    def _run(self, monkeypatch, argv, infos, *, backspace_at: int | None = None):
        """Drive ``drive.main(argv)`` for one step per entry in ``infos``.

        Returns the stub env (its constructor kwargs and reset count) and every call that
        reached the record book. ``backspace_at`` resets the sim clock from inside the
        viewer on that sync, as MuJoCo's own BACKSPACE does.
        """
        import mujoco
        import mujoco.viewer

        import fly_driver.envs.practice_track as practice_track

        model = mujoco.MjModel.from_xml_string(self._CAR_ONLY)

        class StubTrack:
            instance: StubTrack | None = None

            def __init__(self, **kwargs: object) -> None:
                StubTrack.instance = self
                self.kwargs = kwargs
                self.model = model
                self.data = mujoco.MjData(model)
                self.frame_shape = (96, 96, 3)
                self.frame_rate_hz = 50.0
                self.dt = 0.0
                self.centerline = type("Line", (), {"length": 5891.0})()
                self.lap_timer = type("Timer", (), {"completed": 0})()
                self.resets = 0
                self._infos = iter(infos)

            def reset(self, seed: int | None = None):
                del seed
                self.resets += 1
                mujoco.mj_resetData(self.model, self.data)
                return np.zeros(self.frame_shape, dtype=np.uint8), self._info_now()

            def _info_now(self) -> dict[str, object]:
                return TestAgentModeOnTheTrack._info()

            def step(self, control: ControlVector):
                del control
                self.data.time += 0.02
                info = next(self._infos)
                return np.zeros(self.frame_shape, dtype=np.uint8), 0.0, False, False, info

            def close(self) -> None:
                pass

        class StubViewer:
            def __init__(self) -> None:
                self.syncs = 0

            def __enter__(self) -> StubViewer:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

            def is_running(self) -> bool:
                return self.syncs < len(infos)

            def sync(self) -> None:
                self.syncs += 1
                if backspace_at is not None and self.syncs == backspace_at:
                    assert StubTrack.instance is not None
                    mujoco.mj_resetData(StubTrack.instance.model, StubTrack.instance.data)

        class StubAgent:
            def reset(self) -> None:
                pass

            def act(self, frame: object) -> ControlVector:
                del frame
                return ControlVector(steer=0.0, throttle=0.4, brake=0.0)

        recorded: list[tuple[float, dict[str, object]]] = []

        class StubLapLog:
            def __init__(self, path: object) -> None:
                del path

            def record(self, seconds: float, **kwargs: object) -> bool:
                recorded.append((seconds, kwargs))
                return False

            def best(self) -> None:
                return None

        monkeypatch.setattr(practice_track, "PracticeTrack", StubTrack)
        monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *a, **k: StubViewer())
        monkeypatch.setattr(drive, "load_direct_drive_agent", lambda **kwargs: StubAgent())
        monkeypatch.setattr(drive, "LapLog", StubLapLog)

        assert drive.main(argv) == 0
        assert StubTrack.instance is not None
        return StubTrack.instance, recorded

    def test_leaving_the_track_still_ends_the_episode(self, monkeypatch):
        """The drive tool's ``--track-limit`` (0 by default) must not reach the env, where
        0 turns the off-track restart off and strands the car in the grass."""
        env, _ = self._run(monkeypatch, ["--agent", "--no-hud"], [self._info()])
        assert "track_limit" not in env.kwargs
        assert "terminate_off_track" not in env.kwargs

    def test_it_sees_the_scene_training_renders(self, monkeypatch):
        """Racing line on, as in training, although the drive tool leaves it off for a human."""
        env, _ = self._run(monkeypatch, ["--agent", "--no-hud"], [self._info()])
        assert env.kwargs["scene"].racing_line is True

    def test_a_cut_lap_is_recorded_as_raw_time_plus_penalty(self, monkeypatch, tmp_path):
        lap = self._info(lap_complete=True, lap_time=92.5, lap_time_raw=90.0, lap_penalty_s=2.5)
        _, recorded = self._run(
            monkeypatch,
            ["--agent", "--no-hud", "--lap-log", str(tmp_path / "laps.md")],
            [self._info(), lap],
        )
        assert len(recorded) == 1
        seconds, kwargs = recorded[0]
        assert seconds == 90.0, "the penalty was counted into the raw time"
        assert kwargs["penalty_seconds"] == 2.5
        assert kwargs["driver"] == "direct-drive"

    def test_a_clean_lap_is_recorded_with_no_penalty(self, monkeypatch, tmp_path):
        lap = self._info(lap_complete=True, lap_time=90.0, lap_time_raw=90.0, lap_penalty_s=None)
        _, recorded = self._run(
            monkeypatch,
            ["--agent", "--no-hud", "--lap-log", str(tmp_path / "laps.md")],
            [lap],
        )
        assert recorded == [(90.0, {"penalty_seconds": 0.0, "driver": "direct-drive"})]

    def test_backspace_in_the_viewer_resets_the_env(self, monkeypatch):
        """MuJoCo's BACKSPACE zeroes the clock behind the env; the loop must notice."""
        env, _ = self._run(
            monkeypatch,
            ["--agent", "--no-hud"],
            [self._info(), self._info(), self._info()],
            backspace_at=1,
        )
        assert env.resets == 2, "the start plus one reset for BACKSPACE"

    def test_no_backspace_no_extra_reset(self, monkeypatch):
        env, _ = self._run(
            monkeypatch, ["--agent", "--no-hud"], [self._info(), self._info(), self._info()]
        )
        assert env.resets == 1
