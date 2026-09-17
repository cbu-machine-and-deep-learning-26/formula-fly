"""Practice-track environment tests (GH-16).

The headline test is the issue's own acceptance criterion -- a random policy steps the env
without crashing -- and it is first in the file. Everything after it guards something that
fails *silently*: a frame the eye will reject, a step interval that quietly stops matching
the optic lobe's integration rate, a camera buried in the bodywork returning a perfectly
plausible flat image, or a frame with no motion signal in it at all.

That last one is the reason this env exists rather than Gymnasium CarRacing (`AGENTS.md`
§6): flyvis models T4/T5, which are elementary *motion* detectors, and a top-down camera
gives them nothing to detect. Up to now that claim lived in a docstring and had never been
measured. :class:`TestOpticFlow` measures it.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.practice_track import PracticeTrack, StepResult
from fly_driver.envs.scene import SceneConfig
from fly_driver.interface import FRAME_DTYPE, FRAME_RATE_HZ, FRAME_SHAPE, ControlVector


def _started(env: PracticeTrack) -> np.ndarray:
    """``env.reset()``, or skip the test if this machine cannot render offscreen."""
    try:
        return env.reset()
    except Exception as exc:  # pragma: no cover - depends on the machine
        env.close()
        pytest.skip(f"no offscreen GL context: {exc}")


def _drive(env: PracticeTrack, control: ControlVector, steps: int) -> StepResult:
    """Hold one control for a while and return the last step."""
    result = None
    for _ in range(steps):
        result = env.step(control)
    assert result is not None
    return result


FLAT_OUT = ControlVector(steer=0.0, throttle=1.0, brake=0.0)
COASTING = ControlVector.neutral()


class TestArguments:
    """Constructor validation. None of these render, so they run anywhere."""

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"frame_shape": (96, 96)},
            {"frame_shape": (96, 96, 1)},
            {"frame_shape": (0, 96, 3)},
            {"frame_rate_hz": 0.0},
            {"frame_rate_hz": -50.0},
            {"track_limit": -0.1},
            {"track_limit": 1.5},
            {"max_steps": 0},
            {"max_steps": -10},
            {"camera": "not_a_camera"},
        ],
    )
    def test_rejects_bad_arguments(self, kwargs):
        with pytest.raises(ValueError):
            PracticeTrack(**kwargs)

    def test_rejects_a_frame_larger_than_the_offscreen_buffer(self):
        """MuJoCo raises for this deep inside the renderer, a long way from the cause."""
        scene = SceneConfig(offscreen_width=64, offscreen_height=64)
        with pytest.raises(ValueError, match="offscreen"):
            PracticeTrack(scene=scene)

    def test_rejects_a_rate_that_does_not_divide_the_timestep(self):
        """A fractional substep count makes the frame interval drift away from 1/rate, and
        since one frame is one Euler step of the optic lobe, nothing downstream would
        notice the eye integrating at a dt it was never fitted at."""
        with pytest.raises(ValueError, match="whole number"):
            PracticeTrack(frame_rate_hz=51.0)

    def test_step_before_reset_is_an_error(self):
        env = PracticeTrack()
        try:
            with pytest.raises(RuntimeError, match="reset"):
                env.step(COASTING)
        finally:
            env.close()

    def test_step_rejects_anything_that_is_not_a_control_vector(self):
        """A bare tuple would be silently unpackable in a dozen plausible ways."""
        env = PracticeTrack()
        try:
            with pytest.raises(TypeError):
                env.step((0.0, 1.0, 0.0))
        finally:
            env.close()


class TestTiming:
    """The substep arithmetic, which needs no renderer and no reset."""

    def test_fifty_hz_is_ten_physics_steps(self):
        env = PracticeTrack()
        try:
            assert env.substeps == 10
            assert env.dt == pytest.approx(1.0 / FRAME_RATE_HZ)
        finally:
            env.close()

    def test_defaults_come_from_the_shared_contract(self):
        """Not restated here. The eye reads the same two constants."""
        env = PracticeTrack()
        try:
            assert env.frame_shape == FRAME_SHAPE
            assert env.frame_rate_hz == FRAME_RATE_HZ
        finally:
            env.close()


@pytest.mark.render
class TestARandomPolicyCanStepIt:
    """GH-16's acceptance criterion, stated the way the issue states it."""

    def test_four_hundred_random_steps_without_crashing(self):
        env = PracticeTrack()
        try:
            _started(env)
            rng = np.random.default_rng(0)
            episodes = 0
            for _ in range(400):
                action = ControlVector.clipped(
                    steer=rng.uniform(-1.0, 1.0),
                    throttle=rng.uniform(0.0, 1.0),
                    brake=rng.uniform(0.0, 1.0),
                )
                result = env.step(action)
                assert np.all(np.isfinite(env.data.qpos))
                assert np.all(np.isfinite(env.data.qvel))
                assert not result.info["diverged"]
                if result.done:
                    episodes += 1
                    env.reset()
            assert episodes >= 0  # the point is that nothing raised
        finally:
            env.close()


@pytest.mark.render
class TestTheFrameIsWhatTheEyeAsksFor:
    def test_shape_and_dtype_match_the_contract(self):
        env = PracticeTrack()
        try:
            frame = _started(env)
            assert frame.shape == FRAME_SHAPE
            assert frame.dtype == FRAME_DTYPE
            assert env.step(FLAT_OUT).frame.shape == FRAME_SHAPE
        finally:
            env.close()

    def test_a_non_default_resolution_is_honoured(self):
        """The eye takes frame_shape as a constructor argument, so this has to be real."""
        env = PracticeTrack(frame_shape=(64, 128, 3))
        try:
            assert _started(env).shape == (64, 128, 3)
        finally:
            env.close()

    def test_the_frame_is_not_one_flat_colour(self):
        """A camera inside the bodywork, or pointed at the sky, renders a perfectly
        plausible constant image and every shape assertion above still passes."""
        env = PracticeTrack()
        try:
            frame = _started(env)
            assert frame.std() > 10.0, "the head camera is looking at nothing"
        finally:
            env.close()

    def test_each_step_returns_a_fresh_array(self):
        """The renderer is free to reuse its buffer. A policy comparing this frame with
        the last would then be comparing a frame with itself."""
        env = PracticeTrack()
        try:
            first = _started(env)
            second = _drive(env, FLAT_OUT, 40).frame
            assert first is not second
            assert not np.array_equal(first, second)
        finally:
            env.close()


@pytest.mark.render
class TestOpticFlow:
    """The reason MuJoCo beat CarRacing. Asserted in prose until now; measured here.

    Both checks are deliberately about *change between frames*, which is what an
    elementary motion detector reads. Neither asks the image to zoom cleanly: at 96x96
    the sky and the car's own bodywork fill most of the frame and the horizon barely
    expands at all, so a whole-frame scale fit would be measuring the wrong thing.
    """

    @staticmethod
    def _motion_energy(frames: list[np.ndarray]) -> float:
        stack = np.stack([frame.astype(np.float64).mean(axis=2) for frame in frames])
        return float(np.abs(np.diff(stack, axis=0)).mean())

    @staticmethod
    def _sideways_shift(before: np.ndarray, after: np.ndarray, span: int = 12) -> int:
        """Horizontal shift that best maps one frame onto the next, in pixels.

        Restricted to the band between the sky and the bodywork, which is where the
        ground, the kerbs and the trees are. Negative means the world moved left.
        """
        height = before.shape[0]
        rows = slice(int(height * 0.30), int(height * 0.60))
        a = before.astype(np.float64).mean(axis=2)[rows]
        b = after.astype(np.float64).mean(axis=2)[rows]
        width = a.shape[1]
        best, best_error = 0, float("inf")
        for shift in range(-span, span + 1):
            rolled = np.roll(a, shift, axis=1)
            keep = slice(shift, None) if shift >= 0 else slice(None, width + shift)
            error = float(np.abs(rolled[:, keep] - b[:, keep]).mean())
            if error < best_error:
                best_error, best = error, shift
        return best

    def test_a_moving_car_makes_motion_signal_and_a_parked_one_does_not(self):
        env = PracticeTrack()
        try:
            _started(env)
            _drive(env, COASTING, 60)  # let the springs settle
            still = self._motion_energy([env.step(COASTING).frame for _ in range(8)])
            _drive(env, FLAT_OUT, 120)
            moving = self._motion_energy([env.step(FLAT_OUT).frame for _ in range(8)])
        finally:
            env.close()
        assert still < 0.5, f"a parked car's camera is moving: {still:.3f}"
        assert moving > 10 * max(still, 0.05), f"driving produced no motion signal: {moving:.3f}"

    @pytest.mark.slow
    def test_steering_slides_the_world_the_other_way(self):
        """Turning right sweeps the scene left across the retina, and vice versa. Get the
        sign wrong and a policy would simply learn the mirror image, which is exactly the
        kind of error `AGENTS.md` §11 says to pin with a test rather than eyeball."""
        shifts = {}
        for name, steer in (("left", -0.8), ("straight", 0.0), ("right", 0.8)):
            env = PracticeTrack()
            try:
                _started(env)
                _drive(env, FLAT_OUT, 120)
                control = ControlVector(steer=steer, throttle=0.3, brake=0.0)
                before = env.step(control).frame
                after = _drive(env, control, 9).frame
                shifts[name] = self._sideways_shift(before, after)
            finally:
                env.close()
        assert shifts["right"] < shifts["straight"] < shifts["left"], shifts


@pytest.mark.render
class TestEpisodeBoundaries:
    def test_leaving_the_circuit_ends_the_episode(self):
        env = PracticeTrack()
        try:
            _started(env)
            assert not _drive(env, FLAT_OUT, 120).terminated, "straight ahead is not off track"
            lock = ControlVector(steer=1.0, throttle=1.0, brake=0.0)
            for _ in range(200):
                result = env.step(lock)
                if result.terminated:
                    break
            assert result.terminated, "full lock at speed never left the circuit"
            assert result.info["off_track_fraction"] > 0.15
        finally:
            env.close()

    def test_the_rule_can_be_turned_off(self):
        env = PracticeTrack(track_limit=0.0)
        try:
            _started(env)
            lock = ControlVector(steer=1.0, throttle=1.0, brake=0.0)
            _drive(env, FLAT_OUT, 120)
            assert not _drive(env, lock, 200).terminated
        finally:
            env.close()

    def test_max_steps_truncates_rather_than_terminates(self):
        """Bootstrapping a value function treats the two differently, so they are separate
        flags rather than one `done`."""
        env = PracticeTrack(max_steps=30)
        try:
            _started(env)
            result = _drive(env, FLAT_OUT, 30)
            assert result.truncated
            assert not result.terminated
            assert result.done
            assert result.info["steps"] == 30
        finally:
            env.close()

    def test_reset_puts_the_car_back_and_restarts_the_clock(self):
        env = PracticeTrack()
        try:
            _started(env)
            _drive(env, FLAT_OUT, 60)
            moved = float(env.data.time)
            env.reset()
            assert env.steps == 0
            assert float(env.data.time) == 0.0 < moved
            assert env.dynamics.gear == 0
            assert env.lap_timer.completed == 0
        finally:
            env.close()


@pytest.mark.render
class TestTimingHolds:
    def test_one_step_is_one_frame_interval_of_simulated_time(self):
        """One frame is one Euler step of the optic lobe. If this drifts, the eye is
        integrating at a dt the pretrained network was never fitted at, and nothing
        downstream of it can tell."""
        env = PracticeTrack()
        try:
            _started(env)
            before = float(env.data.time)
            env.step(FLAT_OUT)
            assert float(env.data.time) - before == pytest.approx(1.0 / FRAME_RATE_HZ)
            _drive(env, FLAT_OUT, 99)
            assert float(env.data.time) == pytest.approx(100 / FRAME_RATE_HZ)
            assert env.steps == 100
        finally:
            env.close()


@pytest.mark.render
class TestDeterminism:
    """`AGENTS.md` §11 asks for this explicitly: the eval protocol has to be reproducible."""

    def test_the_same_seed_gives_the_same_frames_and_the_same_trajectory(self):
        controls = [
            ControlVector.clipped(steer=np.sin(i * 0.3), throttle=0.6, brake=0.0) for i in range(60)
        ]
        runs = []
        for _ in range(2):
            env = PracticeTrack(seed=7)
            try:
                frames = [_started(env)]
                infos = []
                for control in controls:
                    result = env.step(control)
                    frames.append(result.frame)
                    infos.append(result.info["progress_m"])
                runs.append((frames, infos, env.data.qpos.copy()))
            finally:
                env.close()
        first, second = runs
        assert all(np.array_equal(a, b) for a, b in zip(first[0], second[0], strict=True))
        assert first[1] == second[1]
        assert np.array_equal(first[2], second[2])


@pytest.mark.render
class TestInfo:
    """Raw signals for GH-17 to build a reward from. There is deliberately no reward here."""

    def test_it_reports_no_reward(self):
        env = PracticeTrack()
        try:
            _started(env)
            info = env.step(FLAT_OUT).info
            assert "reward" not in info
        finally:
            env.close()

    def test_progress_is_positive_going_forwards(self):
        env = PracticeTrack()
        try:
            _started(env)
            result = _drive(env, FLAT_OUT, 80)
            assert result.info["progress_m"] > 0.0
            assert result.info["speed_mps"] > 5.0
            assert result.info["on_track"]
            assert result.info["off_track_fraction"] == 0.0
        finally:
            env.close()

    def test_the_car_starts_behind_the_line_on_an_out_lap(self):
        """The clock must not start at boot: the run-up is not charged to lap one."""
        env = PracticeTrack()
        try:
            _started(env)
            info = env.step(FLAT_OUT).info
            assert info["on_out_lap"]
            assert info["lap_time"] == 0.0
            assert info["lap_count"] == 0
            assert info["lap_completed"] is None
        finally:
            env.close()

    def test_the_clock_starts_when_the_car_crosses_the_line(self):
        env = PracticeTrack()
        try:
            _started(env)
            for _ in range(600):
                info = env.step(FLAT_OUT).info
                if not info["on_out_lap"]:
                    break
            assert not info["on_out_lap"], "never reached the start line"
            assert _drive(env, FLAT_OUT, 20).info["lap_time"] > 0.0
        finally:
            env.close()


@pytest.mark.render
class TestTheEyeSeam:
    """Proof the vision stage plugs in, on a machine with no flyvis and no torch.

    The stand-in stands in for :class:`fly_driver.eyes.FlyvisEye`: same two methods, same
    contract, and it checks the frame it is handed the way the real resampler does. When
    the real eye arrives it drops into the same slot -- which is the whole point of
    `AGENTS.md` §3 having one interface.
    """

    class StandInEye:
        feature_dim = 5768

        def __init__(self, frame_shape, frame_rate_hz):
            assert frame_rate_hz >= 50.0, "flyvis cannot integrate slower than 50 Hz"
            self.frame_shape = tuple(frame_shape)
            self.resets = 0
            self.frames = 0

        def reset(self):
            self.resets += 1

        def encode(self, frame):
            assert frame.shape == self.frame_shape, frame.shape
            assert frame.dtype == FRAME_DTYPE
            self.frames += 1
            return np.zeros(self.feature_dim, dtype=np.float32)

    def test_a_closed_loop_runs_through_an_eye_and_a_policy(self):
        env = PracticeTrack(max_steps=50)
        eye = self.StandInEye(env.frame_shape, env.frame_rate_hz)
        rng = np.random.default_rng(3)

        def policy(features):
            assert features.shape == (eye.feature_dim,)
            assert features.dtype == np.float32
            return ControlVector.clipped(rng.uniform(-0.3, 0.3), 0.5, 0.0)

        try:
            for _ in range(2):
                frame = env.reset()
                eye.reset()  # the real eye keeps state between frames; episodes must not
                while True:
                    result = env.step(policy(eye.encode(frame)))
                    frame = result.frame
                    if result.done:
                        break
        finally:
            env.close()

        assert eye.resets == 2
        assert eye.frames == 100


class TestClose:
    def test_closing_twice_is_safe(self):
        env = PracticeTrack()
        env.close()
        env.close()

    def test_using_it_after_close_raises_rather_than_crashing_the_process(self):
        """A use-after-free in a GL context is a segfault with no traceback."""
        env = PracticeTrack()
        env.close()
        with pytest.raises(RuntimeError, match="closed"):
            env.reset()
        with pytest.raises(RuntimeError, match="closed"):
            env.step(COASTING)

    @pytest.mark.render
    def test_it_is_a_context_manager(self):
        with PracticeTrack() as env:
            _started(env)
            env.step(FLAT_OUT)
        with pytest.raises(RuntimeError, match="closed"):
            env.step(FLAT_OUT)
