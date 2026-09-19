"""Practice-track environment tests (GH-16).

The headline test is the issue's own acceptance criterion -- a random policy steps the env
without crashing -- and it is first in the file. Everything after it guards something that
fails *silently*: a frame the eye will reject, a step interval that quietly stops matching
the optic lobe's integration rate, a camera buried in the bodywork returning a perfectly
plausible flat image, an info key the evaluation harness reads under a different meaning, or
a frame with no motion signal in it at all.

That last one is the reason this env exists rather than Gymnasium CarRacing (`AGENTS.md`
§6): flyvis models T4/T5, which are elementary *motion* detectors, and a top-down camera
gives them nothing to detect. #16 asks for "expansion when going forward and slide when
turning" and until now that lived in a docstring and had never been measured.
:class:`TestOpticFlow` measures both.
"""

from __future__ import annotations

from collections import namedtuple

import mujoco
import numpy as np
import pytest

from fly_driver.envs.penalty import PenaltyWeights
from fly_driver.envs.practice_track import PracticeTrack, ProgressReward
from fly_driver.envs.scene import SceneConfig
from fly_driver.interface import FRAME_DTYPE, FRAME_RATE_HZ, FRAME_SHAPE, ControlVector

Step = namedtuple("Step", "frame reward terminated truncated info")

FLAT_OUT = ControlVector(steer=0.0, throttle=1.0, brake=0.0)
COASTING = ControlVector.neutral()


def _started(env: PracticeTrack) -> tuple[np.ndarray, dict]:
    """``env.reset()``, or skip the test if this machine cannot render offscreen."""
    try:
        return env.reset()
    except Exception as exc:  # pragma: no cover - depends on the machine
        env.close()
        pytest.skip(f"no offscreen GL context: {exc}")


def _step(env: PracticeTrack, control) -> Step:
    return Step(*env.step(control))


def _drive(env: PracticeTrack, control, steps: int) -> Step:
    """Hold one control for a while and return the last step."""
    result = None
    for _ in range(steps):
        result = _step(env, control)
    assert result is not None
    return result


def _speed_mph(env: PracticeTrack) -> float:
    return env.dynamics.speed_mps(env.data) * 2.23694


def _yaw_rate(env: PracticeTrack) -> float:
    """Magnitude of the car's yaw rate, rad/s, straight from the simulation."""
    velocity = np.zeros(6)
    body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "car")
    mujoco.mj_objectVelocity(env.model, env.data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
    return abs(float(velocity[2]))


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

    def test_render_before_reset_is_an_error(self):
        env = PracticeTrack()
        try:
            with pytest.raises(RuntimeError, match="reset"):
                env.render()
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
            assert env.observation_shape == FRAME_SHAPE
            assert env.action_shape == (3,)
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
            for _ in range(400):
                action = rng.uniform([-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]).astype(np.float32)
                result = _step(env, action)
                assert np.all(np.isfinite(env.data.qpos))
                assert np.all(np.isfinite(env.data.qvel))
                assert np.isfinite(result.reward)
                assert not result.info["diverged"]
                if result.terminated or result.truncated:
                    env.reset()
        finally:
            env.close()


@pytest.mark.render
class TestTheEvaluationHarnessCanDriveIt:
    """`fly_driver.training.evaluation` (GH-18) declares an ``Env`` protocol and the dummy
    track speaks it. Two env shapes in one repo is exactly the drift `AGENTS.md` §3 exists
    to stop, so the shape is pinned here rather than left to a reviewer to notice."""

    def test_reset_returns_a_frame_and_an_info_mapping(self):
        env = PracticeTrack()
        try:
            returned = env.reset(seed=11)
            assert isinstance(returned, tuple) and len(returned) == 2
            frame, info = returned
            assert frame.dtype == FRAME_DTYPE
            assert isinstance(info, dict)
        finally:
            env.close()

    def test_step_returns_the_gymnasium_five_tuple(self):
        env = PracticeTrack()
        try:
            _started(env)
            returned = env.step(np.array([0.0, 1.0, 0.0], dtype=np.float32))
            assert isinstance(returned, tuple) and len(returned) == 5
            frame, reward, terminated, truncated, info = returned
            assert frame.dtype == FRAME_DTYPE
            assert isinstance(reward, float)
            assert isinstance(terminated, bool) and isinstance(truncated, bool)
            assert isinstance(info, dict)
        finally:
            env.close()

    def test_render_gives_the_last_frame_as_a_copy(self):
        """The harness calls render() once per step while recording video and keeps the
        array. Handing back the live buffer would make every recorded frame the last one."""
        env = PracticeTrack()
        try:
            frame, _ = _started(env)
            rendered = env.render()
            assert np.array_equal(rendered, frame)
            assert rendered is not frame
            moved = _drive(env, FLAT_OUT, 30).frame
            assert np.array_equal(env.render(), moved)
        finally:
            env.close()

    @pytest.mark.parametrize(
        "action",
        [
            ControlVector(steer=0.0, throttle=0.5, brake=0.0),
            np.array([0.0, 0.5, 0.0], dtype=np.float32),
            [0.0, 0.5, 0.0],
            (0.0, 0.5, 0.0),
        ],
    )
    def test_it_accepts_every_action_form_the_harness_might_send(self, action):
        """The harness converts an agent's output to a (3,) float32 array before it reaches
        an env, so refusing arrays here would make this env unevaluatable."""
        env = PracticeTrack()
        try:
            _started(env)
            env.step(action)
        finally:
            env.close()

    @pytest.mark.parametrize(
        "action",
        [
            np.array([2.0, 0.0, 0.0]),  # out of range
            np.array([np.nan, 0.0, 0.0]),  # non-finite
            np.zeros(4),  # wrong shape
        ],
    )
    def test_it_refuses_a_bad_action_rather_than_clipping_it(self, action):
        env = PracticeTrack()
        try:
            _started(env)
            with pytest.raises(ValueError):
                env.step(action)
        finally:
            env.close()

    def test_lap_time_is_the_completed_lap_not_the_running_clock(self):
        """The harness reads info["lap_time"] *only* when info["lap_complete"] is true, and
        substitutes steps / frame_rate_hz when it is None. A running clock here would
        therefore be reported as a lap time that looks entirely reasonable and is wrong."""
        env = PracticeTrack()
        try:
            _started(env)
            info = _drive(env, FLAT_OUT, 60).info
            assert info["lap_complete"] is False
            assert info["lap_time"] is None
            assert info["lap_elapsed_s"] >= 0.0  # the running clock lives under its own key
        finally:
            env.close()


@pytest.mark.render
class TestReward:
    def test_the_terms_add_up_to_the_reward(self):
        env = PracticeTrack()
        try:
            _started(env)
            result = _drive(env, FLAT_OUT, 40)
            assert sum(result.info["reward_terms"].values()) == pytest.approx(result.reward)
        finally:
            env.close()

    def test_standing_still_earns_nothing(self):
        """`AGENTS.md` §11: verify the agent cannot farm reward without progressing."""
        env = PracticeTrack()
        try:
            _started(env)
            parked = sum(_step(env, COASTING).reward for _ in range(100))
        finally:
            env.close()
        assert parked <= 0.0, f"a parked car earned {parked:.3f}"

    def test_driving_earns_more_than_coasting(self):
        env = PracticeTrack()
        try:
            _started(env)
            driven = sum(_step(env, FLAT_OUT).reward for _ in range(100))
        finally:
            env.close()
        assert driven > 10.0, f"100 steps of full throttle earned only {driven:.3f}"

    def test_leaving_the_circuit_is_punished(self):
        env = PracticeTrack(track_limit=0.0)  # keep going so the penalty is observable
        try:
            _started(env)
            _drive(env, FLAT_OUT, 120)
            lock = ControlVector(steer=1.0, throttle=1.0, brake=0.0)
            for _ in range(300):
                result = _step(env, lock)
                if result.info["off_track"]:
                    break
            assert result.info["off_track"], "full lock at speed never left the circuit"
            assert result.info["reward_terms"]["off_track"] < 0.0
        finally:
            env.close()

    def test_the_reward_function_is_an_argument(self):
        """GH-17 owns reward shaping. This env supplies a default and the seam, not a rule."""

        def always_seven(info, dt):
            del info, dt
            return 7.0, {"seven": 7.0}

        env = PracticeTrack(reward=always_seven)
        try:
            _started(env)
            assert _step(env, FLAT_OUT).reward == 7.0
        finally:
            env.close()

    def test_the_default_matches_the_dummy_tracks_constants(self):
        """The dummy exists so the harness runs without MuJoCo. A stand-in whose returns
        are on a different scale from the real thing is a poor stand-in."""
        reward = ProgressReward()
        assert (reward.lateral_penalty, reward.lap_bonus, reward.off_track_penalty) == (
            0.1,
            100.0,
            -10.0,
        )


@pytest.mark.render
class TestTheFrameIsWhatTheEyeAsksFor:
    def test_shape_and_dtype_match_the_contract(self):
        env = PracticeTrack()
        try:
            frame, _ = _started(env)
            assert frame.shape == FRAME_SHAPE
            assert frame.dtype == FRAME_DTYPE
            assert _step(env, FLAT_OUT).frame.shape == FRAME_SHAPE
        finally:
            env.close()

    def test_a_non_default_resolution_is_honoured(self):
        """The eye takes frame_shape as a constructor argument, so this has to be real."""
        env = PracticeTrack(frame_shape=(64, 128, 3))
        try:
            frame, _ = _started(env)
            assert frame.shape == (64, 128, 3)
        finally:
            env.close()

    def test_the_frame_is_not_one_flat_colour(self):
        """A camera inside the bodywork, or pointed at the sky, renders a perfectly
        plausible constant image and every shape assertion above still passes."""
        env = PracticeTrack()
        try:
            frame, _ = _started(env)
            assert frame.std() > 10.0, "the head camera is looking at nothing"
        finally:
            env.close()

    def test_each_step_returns_a_fresh_array(self):
        """The renderer is free to reuse its buffer. A policy comparing this frame with
        the last would then be comparing a frame with itself."""
        env = PracticeTrack()
        try:
            first, _ = _started(env)
            second = _drive(env, FLAT_OUT, 40).frame
            assert first is not second
            assert not np.array_equal(first, second)
        finally:
            env.close()


@pytest.mark.render
class TestOpticFlow:
    """#16's stated purpose: "the eye sees expansion when going forward and slide when
    turning". Measured here rather than asserted in prose.

    All three checks read *change between frames*, which is what an elementary motion
    detector reads. None asks the whole image to scale cleanly: at 96x96 the sky and the
    car's own bodywork fill most of the frame and the horizon barely moves, so a
    whole-frame fit measures the wrong thing. The flow lives in the band between them.
    """

    #: Rows carrying ground, kerbs and trees -- below the sky, above the bodywork.
    BAND = (0.30, 0.60)

    @classmethod
    def _band(cls, frame: np.ndarray, columns: slice = slice(None)) -> np.ndarray:
        height = frame.shape[0]
        rows = slice(int(height * cls.BAND[0]), int(height * cls.BAND[1]))
        return frame.astype(np.float64).mean(axis=2)[rows, columns]

    @staticmethod
    def _shift(before: np.ndarray, after: np.ndarray, span: int = 20) -> int:
        """Horizontal shift in pixels that best maps one band onto the next.

        Negative means the world moved left. Wrapped columns are excluded from the
        comparison, so a large shift is not rewarded for matching its own tail.
        """
        width = before.shape[1]
        best, best_error = 0, float("inf")
        for shift in range(-span, span + 1):
            rolled = np.roll(before, shift, axis=1)
            keep = slice(shift, None) if shift >= 0 else slice(None, width + shift)
            error = float(np.abs(rolled[:, keep] - after[:, keep]).mean())
            if error < best_error:
                best_error, best = error, shift
        return best

    @staticmethod
    def _motion_energy(frames: list[np.ndarray]) -> float:
        stack = np.stack([frame.astype(np.float64).mean(axis=2) for frame in frames])
        return float(np.abs(np.diff(stack, axis=0)).mean())

    def test_a_moving_car_makes_motion_signal_and_a_parked_one_does_not(self):
        env = PracticeTrack()
        try:
            _started(env)
            _drive(env, COASTING, 60)  # let the springs settle
            still = self._motion_energy([_step(env, COASTING).frame for _ in range(8)])
            _drive(env, FLAT_OUT, 120)
            moving = self._motion_energy([_step(env, FLAT_OUT).frame for _ in range(8)])
        finally:
            env.close()
        assert still < 0.5, f"a parked car's camera is moving: {still:.3f}"
        assert moving > 10 * max(still, 0.05), f"driving produced no motion signal: {moving:.3f}"

    @pytest.mark.slow
    def test_driving_forward_expands_the_view(self):
        """Expansion, in #16's own words: going forward pushes the world outwards, so the
        left of the frame sweeps left while the right sweeps right.

        Gated on speed and yaw rate rather than a step count, so it survives the car being
        recalibrated. The upper bound is not caution: above roughly 140 mph the car covers
        enough ground in seven frames that the two bands stop overlapping and the estimator
        stops meaning anything (both halves collapse to the same saturated value). Inside
        the window it is unambiguous -- measured -2/+20 at 69 mph, -17/+10 at 95 mph and
        -20/+10 at 123 mph.
        """
        env = PracticeTrack()
        try:
            _started(env)
            for _ in range(600):
                _step(env, FLAT_OUT)
                if 60.0 < _speed_mph(env) < 125.0 and _yaw_rate(env) < 0.05:
                    break
            else:  # pragma: no cover - the car always reaches the window on the straight
                pytest.fail("never found a straight stretch inside the speed window")

            before = _step(env, FLAT_OUT).frame
            after = _drive(env, FLAT_OUT, 7).frame
            half = before.shape[1] // 2
            left = self._shift(
                self._band(before, slice(0, half)), self._band(after, slice(0, half))
            )
            right = self._shift(
                self._band(before, slice(half, None)), self._band(after, slice(half, None))
            )
        finally:
            env.close()
        assert left < right, f"the view is not expanding: left={left:+d} right={right:+d}"

    @pytest.mark.slow
    def test_steering_slides_the_world_the_other_way(self):
        """Slide, the other half of #16's sentence. Turning right sweeps the scene left
        across the retina, and vice versa. Get the sign wrong and a policy would simply
        learn the mirror image, which is exactly the kind of error `AGENTS.md` §11 says to
        pin with a test rather than eyeball."""
        shifts = {}
        for name, steer in (("left", -0.8), ("straight", 0.0), ("right", 0.8)):
            env = PracticeTrack()
            try:
                _started(env)
                _drive(env, FLAT_OUT, 120)
                control = ControlVector(steer=steer, throttle=0.3, brake=0.0)
                before = _step(env, control).frame
                after = _drive(env, control, 9).frame
                shifts[name] = self._shift(self._band(before), self._band(after), span=12)
            finally:
                env.close()
        assert shifts["right"] < shifts["straight"] < shifts["left"], shifts


@pytest.mark.render
class TestEpisodeBoundaries:
    def test_leaving_the_circuit_ends_the_episode_when_asked(self):
        """Opt-in now. It used to be the default, and turning it off is the behaviour
        change in this branch -- see test_leaving_the_circuit_no_longer_ends_the_episode."""
        env = PracticeTrack(terminate_off_track=True)
        try:
            _started(env)
            assert not _drive(env, FLAT_OUT, 120).terminated, "straight ahead is not off track"
            lock = ControlVector(steer=1.0, throttle=1.0, brake=0.0)
            for _ in range(200):
                result = _step(env, lock)
                if result.terminated:
                    break
            assert result.terminated, "full lock at speed never left the circuit"
            assert result.info["off_track_fraction"] > 0.15
        finally:
            env.close()

    def test_leaving_the_circuit_no_longer_ends_the_episode(self):
        """The default changed. A terminated episode says something ended and nothing
        about how badly, so going off costs seconds instead -- and an episode that keeps
        running is the one that can be recovered from and learned from."""
        env = PracticeTrack()
        try:
            _started(env)
            lock = ControlVector(steer=1.0, throttle=1.0, brake=0.0)
            _drive(env, FLAT_OUT, 120)
            result = _drive(env, lock, 200)
            assert not result.terminated
            assert result.info["off_track_fraction"] > 0.15, "the car did leave the circuit"
            assert result.info["penalty_s"] > 0.0, "and it was charged for it"
        finally:
            env.close()

    def test_the_rule_can_be_turned_off(self):
        env = PracticeTrack(track_limit=0.0, terminate_off_track=True)
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
            assert result.info["steps"] == 30
        finally:
            env.close()

    def test_stepping_past_the_end_raises(self):
        """A loop that ignores `terminated` should fail loudly, not keep driving a car that
        is already in a field. The dummy track raises here too, with the same message."""
        env = PracticeTrack(max_steps=5)
        try:
            _started(env)
            _drive(env, FLAT_OUT, 5)
            with pytest.raises(RuntimeError, match="episode has ended"):
                env.step(FLAT_OUT)
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
            _step(env, FLAT_OUT)
            assert float(env.data.time) - before == pytest.approx(1.0 / FRAME_RATE_HZ)
            _drive(env, FLAT_OUT, 99)
            assert float(env.data.time) == pytest.approx(100 / FRAME_RATE_HZ)
            assert env.steps == 100
        finally:
            env.close()


@pytest.mark.render
class TestDeterminism:
    """`AGENTS.md` §11 asks for this explicitly: the eval protocol has to be reproducible."""

    def test_the_same_seed_gives_the_same_trajectory(self):
        controls = [
            ControlVector.clipped(steer=np.sin(i * 0.3), throttle=0.6, brake=0.0) for i in range(60)
        ]
        runs = []
        for _ in range(2):
            env = PracticeTrack(seed=7)
            try:
                frames = [_started(env)[0]]
                rewards = []
                for control in controls:
                    result = _step(env, control)
                    frames.append(result.frame)
                    rewards.append(result.reward)
                runs.append((frames, rewards, env.data.qpos.copy()))
            finally:
                env.close()
        first, second = runs

        # The physics is the part that has to be exact, and is.
        assert np.array_equal(first[2], second[2])
        assert first[1] == second[1]

        # The rendering is not, quite. MuJoCo hands rasterisation to the GPU, and two
        # identical runs on this machine occasionally disagree by a single level out of 255
        # on a scattering of subpixels along polygon edges -- measured over 488 frame pairs:
        # 0.6% of frames affected, never more than 1/255, at most 0.025% of subpixels, so a
        # mean absolute difference of 0.00025 levels. Far below anything the hex resampler's
        # 13-pixel mean filter can see, but it does mean frames must never be hashed or
        # compared for exact equality in a determinism check.
        #
        # The bound is on the mean rather than on a share of subpixels: the share has a long
        # tail as GPU scheduling varies, while a genuinely different frame moves the mean by
        # tens of levels. 0.05 leaves two orders of magnitude of headroom either way.
        for before, after in zip(first[0], second[0], strict=True):
            delta = np.abs(before.astype(np.int16) - after.astype(np.int16))
            assert delta.max() <= 2, f"a pixel moved by {delta.max()} levels, not GPU noise"
            assert delta.mean() < 0.05, f"mean difference {delta.mean():.4f} levels"

    def test_reset_takes_a_seed(self):
        """The harness seeds per episode, `config.seed + episode`, not per env."""
        env = PracticeTrack(seed=0)
        try:
            _started(env)
            env.reset(seed=41)
            assert env.seed == 41
            drawn = env.rng.random()
            env.reset(seed=41)
            assert env.rng.random() == drawn
        finally:
            env.close()


@pytest.mark.render
class TestInfo:
    def test_progress_is_positive_going_forwards(self):
        env = PracticeTrack()
        try:
            _started(env)
            info = _drive(env, FLAT_OUT, 80).info
            assert info["progress_m"] > 0.0
            assert info["speed_mps"] > 5.0
            assert info["on_track"]
            assert info["off_track_fraction"] == 0.0
            assert info["off_track"] is False
        finally:
            env.close()

    def test_the_car_starts_behind_the_line_on_an_out_lap(self):
        """The clock must not start at boot: the run-up is not charged to lap one."""
        env = PracticeTrack()
        try:
            _started(env)
            info = _step(env, FLAT_OUT).info
            assert info["on_out_lap"]
            assert info["lap_elapsed_s"] == 0.0
            assert info["lap_count"] == 0
            assert info["lap_complete"] is False
        finally:
            env.close()

    def test_the_clock_starts_when_the_car_crosses_the_line(self):
        # Track limits off: flat out from the grid runs wide at the first corner, and this
        # test is about the lap clock, not about how well a held throttle drives.
        env = PracticeTrack(track_limit=0.0)
        try:
            _started(env)
            for _ in range(600):
                info = _step(env, FLAT_OUT).info
                if not info["on_out_lap"]:
                    break
            assert not info["on_out_lap"], "never reached the start line"
            assert _drive(env, FLAT_OUT, 20).info["lap_elapsed_s"] > 0.0
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
            for episode in range(2):
                frame, _ = env.reset(seed=episode)
                eye.reset()  # the real eye keeps state between frames; episodes must not
                while True:
                    result = _step(env, policy(eye.encode(frame)))
                    frame = result.frame
                    if result.terminated or result.truncated:
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
        with pytest.raises(RuntimeError, match="closed"):
            env.render()

    @pytest.mark.render
    def test_it_is_a_context_manager(self):
        with PracticeTrack() as env:
            _started(env)
            env.step(FLAT_OUT)
        with pytest.raises(RuntimeError, match="closed"):
            env.step(FLAT_OUT)


@pytest.mark.render
class TestThePenalty:
    """Seconds owed for leaving the circuit, published every step."""

    def test_a_clean_run_owes_nothing(self):
        env = PracticeTrack()
        try:
            _started(env)
            result = _drive(env, FLAT_OUT, 120)
            assert result.info["penalty_s"] == 0.0
            assert result.info["excursions"] == 0
        finally:
            env.close()

    def test_the_penalty_grows_while_the_car_is_off(self):
        """Visible as it is earned rather than appearing at the line, so a driver can see
        what a mistake is costing them while they are still making it."""
        env = PracticeTrack()
        try:
            _started(env)
            _drive(env, FLAT_OUT, 120)
            lock = ControlVector(steer=1.0, throttle=1.0, brake=0.0)
            owed = []
            for _ in range(200):
                owed.append(_step(env, lock).info["penalty_s"])
            assert max(owed) > 0.0, "full lock at speed never left the circuit"
            assert owed == sorted(owed), "a penalty already earned cannot be given back"
        finally:
            env.close()

    def test_the_charge_is_made_on_the_depth_the_env_measures(self):
        """The wiring, not the formula -- the curve from depth to seconds is pinned
        deterministically in test_penalty.py. With the time term switched off the total is
        the peak of each excursion, so it can be checked against the depths actually seen
        without depending on how the car happens to slide."""
        env = PracticeTrack(penalty_weights=PenaltyWeights(peak_weight=1.0, time_weight=0.0))
        try:
            _started(env)
            _drive(env, FLAT_OUT, 120)
            lock = ControlVector(steer=1.0, throttle=1.0, brake=0.0)
            deepest = 0.0
            for _ in range(200):
                deepest = max(deepest, _step(env, lock).info["off_track_fraction"])
            assert deepest > 0.0, "full lock at speed never left the circuit"
            # Equal for a single excursion, more if the car rejoined and went off again.
            assert env.penalty.total_seconds == pytest.approx(deepest) or (
                env.penalty.total_seconds > deepest
            )
        finally:
            env.close()

    def test_weights_are_configurable(self):
        env = PracticeTrack(penalty_weights=PenaltyWeights(peak_weight=0.0, time_weight=0.0))
        try:
            _started(env)
            _drive(env, FLAT_OUT, 120)
            lock = ControlVector(steer=1.0, throttle=1.0, brake=0.0)
            result = _drive(env, lock, 200)
            assert result.info["off_track_fraction"] > 0.0, "the car did go off"
            assert result.info["penalty_s"] == 0.0, "but these weights charge nothing"
        finally:
            env.close()

    def test_reset_clears_what_was_owed(self):
        env = PracticeTrack()
        try:
            _started(env)
            _drive(env, FLAT_OUT, 120)
            _drive(env, ControlVector(steer=1.0, throttle=1.0, brake=0.0), 200)
            assert env.penalty.total_seconds > 0.0
            _, info = env.reset()
            assert info["penalty_s"] == 0.0
            assert info["excursions"] == 0
        finally:
            env.close()


@pytest.mark.render
class TestSegmentTiming:
    """Which part of the circuit the car is on, and how long it spent there."""

    def test_the_car_starts_in_a_segment(self):
        env = PracticeTrack()
        try:
            _, info = _started(env)
            assert info["segment"] is not None
            assert info["segment"] in {s.name for s in env.segment_timer.segments}
        finally:
            env.close()

    def test_the_segment_clock_runs(self):
        env = PracticeTrack()
        try:
            _started(env)
            first = _drive(env, FLAT_OUT, 5).info["segment_elapsed_s"]
            later = _drive(env, FLAT_OUT, 5).info["segment_elapsed_s"]
            assert later > first
        finally:
            env.close()

    def test_driving_on_completes_a_segment_and_reports_its_time(self):
        """The signal GH-17's reward shaping and the delta display both read. A segment
        that never completes looks exactly like a car that never moved."""
        env = PracticeTrack()
        try:
            _started(env)
            completed = [_step(env, FLAT_OUT).info for _ in range(600)]
            completed = [info for info in completed if info["segment_complete"] is not None]
            assert completed, "600 steps flat out and no segment boundary was crossed"
            assert all(info["segment_time"] > 0.0 for info in completed)
            assert all(isinstance(info["segment_clean"], bool) for info in completed)
        finally:
            env.close()

    def test_a_segment_is_clean_exactly_when_no_step_inside_it_went_off(self):
        """Sticky and accurate. A corner cut at its entry must not come back clean by the
        exit, and a segment driven entirely on the circuit must not be marked dirty by the
        previous one's mistake -- both would quietly decide which times set records."""
        env = PracticeTrack()
        try:
            _started(env)
            threshold = env.penalty.weights.minimum_depth
            saw_off = False
            checked = 0
            for _ in range(900):
                info = _step(env, FLAT_OUT).info
                saw_off = saw_off or info["off_track_fraction"] > threshold
                if info["segment_complete"] is not None:
                    assert info["segment_clean"] is not saw_off
                    checked += 1
                    # The completing step is on the boundary, so its state carries into
                    # the segment that just started.
                    saw_off = info["off_track_fraction"] > threshold
            assert checked, "no segment completed, so nothing was actually checked"
        finally:
            env.close()

    def test_the_segment_clock_restarts_at_a_boundary(self):
        env = PracticeTrack()
        try:
            _started(env)
            for _ in range(600):
                info = _step(env, FLAT_OUT).info
                if info["segment_complete"] is not None:
                    assert info["segment_elapsed_s"] < info["segment_time"]
                    return
            pytest.fail("no segment boundary was crossed")
        finally:
            env.close()

    def test_going_off_marks_the_segment_dirty(self):
        env = PracticeTrack()
        try:
            _started(env)
            _drive(env, FLAT_OUT, 120)
            result = _drive(env, ControlVector(steer=1.0, throttle=1.0, brake=0.0), 200)
            assert result.info["off_track_fraction"] > 0.0
            assert result.info["segment_dirty"] is True
        finally:
            env.close()

    def test_reset_puts_the_car_back_in_the_first_segment(self):
        env = PracticeTrack()
        try:
            _started(env)
            _drive(env, FLAT_OUT, 300)
            _, info = env.reset()
            assert info["segment_elapsed_s"] == 0.0
            assert info["segment_dirty"] is False
        finally:
            env.close()
