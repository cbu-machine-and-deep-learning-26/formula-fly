"""Shared-contract smoke tests (GH-20): the stages composed, and the seams checked.

The doubles here are deliberately the simplest possible eye and policy -- pooled luminance
and steer-toward-the-bright-side -- so the loop runs without torch, flyvis or a GPU. They
are also real implementations of the protocols, which is the point: anything that satisfies
:class:`~fly_driver.interface.Eye` and :class:`~fly_driver.interface.Policy` can be dropped
into a :class:`~fly_driver.drivers.DirectDriveAgent` and driven by the practice track.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.drivers import DirectDriveAgent, EmbodiedDriveAgent, PassthroughBody
from fly_driver.interface import (
    FEATURE_DTYPE,
    FRAME_SHAPE,
    Body,
    ControlVector,
    Driver,
    Eye,
    Policy,
)

BLOCK = 8


class PooledLuminanceEye:
    """Mean luminance over ``block``-pixel squares, flattened. The simplest eye there is."""

    def __init__(self, frame_shape: tuple[int, int, int] = FRAME_SHAPE, block: int = BLOCK):
        self.frame_shape = frame_shape
        self.block = block
        height, width, _ = frame_shape
        self.feature_dim = (height // block) * (width // block)
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def encode(self, frame: np.ndarray) -> np.ndarray:
        height, width, _ = self.frame_shape
        rows, cols = height // self.block, width // self.block
        luminance = frame[: rows * self.block, : cols * self.block].mean(axis=2) / 255.0
        pooled = luminance.reshape(rows, self.block, cols, self.block).mean(axis=(1, 3))
        return pooled.reshape(-1).astype(FEATURE_DTYPE)


class SteerTowardBrightPolicy:
    """Steer toward whichever half of the view is brighter, at fixed throttle."""

    def __init__(self, feature_dim: int, throttle: float = 0.3):
        self.feature_dim = feature_dim
        self.throttle = throttle
        self.seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> None:
        self.seeds.append(seed)

    def act(self, features: np.ndarray) -> ControlVector:
        side = int(np.sqrt(features.size))
        grid = features.reshape(side, -1)
        left, right = grid[:, : grid.shape[1] // 2].mean(), grid[:, grid.shape[1] // 2 :].mean()
        steer = float(np.clip((right - left) * 4.0, -1.0, 1.0))
        return ControlVector(steer=steer, throttle=self.throttle, brake=0.0)


class ArrayPolicy:
    """A policy head that emits a raw array, the way a network would."""

    def __init__(self, feature_dim: int, output: np.ndarray):
        self.feature_dim = feature_dim
        self.output = output

    def reset(self, seed: int | None = None) -> None:
        pass

    def act(self, features: np.ndarray) -> np.ndarray:
        return self.output


class HalfThrottleBody:
    """A body whose wings only ever deliver half the intended throttle."""

    def __init__(self):
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def actuate(self, intent: ControlVector) -> ControlVector:
        return ControlVector(steer=intent.steer, throttle=intent.throttle / 2, brake=intent.brake)


def _frame(brighter: str = "none", shape: tuple[int, int, int] = FRAME_SHAPE) -> np.ndarray:
    frame = np.full(shape, 100, dtype=np.uint8)
    half = shape[1] // 2
    if brighter == "right":
        frame[:, half:] = 200
    elif brighter == "left":
        frame[:, :half] = 200
    return frame


def _direct() -> DirectDriveAgent:
    eye = PooledLuminanceEye()
    return DirectDriveAgent(eye, SteerTowardBrightPolicy(eye.feature_dim))


class TestTheProtocolsAreCheckable:
    def test_the_doubles_satisfy_eye_and_policy(self):
        eye = PooledLuminanceEye()
        assert isinstance(eye, Eye)
        assert isinstance(SteerTowardBrightPolicy(eye.feature_dim), Policy)

    def test_drivers_satisfy_driver_and_bodies_satisfy_body(self):
        assert isinstance(_direct(), Driver)
        eye = PooledLuminanceEye()
        embodied = EmbodiedDriveAgent(
            eye, SteerTowardBrightPolicy(eye.feature_dim), PassthroughBody()
        )
        assert isinstance(embodied, Driver)
        assert isinstance(PassthroughBody(), Body)
        assert isinstance(HalfThrottleBody(), Body)

    def test_something_missing_a_stage_method_is_not_a_driver(self):
        assert not isinstance(PooledLuminanceEye(), Driver)
        assert not isinstance(PassthroughBody(), Policy)


class TestDirectDrive:
    def test_a_frame_in_gives_a_control_vector_out(self):
        driver = _direct()
        driver.reset(seed=0)
        control = driver.act(_frame("right"))
        assert isinstance(control, ControlVector)
        assert control.steer > 0.0
        assert driver.act(_frame("left")).steer < 0.0
        assert driver.act(_frame()).steer == 0.0

    def test_frame_shape_and_feature_dim_come_from_the_eye(self):
        driver = _direct()
        assert driver.frame_shape == FRAME_SHAPE
        assert driver.feature_dim == (96 // BLOCK) ** 2

    def test_the_wrong_frame_shape_is_refused_not_resized(self):
        """The failure `AGENTS.md` §11 names: a quietly resized frame corrupts the retina."""
        driver = _direct()
        with pytest.raises(ValueError, match="No implicit resize"):
            driver.act(_frame(shape=(48, 48, 3)))
        with pytest.raises(ValueError, match="No implicit resize"):
            driver.act(_frame(shape=(96, 96, 4)))

    def test_the_wrong_frame_dtype_is_refused_not_cast(self):
        driver = _direct()
        with pytest.raises(TypeError, match="No implicit cast"):
            driver.act(_frame().astype(np.float32) / 255.0)

    def test_mismatched_feature_dims_fail_at_construction(self):
        eye = PooledLuminanceEye()
        with pytest.raises(ValueError, match="not wired"):
            DirectDriveAgent(eye, SteerTowardBrightPolicy(eye.feature_dim + 1))

    @pytest.mark.parametrize(
        ("bad", "error"),
        [
            (np.zeros(3, dtype=np.float32), "features"),  # wrong length
            (np.full((96 // BLOCK) ** 2, np.nan, dtype=np.float32), "finite"),
            (np.zeros((96 // BLOCK) ** 2, dtype=np.float64), "float32"),
            (np.zeros((12, 12), dtype=np.float32), "features"),  # not flat
        ],
    )
    def test_a_misbehaving_eye_is_caught_at_the_seam(self, bad, error):
        eye = PooledLuminanceEye()
        eye.encode = lambda frame: bad  # type: ignore[method-assign]
        driver = DirectDriveAgent(eye, SteerTowardBrightPolicy(eye.feature_dim))
        with pytest.raises((ValueError, TypeError), match=error):
            driver.act(_frame())

    def test_a_policy_may_return_an_array_and_it_is_range_checked(self):
        eye = PooledLuminanceEye()
        fine = DirectDriveAgent(eye, ArrayPolicy(eye.feature_dim, np.array([0.5, 1.0, 0.0])))
        assert fine.act(_frame()) == ControlVector(steer=0.5, throttle=1.0, brake=0.0)

        saturated = DirectDriveAgent(eye, ArrayPolicy(eye.feature_dim, np.array([1.5, 0.0, 0.0])))
        with pytest.raises(ValueError, match="clipped"):
            saturated.act(_frame())

    def test_reset_reaches_every_stage_with_the_seed(self):
        eye = PooledLuminanceEye()
        policy = SteerTowardBrightPolicy(eye.feature_dim)
        driver = DirectDriveAgent(eye, policy)
        driver.reset(seed=7)
        driver.reset()
        assert eye.resets == 2
        assert policy.seeds == [7, None]


class TestEmbodiedDrive:
    def test_a_passthrough_body_is_direct_drive(self):
        """Both compositions implement the same contract, and with no body dynamics they
        agree frame for frame -- the ticket's acceptance criterion, asserted."""
        eye = PooledLuminanceEye()
        direct = DirectDriveAgent(eye, SteerTowardBrightPolicy(eye.feature_dim))
        embodied = EmbodiedDriveAgent(
            eye, SteerTowardBrightPolicy(eye.feature_dim), PassthroughBody()
        )
        for kind in ("right", "left", "none"):
            assert embodied.act(_frame(kind)) == direct.act(_frame(kind))

    def test_the_body_shapes_the_control(self):
        eye = PooledLuminanceEye()
        embodied = EmbodiedDriveAgent(
            eye, SteerTowardBrightPolicy(eye.feature_dim), HalfThrottleBody()
        )
        control = embodied.act(_frame("right"))
        assert control.throttle == pytest.approx(0.15)
        assert control.steer > 0.0

    def test_reset_reaches_the_body(self):
        eye = PooledLuminanceEye()
        body = HalfThrottleBody()
        EmbodiedDriveAgent(eye, SteerTowardBrightPolicy(eye.feature_dim), body).reset(seed=1)
        assert body.resets == 1 and eye.resets == 1

    def test_a_body_returning_a_raw_array_is_refused(self):
        """An array has bypassed ControlVector's range check, which is the whole seam."""
        eye = PooledLuminanceEye()
        body = HalfThrottleBody()
        body.actuate = lambda intent: intent.to_array()  # type: ignore[method-assign]
        embodied = EmbodiedDriveAgent(eye, SteerTowardBrightPolicy(eye.feature_dim), body)
        with pytest.raises(TypeError, match="ControlVector"):
            embodied.act(_frame())


class FakeTrack:
    """A frame-emitting env with no physics: enough to close the loop without MuJoCo."""

    frame_shape = FRAME_SHAPE

    def __init__(self):
        self.received: list[ControlVector] = []

    def reset(self, *, seed: int | None = None):
        self.received.clear()
        return _frame("right"), {}

    def step(self, action):
        control = ControlVector.from_any(action)
        self.received.append(control)
        kind = "left" if len(self.received) % 2 else "right"
        return _frame(kind), 0.0, False, len(self.received) >= 10, {}


class TestSmokeTheLoop:
    def test_frame_in_control_out_for_ten_steps_on_a_fake_track(self):
        env = FakeTrack()
        driver = _direct()
        frame, _ = env.reset(seed=0)
        driver.reset(seed=0)
        truncated = False
        while not truncated:
            frame, _, _, truncated, _ = env.step(driver.act(frame))
        assert len(env.received) == 10
        assert all(isinstance(control, ControlVector) for control in env.received)
        assert env.received[0].steer > 0.0 and env.received[1].steer < 0.0

    @pytest.mark.render
    @pytest.mark.parametrize("body", [None, PassthroughBody(), HalfThrottleBody()])
    def test_the_practice_track_can_be_driven_by_either_composition(self, body):
        """The real env, the real head camera, and a driver built only from the contract."""
        from fly_driver.envs.practice_track import PracticeTrack

        env = PracticeTrack(max_steps=60)
        try:
            try:
                frame, _ = env.reset(seed=0)
            except Exception as exc:  # pragma: no cover - depends on the machine
                pytest.skip(f"no offscreen GL context: {exc}")
            eye = PooledLuminanceEye(env.frame_shape)
            policy = SteerTowardBrightPolicy(eye.feature_dim, throttle=0.5)
            driver: Driver = (
                DirectDriveAgent(eye, policy)
                if body is None
                else EmbodiedDriveAgent(eye, policy, body)
            )
            assert driver.frame_shape == env.frame_shape
            driver.reset(seed=0)
            done = False
            steps = 0
            while not done:
                control = driver.act(frame)
                assert isinstance(control, ControlVector)
                frame, reward, terminated, truncated, info = env.step(control)
                assert np.isfinite(reward) and not info["diverged"]
                done = terminated or truncated
                steps += 1
            assert steps == 60
            assert np.all(np.isfinite(env.data.qpos))
        finally:
            env.close()
