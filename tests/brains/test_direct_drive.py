"""Vision + brain direct drive, without flyvis and without a trained lap (GH-23)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from fly_driver.brains.central_complex import CentralComplexBrain  # noqa: E402
from fly_driver.brains.constants import VOLTAGE_LIMIT  # noqa: E402
from fly_driver.drivers import DirectDriveAgent  # noqa: E402
from fly_driver.envs.dummy_track import DummyTrackEnv  # noqa: E402
from fly_driver.eyes.pixel_eye import PixelEye  # noqa: E402
from fly_driver.interface import FRAME_SHAPE, ControlVector, Driver  # noqa: E402

pytestmark = pytest.mark.usefixtures("cpu_default_device")


class MeanPolicy:
    """Steer from the mean descending activation. A stand-in head, not a trained policy."""

    def __init__(self, feature_dim: int) -> None:
        self.feature_dim = feature_dim
        self.seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> None:
        self.seeds.append(seed)

    def act(self, features: np.ndarray) -> ControlVector:
        steer = float(np.clip(float(features.mean()), -1.0, 1.0))
        return ControlVector(steer=steer, throttle=0.3, brake=0.0)


def _driver(frame_shape: tuple[int, int, int] = FRAME_SHAPE) -> DirectDriveAgent:
    eye = PixelEye(frame_shape=frame_shape, downsample=8)
    brain = CentralComplexBrain(eye.feature_dim, neuron_count=48, output_dim=8, seed=0)
    return DirectDriveAgent(eye, MeanPolicy(brain.output_dim), brain=brain)


class TestDirectDriveWithABrain:
    def test_the_driver_is_sized_to_the_brain_output(self):
        driver = _driver()
        assert isinstance(driver, Driver)
        assert driver.feature_dim == 8
        assert driver.frame_shape == FRAME_SHAPE

    def test_a_long_rollout_on_camera_frames_stays_finite(self):
        driver = _driver()
        driver.reset(seed=0)
        rng = np.random.default_rng(0)
        for _ in range(400):
            frame = rng.integers(0, 256, size=FRAME_SHAPE, dtype=np.uint8)
            control = driver.act(frame)
            assert isinstance(control, ControlVector)
            assert np.all(np.isfinite(control.to_array()))
        activity = driver.brain.activity()
        assert np.all(np.isfinite(activity))
        assert np.max(np.abs(activity)) <= VOLTAGE_LIMIT

    def test_the_dummy_track_accepts_the_control(self):
        shape = (64, 64, 3)
        driver = _driver(shape)
        env = DummyTrackEnv(frame_shape=shape, max_steps=30, wind_std=0.0)
        frame, _ = env.reset(seed=0)
        driver.reset(seed=0)
        steps = 0
        done = False
        while not done:
            control = driver.act(frame)
            frame, reward, terminated, truncated, _ = env.step(control)
            assert np.isfinite(reward)
            done = terminated or truncated
            steps += 1
        assert steps >= 1

    @pytest.mark.render
    def test_the_practice_track_steps_under_vision_plus_brain(self):
        from fly_driver.envs.practice_track import PracticeTrack

        env = PracticeTrack(max_steps=5)
        try:
            try:
                frame, _ = env.reset(seed=0)
            except Exception as exc:  # pragma: no cover - depends on the machine
                pytest.skip(f"no offscreen GL context: {exc}")
            driver = _driver(tuple(env.frame_shape))
            driver.reset(seed=0)
            for _ in range(5):
                control = driver.act(frame)
                frame, reward, terminated, truncated, info = env.step(control)
                assert isinstance(control, ControlVector)
                assert np.isfinite(reward)
                assert not info["diverged"]
                if terminated or truncated:
                    break
            assert np.all(np.isfinite(env.data.qpos))
        finally:
            env.close()
