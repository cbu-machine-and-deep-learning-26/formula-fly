"""Each way the steering-source diagnostic feeds a driver (GH-104), on stubs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fly_driver.analysis.steering_source import (  # noqa: E402
    WARM_UP_FRAMES,
    StageRecorder,
    run_closed_loop,
    run_frozen_frame,
    stage_report,
)
from fly_driver.interface import ControlVector  # noqa: E402

pytestmark = pytest.mark.usefixtures("cpu_default_device")


class _Policy(torch.nn.Module):
    """Raw steering 3.0 on every frame, far outside the car's range."""

    device = torch.device("cpu")

    def forward(self, batch: Any) -> tuple[Any, Any]:
        del batch
        return torch.tensor([[3.0, 0.5, 0.0]]), torch.zeros(1)


class _Stage:
    """Stands in for the eye and the brain: the frame number, then a count of its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def encode(self, frame: Any) -> np.ndarray:
        self.calls += 1
        return np.array([float(frame), float(self.calls)], dtype=np.float32)

    def step(self, features: Any) -> np.ndarray:
        return np.asarray(features, dtype=np.float32)


class _Driver:
    def __init__(self) -> None:
        self.eye = _Stage()
        self.brain = _Stage()
        self.policy = _Policy()
        self.reset_seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)
        self.eye.reset()


class _Env:
    """Serves frames 0, 1, 2, ... and ends after ``length`` steps."""

    frame_rate_hz = 50.0

    def __init__(self, length: int) -> None:
        self.length = length
        self.taken = 0
        self.actions: list[ControlVector] = []

    def reset(self, *, seed: int | None = None) -> tuple[int, dict[str, Any]]:
        del seed
        self.taken = 0
        return 0, {}

    def step(self, action: ControlVector) -> tuple[int, float, bool, bool, dict[str, Any]]:
        self.actions.append(action)
        self.taken += 1
        return self.taken, 0.0, self.taken >= self.length, False, {}


def test_recorder_clips_what_it_returns_but_keeps_the_raw_output() -> None:
    recorder = StageRecorder(_Driver())
    recorder.reset(seed=0)
    control = recorder.act(1)
    assert control.steer == 1.0
    assert recorder.raw_actions[0][0] == pytest.approx(3.0)
    assert len(recorder.eye_outputs) == len(recorder.brain_outputs) == 1


def test_closed_loop_ends_with_the_episode() -> None:
    env, recorder = _Env(length=5), StageRecorder(_Driver())
    run_closed_loop(env, recorder, seed=2, max_frames=100)
    assert env.taken == 5
    assert len(recorder.raw_actions) == 5
    assert recorder.driver.reset_seeds == [2]


def test_frozen_frame_repeats_one_image_after_the_drive_and_forgets_the_drive() -> None:
    env, recorder = _Env(length=100), StageRecorder(_Driver())
    run_frozen_frame(env, recorder, seed=0, drive_first=7, frames=4)
    assert env.taken == 7
    assert [output[0] for output in recorder.eye_outputs] == [7.0] * 4


def test_stage_report_skips_the_warm_up_and_reads_the_raw_steering() -> None:
    env, recorder = _Env(length=1000), StageRecorder(_Driver())
    run_closed_loop(env, recorder, seed=0, max_frames=WARM_UP_FRAMES + 50)
    report = stage_report(recorder, "test", 50.0)
    assert report.seconds == pytest.approx(1.0)
    assert report.flips_per_s == 0.0
    assert report.mean_abs_steer == 1.0
    assert report.full_lock_share == 1.0
    assert report.raw_steer_median == pytest.approx(3.0)
    assert report.raw_steer_mean == pytest.approx(3.0)


def test_stage_report_refuses_too_few_frames() -> None:
    env, recorder = _Env(length=1000), StageRecorder(_Driver())
    run_closed_loop(env, recorder, seed=0, max_frames=WARM_UP_FRAMES + 2)
    with pytest.raises(ValueError, match="after the warm-up"):
        stage_report(recorder, "test", 50.0)
