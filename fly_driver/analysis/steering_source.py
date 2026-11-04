"""Where a driver's steering chatter comes from, stage by stage (GH-104).

Three ways of feeding a trained driver tell its causes apart:

- **Closed loop:** it drives the car, so what it sees depends on its own steering.
- **Frozen frame:** it sees one image repeated. Anything that still moves is inside the eye or
  the brain, not in the car or the scene.
- **Shadow:** a smooth script drives while the driver only watches. If it still flips, the flips
  don't need the car to answer its own steering. Its frames come from a smooth drive it never
  trained on, so read this one as suggestive.

Torch is only needed to read the policy's raw output, so it is imported where that happens and
the numbers can be tested without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from fly_driver.analysis.channel_overrides import scripted_steer
from fly_driver.analysis.steering_chatter import steering_stats
from fly_driver.interface import ControlVector

__all__ = [
    "StageRecorder",
    "StageReport",
    "lag1_autocorrelation",
    "power_above",
    "relative_step",
    "run_closed_loop",
    "run_frozen_frame",
    "run_shadow",
    "stage_report",
]

#: Frames ignored at the start of each measurement, while the eye and brain warm up: 0.5 s.
WARM_UP_FRAMES = 25

#: The speed the shadow drive holds: gentle, so the script stays on the track.
_SHADOW_SPEED_MPS = 45.0


def lag1_autocorrelation(values: npt.ArrayLike) -> float:
    """Mean lag-1 autocorrelation over the columns of ``(frames, dims)`` that move.

    Near 1 is a signal that changes little between frames. Near 0 is frame-to-frame noise. A 1-D
    input counts as one column. NaN if nothing moves.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    centred = array - array.mean(axis=0)
    energy = (centred**2).sum(axis=0)
    moving = energy > 1e-12
    if not moving.any():
        return float("nan")
    return float(np.mean((centred[1:] * centred[:-1]).sum(axis=0)[moving] / energy[moving]))


def relative_step(values: npt.ArrayLike) -> float:
    """Mean frame-to-frame change of a ``(frames, dims)`` vector, over its mean length."""
    array = np.asarray(values, dtype=np.float64)
    steps = np.linalg.norm(np.diff(array, axis=0), axis=1)
    return float(steps.mean() / max(np.linalg.norm(array, axis=1).mean(), 1e-12))


def power_above(signal: npt.ArrayLike, frame_rate_hz: float, cutoff_hz: float) -> float:
    """Share of a signal's power above ``cutoff_hz``, leaving out its mean. NaN if flat."""
    array = np.asarray(signal, dtype=np.float64)
    power = np.abs(np.fft.rfft(array - array.mean())) ** 2
    frequencies = np.fft.rfftfreq(array.size, 1.0 / frame_rate_hz)
    total = float(power[1:].sum())
    return float(power[frequencies > cutoff_hz].sum() / total) if total > 1e-12 else float("nan")


class StageRecorder:
    """Runs a driver's eye, brain and policy by hand, keeping each stage's output.

    Args:
        driver: A :class:`~fly_driver.drivers.DirectDriveAgent` with a brain and an
            :class:`~fly_driver.policies.mlp_policy.MlpPolicy`.
    """

    def __init__(self, driver: Any) -> None:
        self.driver = driver
        self.eye_outputs: list[npt.NDArray[np.float64]] = []
        self.brain_outputs: list[npt.NDArray[np.float64]] = []
        self.raw_actions: list[npt.NDArray[np.float64]] = []

    def reset(self, seed: int | None = None) -> None:
        """Start an episode and forget what was recorded."""
        self.driver.reset(seed=seed)
        self.forget()

    def forget(self) -> None:
        """Clear the recordings, leaving the driver's state alone."""
        self.eye_outputs, self.brain_outputs, self.raw_actions = [], [], []

    def act(self, frame: Any) -> ControlVector:
        """One frame through every stage. Returns the clipped mean, as evaluation would."""
        import torch

        eye_output = self.driver.eye.encode(frame)
        brain_output = self.driver.brain.step(eye_output)
        policy = self.driver.policy
        with torch.no_grad():
            mean, _ = policy(torch.as_tensor(brain_output, device=policy.device)[None])
        raw = mean[0].cpu().numpy().astype(np.float64)
        self.eye_outputs.append(np.asarray(eye_output, dtype=np.float64))
        self.brain_outputs.append(np.asarray(brain_output, dtype=np.float64))
        self.raw_actions.append(raw)
        return ControlVector.clipped(*raw.tolist())


@dataclass(frozen=True)
class StageReport:
    """What each stage did over one measurement.

    Args:
        label: Which measurement.
        seconds: Length, after the warm-up.
        flips_per_s: Clipped steering flips.
        mean_abs_steer: Mean clipped ``abs(steer)``.
        full_lock_share: Share of frames at full lock.
        raw_steer_median: Median ``abs`` of the raw steering, where the car's range is 1.
        raw_steer_p95: Its 95th percentile.
        raw_steer_mean: Mean signed raw steering, which shows a lean to one side.
        lag1_eye: Eye output's lag-1 autocorrelation.
        lag1_brain: Brain output's.
        lag1_raw_steer: Raw steering's.
        relative_step_eye: Eye output's change per frame, relative to its size.
        relative_step_brain: Brain output's.
        power_above_5hz: Share of the clipped steering's power above 5 Hz.
    """

    label: str
    seconds: float
    flips_per_s: float
    mean_abs_steer: float
    full_lock_share: float
    raw_steer_median: float
    raw_steer_p95: float
    raw_steer_mean: float
    lag1_eye: float
    lag1_brain: float
    lag1_raw_steer: float
    relative_step_eye: float
    relative_step_brain: float
    power_above_5hz: float


def stage_report(recorder: StageRecorder, label: str, frame_rate_hz: float) -> StageReport:
    """Summarise a recorder, skipping the warm-up.

    Raises:
        ValueError: If fewer than 10 frames remain.
    """
    raw = np.asarray(recorder.raw_actions)[WARM_UP_FRAMES:]
    if raw.shape[0] < 10:
        raise ValueError(f"{label}: only {raw.shape[0]} frames after the warm-up")
    eye = np.asarray(recorder.eye_outputs)[WARM_UP_FRAMES:]
    brain = np.asarray(recorder.brain_outputs)[WARM_UP_FRAMES:]
    steer = np.clip(raw[:, 0], -1.0, 1.0)
    stats = steering_stats(steer, frame_rate_hz)
    return StageReport(
        label=label,
        seconds=stats.seconds,
        flips_per_s=stats.flips_per_s,
        mean_abs_steer=stats.mean_abs_steer,
        full_lock_share=stats.full_lock_share,
        raw_steer_median=float(np.median(np.abs(raw[:, 0]))),
        raw_steer_p95=float(np.percentile(np.abs(raw[:, 0]), 95)),
        raw_steer_mean=float(raw[:, 0].mean()),
        lag1_eye=lag1_autocorrelation(eye),
        lag1_brain=lag1_autocorrelation(brain),
        lag1_raw_steer=lag1_autocorrelation(raw[:, 0]),
        relative_step_eye=relative_step(eye),
        relative_step_brain=relative_step(brain),
        power_above_5hz=power_above(steer, frame_rate_hz, 5.0),
    )


def run_closed_loop(env: Any, recorder: StageRecorder, *, seed: int, max_frames: int) -> None:
    """The driver drives until the episode ends or ``max_frames``."""
    frame, _ = env.reset(seed=seed)
    recorder.reset(seed=seed)
    for _ in range(max_frames):
        frame, _, terminated, truncated, _ = env.step(recorder.act(frame))
        if terminated or truncated:
            break


def run_frozen_frame(
    env: Any, recorder: StageRecorder, *, seed: int, drive_first: int, frames: int
) -> None:
    """The driver sees one image repeated, taken after driving for ``drive_first`` frames."""
    frame, _ = env.reset(seed=seed)
    recorder.reset(seed=seed)
    for _ in range(drive_first):
        frame, _, terminated, truncated, _ = env.step(recorder.act(frame))
        if terminated or truncated:
            break
    recorder.reset(seed=seed)
    for _ in range(frames):
        recorder.act(frame)


def run_shadow(env: Any, recorder: StageRecorder, *, seed: int, max_frames: int) -> None:
    """A script drives gently while the driver only watches."""
    frame, info = env.reset(seed=seed)
    recorder.reset(seed=seed)
    for _ in range(max_frames):
        recorder.act(frame)  # its answer is thrown away
        throttle = 1.0 if float(info.get("speed_mps", 0.0)) < _SHADOW_SPEED_MPS else 0.0
        control = ControlVector(steer=scripted_steer(env, info), throttle=throttle, brake=0.0)
        frame, _, terminated, truncated, info = env.step(control)
        if terminated or truncated:
            break
