"""Neuron-count versus frame-time for the live central-complex module.

The budget is one environment frame. At :data:`~fly_driver.interface.FRAME_RATE_HZ`
that is 20 ms, and it has to hold the renderer, the eye, this module, and the policy.
A size whose own step already exceeds that budget cannot be the live width. The curve
times :meth:`~fly_driver.brains.central_complex.CentralComplexBrain.step` only, which
is the term that grows with neuron count.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from fly_driver.brains.blank import blank_trainable_parameter_count
from fly_driver.brains.central_complex import CentralComplexBrain
from fly_driver.interface import FRAME_RATE_HZ

__all__ = ["FrameTimePoint", "frame_budget_ms", "sweep_frame_times"]


def frame_budget_ms(frame_rate_hz: float = FRAME_RATE_HZ) -> float:
    """Milliseconds available for one frame at ``frame_rate_hz``."""
    if not math.isfinite(frame_rate_hz) or frame_rate_hz <= 0:
        raise ValueError(f"frame_rate_hz must be a positive finite number, got {frame_rate_hz}")
    return 1000.0 / float(frame_rate_hz)


@dataclass(frozen=True)
class FrameTimePoint:
    """One neuron count on the frame-time curve, with the parameter counts beside it."""

    neuron_count: int
    input_dim: int
    output_dim: int
    constrained_trainable: int
    blank_trainable: int
    median_frame_ms: float
    p95_frame_ms: float
    frame_budget_ms: float

    @property
    def meets_frame_budget(self) -> bool:
        """Whether the median brain step fits inside one environment frame."""
        return self.median_frame_ms <= self.frame_budget_ms


def sweep_frame_times(
    neuron_counts: Sequence[int],
    *,
    input_dim: int,
    output_dim: int | None = None,
    steps: int = 40,
    warmup: int = 10,
    frame_rate_hz: float = FRAME_RATE_HZ,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> tuple[FrameTimePoint, ...]:
    """Time one forward step across neuron counts.

    Args:
        neuron_counts: Widths to build, in the order to report them.
        input_dim: Eye feature count fed to every size.
        output_dim: Descending width. ``None`` lets each size pick its own default.
        steps: Timed steps per size, after ``warmup``.
        warmup: Untimed steps, so the first call's allocation is not the median.
        frame_rate_hz: Integration rate passed to each module. Also sets the budget.
        seed: Wiring seed. The same seed is used at every width.
        device: Where each module runs.

    Returns:
        One :class:`FrameTimePoint` per neuron count.

    Raises:
        ValueError: If ``neuron_counts`` is empty or a step count is not positive.
    """
    if not neuron_counts:
        raise ValueError("neuron_counts must list at least one width")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"warmup must be a non-negative integer, got {warmup!r}")
    budget = frame_budget_ms(frame_rate_hz)
    points: list[FrameTimePoint] = []
    for neuron_count in neuron_counts:
        brain = CentralComplexBrain(
            input_dim,
            neuron_count,
            output_dim,
            frame_rate_hz=frame_rate_hz,
            seed=seed,
            device=device,
        )
        median_ms, p95_ms = _time_steps(
            brain, steps=steps, warmup=warmup, input_dim=input_dim, seed=seed
        )
        points.append(
            FrameTimePoint(
                neuron_count=brain.neuron_count,
                input_dim=brain.input_dim,
                output_dim=brain.output_dim,
                constrained_trainable=brain.trainable_parameter_count(),
                blank_trainable=blank_trainable_parameter_count(
                    brain.input_dim, brain.neuron_count
                ),
                median_frame_ms=median_ms,
                p95_frame_ms=p95_ms,
                frame_budget_ms=budget,
            )
        )
    return tuple(points)


def _time_steps(
    brain: CentralComplexBrain, *, steps: int, warmup: int, input_dim: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    brain.reset()
    for _ in range(warmup):
        brain.step(_drive(rng, input_dim))
    if brain.voltage.is_cuda:
        torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(steps):
        features = _drive(rng, input_dim)
        if brain.voltage.is_cuda:
            torch.cuda.synchronize()
        started = time.perf_counter()
        brain.step(features)
        if brain.voltage.is_cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    return _percentile(samples, 0.50), _percentile(samples, 0.95)


def _drive(rng: np.random.Generator, input_dim: int) -> np.ndarray:
    return rng.standard_normal(input_dim).astype(np.float32)


def _percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]
