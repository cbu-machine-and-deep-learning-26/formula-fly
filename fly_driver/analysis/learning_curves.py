"""Sample efficiency from a series of evaluation reports.

A training run evaluates its policy every so often; each :class:`EvalReport`
carries ``config.env_steps``. Lining those up gives a learning curve, and
sample efficiency is the number of environment steps until a metric first
reaches a threshold (``None`` if it never does).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import numpy.typing as npt

from fly_driver.training.evaluation import CLEAN_CONDITION, EvalReport

__all__ = ["CURVE_METRICS", "learning_curve", "steps_to_threshold"]

CURVE_METRICS = ("mean_return", "lap_completion_rate", "mean_lap_time_s")


def steps_to_threshold(
    env_steps: Sequence[int], values: Sequence[float], threshold: float
) -> int | None:
    """Return the first ``env_steps`` entry whose value reaches ``threshold``.

    Args:
        env_steps: Non-decreasing training step counts.
        values: Metric at each step count; ``NaN`` never counts as reached.
        threshold: Reached when ``value >= threshold``.

    Raises:
        ValueError: If the sequences differ in length or steps decrease.
    """
    steps = np.asarray(env_steps, dtype=np.int64)
    metric = np.asarray(values, dtype=np.float64)
    if steps.shape != metric.shape or steps.ndim != 1:
        raise ValueError("env_steps and values must be 1-D sequences of equal length")
    if np.any(np.diff(steps) < 0):
        raise ValueError("env_steps must be non-decreasing")
    reached = np.flatnonzero(metric >= threshold)
    return int(steps[reached[0]]) if reached.size else None


def learning_curve(
    reports: Iterable[EvalReport],
    metric: str = "mean_return",
    condition: str = CLEAN_CONDITION,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Extract ``(env_steps, metric)`` arrays sorted by ``env_steps``.

    Args:
        reports: Evaluation reports from one training run.
        metric: One of :data:`CURVE_METRICS`; ``None`` values become ``NaN``.
        condition: Which condition's summary to read.
    """
    if metric not in CURVE_METRICS:
        raise ValueError(f"metric must be one of {CURVE_METRICS}, got {metric!r}")
    points = []
    for report in reports:
        value = getattr(report.summary(condition), metric)
        points.append((report.config.env_steps, np.nan if value is None else float(value)))
    points.sort(key=lambda point: point[0])
    steps = np.array([point[0] for point in points], dtype=np.int64)
    values = np.array([point[1] for point in points], dtype=np.float64)
    return steps, values
