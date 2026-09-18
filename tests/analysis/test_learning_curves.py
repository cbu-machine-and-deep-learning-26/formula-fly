"""Tests for learning-curve extraction and steps-to-threshold."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from fly_driver.analysis import learning_curve, steps_to_threshold
from fly_driver.envs import DummyTrackEnv
from fly_driver.policies import ConstantAgent
from fly_driver.training import EvalConfig, evaluate


def test_steps_to_threshold_finds_first_crossing() -> None:
    """The first step count whose value reaches the threshold is returned."""
    steps = [0, 1000, 2000, 3000]
    assert steps_to_threshold(steps, [-5.0, 1.0, 10.0, 12.0], 10.0) == 2000
    assert steps_to_threshold(steps, [-5.0, 1.0, 2.0, 3.0], 10.0) is None
    assert steps_to_threshold(steps, [np.nan, 11.0, 12.0, 13.0], 10.0) == 1000
    assert steps_to_threshold([], [], 1.0) is None


def test_steps_to_threshold_validates_inputs() -> None:
    """Mismatched lengths and decreasing steps are errors."""
    with pytest.raises(ValueError, match="equal length"):
        steps_to_threshold([0, 1], [1.0], 0.0)
    with pytest.raises(ValueError, match="non-decreasing"):
        steps_to_threshold([10, 5], [1.0, 2.0], 0.0)


def test_learning_curve_orders_reports_by_env_steps() -> None:
    """Reports are sorted by training steps and None metrics become NaN."""
    base = EvalConfig(episodes=1, max_steps=5)
    reports = [
        evaluate(DummyTrackEnv, ConstantAgent(), replace(base, env_steps=steps))
        for steps in (2000, 0, 1000)
    ]

    steps, returns = learning_curve(reports, "mean_return")
    np.testing.assert_array_equal(steps, [0, 1000, 2000])
    assert returns.shape == (3,) and np.all(np.isfinite(returns))
    _, lap_times = learning_curve(reports, "mean_lap_time_s")
    assert np.all(np.isnan(lap_times))
    assert steps_to_threshold(steps, returns, returns.max()) == 0

    with pytest.raises(ValueError, match="metric"):
        learning_curve(reports, "reward")
