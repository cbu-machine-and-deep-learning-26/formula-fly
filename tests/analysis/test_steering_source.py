"""The numbers behind the steering-source diagnostic (GH-104), hand-computed and torch-free."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fly_driver.analysis.steering_source import lag1_autocorrelation, power_above, relative_step


def test_lag1_of_a_hand_computed_series() -> None:
    # Centred it is -1.5, -0.5, 0.5, 1.5. Numerator 0.75 + -0.25 + 0.75 = 1.25, over energy 5.
    assert lag1_autocorrelation([1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.25)


def test_lag1_of_an_alternating_series_is_near_minus_one() -> None:
    assert lag1_autocorrelation(np.tile([1.0, -1.0], 500)) == pytest.approx(-1.0, abs=0.01)


def test_lag1_averages_only_the_columns_that_move() -> None:
    moving = [1.0, 2.0, 3.0, 4.0]
    columns = np.column_stack([moving, np.ones(4)])
    assert lag1_autocorrelation(columns) == pytest.approx(0.25)


def test_lag1_of_a_flat_signal_is_nan() -> None:
    assert math.isnan(lag1_autocorrelation(np.ones(10)))


def test_relative_step_of_a_hand_computed_vector() -> None:
    # (3, 0) to (0, 4) is a step of length 5. The lengths are 3 and 4, so their mean is 3.5.
    values = [[3.0, 0.0], [0.0, 4.0]]
    assert relative_step(values) == pytest.approx(5.0 / 3.5)


def test_power_above_separates_a_slow_wave_from_a_fast_one() -> None:
    frames = np.arange(500) / 50.0
    assert power_above(np.sin(2 * np.pi * 2.0 * frames), 50.0, 5.0) == pytest.approx(0.0, abs=1e-6)
    assert power_above(np.sin(2 * np.pi * 10.0 * frames), 50.0, 5.0) == pytest.approx(1.0, abs=1e-6)


def test_power_above_of_a_flat_signal_is_nan() -> None:
    assert math.isnan(power_above(np.ones(100), 50.0, 5.0))
