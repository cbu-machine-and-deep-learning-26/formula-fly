"""Brain viewer panels (GH-23).

The panels are drawing code, so what is worth testing is that they survive the states a
live viewer actually hits -- no data yet, a silent network, a single frame -- rather
than what they look like. A viewer that raises on the first empty frame is a viewer
nobody sees.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.brains.benchmark import ThroughputResult

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fly_driver.analysis import brain_plots  # noqa: E402


@pytest.fixture
def axes():
    figure, axes = plt.subplots()
    yield axes
    plt.close(figure)


def _result(backend: str, neurons: int, wall_ms: float, dt_ms: float = 0.5):
    return ThroughputResult(
        backend=backend,
        num_neurons=neurons,
        num_synapses=neurons * 4,
        dt_ms=dt_ms,
        wall_ms_per_frame=wall_ms,
        spike_rate_hz=1.2,
        budget_ms=9.9,
    )


class TestEmptyStates:
    """Everything is empty on the first frame, and the sweep panels stay empty for
    several seconds after that."""

    def test_the_raster_survives_having_no_spikes_yet(self, axes):
        brain_plots.draw_raster(axes, [], [], num_neurons=100, window_ms=600, now_ms=0)

    def test_the_raster_survives_a_silent_network(self, axes):
        empty = [np.empty(0, dtype=np.int64)]
        brain_plots.draw_raster(
            axes, empty, [np.empty(0)], num_neurons=100, window_ms=600, now_ms=20
        )

    def test_the_rate_trace_survives_no_history(self, axes):
        brain_plots.draw_rate(axes, [], [], window_ms=600, now_ms=0)

    def test_the_budget_gauge_survives_no_measurements(self, axes):
        brain_plots.draw_budget(axes, [], budget_ms=9.9)

    def test_the_sweep_panels_say_they_are_measuring(self, axes):
        brain_plots.draw_backend_comparison(axes, [], budget_ms=9.9)
        assert any("measuring" in text.get_text() for text in axes.texts)

    def test_the_timestep_panel_says_it_is_measuring(self, axes):
        brain_plots.draw_dt_sweep(axes, [])
        assert any("measuring" in text.get_text() for text in axes.texts)


class TestWithData:
    def test_the_raster_drops_spikes_older_than_the_window(self, axes):
        """Without this the raster keeps every spike of the session and the viewer
        slows to a stop after a minute."""
        old = np.array([1, 2, 3])
        recent = np.array([4, 5])
        brain_plots.draw_raster(
            axes,
            [old, recent],
            [np.full(3, 10.0), np.full(2, 900.0)],
            num_neurons=10,
            window_ms=600,
            now_ms=920,
        )
        drawn = np.concatenate([line.get_ydata() for line in axes.lines])
        assert set(drawn) == {4, 5}

    def test_the_budget_gauge_shows_the_latest_cost(self, axes):
        brain_plots.draw_budget(axes, [5.0, 6.0, 12.3], budget_ms=9.9)
        assert any("12.3" in text.get_text() for text in axes.texts)

    def test_both_backends_get_their_own_line(self, axes):
        results = [
            _result("brian2", 1_000, 35.9),
            _result("brian2", 3_000, 54.5),
            _result("torch:cpu", 1_000, 5.5),
            _result("torch:cpu", 3_000, 12.2),
        ]
        brain_plots.draw_backend_comparison(axes, results, budget_ms=9.9)
        labels = {line.get_label() for line in axes.lines}
        assert {"brian2", "torch:cpu"} <= labels

    def test_a_backend_keeps_one_colour_across_panels(self):
        """The viewer has five panels and no room for five legends."""
        assert brain_plots.BACKEND_COLOURS["brian2"] != brain_plots.BACKEND_COLOURS["torch"]

    def test_the_timestep_panel_plots_cost_and_rate_together(self, axes):
        """Speed bought and accuracy spent belong on the same panel, or the timestep
        looks free."""
        results = [_result("brian2", 3_000, 54.5, dt_ms=dt) for dt in (0.1, 0.5, 1.0)]
        brain_plots.draw_dt_sweep(axes, results)
        assert len(axes.lines) == 1
        assert len(axes.figure.axes) == 2, "the rate axis is missing"

    def test_the_rate_trace_shows_the_current_value(self, axes):
        brain_plots.draw_rate(axes, [0.0, 20.0], [1.0, 2.5], window_ms=600, now_ms=20)
        assert any("2.50" in text.get_text() for text in axes.texts)
