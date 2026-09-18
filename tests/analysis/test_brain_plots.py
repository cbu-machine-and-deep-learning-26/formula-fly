"""Brain viewer panels (GH-23).

These are drawing code, so what matters is not what they look like. It is that they
survive the states a live viewer actually hits -- no data yet, a silent network, a
single frame -- and that repeating an update does not accumulate anything.

That last one is not hypothetical. The first version of this module was a draw function
per panel that cleared its axes and replotted, and one of them called ``twinx()`` on
every redraw. matplotlib keeps everything ever added to a figure, so the figure grew a
new axes thirty times a second and the viewer locked up the machine within seconds of
opening. The tests in :class:`TestNothingAccumulates` exist for that.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.brains.benchmark import ThroughputResult

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fly_driver.analysis.brain_plots import BACKEND_COLOURS, BrainPanels  # noqa: E402


@pytest.fixture
def panels():
    figure = plt.figure(figsize=(12, 6), layout="constrained")
    yield BrainPanels(figure, num_neurons=500, budget_ms=9.9, window_ms=500.0)
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


def _live(panels, *, times=(), neurons=(), rates=(), costs=(), now=20.0):
    panels.update_live(
        spike_times_ms=np.asarray(times, dtype=float),
        spike_neurons=np.asarray(neurons, dtype=np.int64),
        rate_times_ms=[float(i) * 20.0 for i in range(len(rates))],
        rates_hz=list(rates),
        wall_ms=list(costs),
        now_ms=now,
    )


class TestEmptyStates:
    """Everything is empty on the first frame, and the measured panels stay empty
    until someone presses a key."""

    def test_the_first_frame_draws_nothing_and_does_not_raise(self, panels):
        _live(panels, now=0.0)

    def test_a_silent_network_is_fine(self, panels):
        _live(panels, rates=[0.0], costs=[4.2])

    def test_the_measured_panels_say_how_to_fill_them(self, panels):
        hints = [text.get_text() for text in panels.backend_axes.texts]
        hints += [text.get_text() for text in panels.timestep_axes.texts]
        assert any("press m" in hint for hint in hints)


class TestNothingAccumulates:
    """The freeze this module was rewritten to prevent."""

    def test_the_figure_never_grows_another_axes(self, panels):
        """``twinx()`` inside a redraw is what killed the first version: one new axes
        per frame, thirty times a second, all of them still being rendered."""
        before = len(panels.figure.axes)
        for tick in range(50):
            _live(panels, rates=[1.0], costs=[5.0], now=20.0 * tick)
            panels.update_sweep([_result("torch:cpu", 1_000, 5.5)], [_result("brian2", 500, 30.0)])
        assert len(panels.figure.axes) == before

    def test_repeating_an_update_never_adds_another_line(self, panels):
        counts = []
        for _ in range(20):
            _live(panels, times=[1.0, 2.0], neurons=[1, 2], rates=[1.0], costs=[5.0])
            panels.update_sweep([_result("brian2", 1_000, 35.9)], [])
            counts.append(sum(len(axes.lines) for axes in panels.figure.axes))
        assert len(set(counts[1:])) == 1, f"line count kept changing: {counts}"

    def test_a_busy_network_is_subsampled_rather_than_all_drawn(self, panels):
        """A second of a busy brain is far more spikes than a screen has pixels, and
        drawing them all is the difference between a viewer and a freeze."""
        from fly_driver.analysis.brain_plots import MAX_RASTER_POINTS

        many = MAX_RASTER_POINTS * 3
        _live(
            panels,
            times=np.linspace(0, 500, many),
            neurons=np.arange(many) % 500,
            now=500.0,
        )
        drawn = len(panels.raster_axes.lines[0].get_xdata())
        assert drawn <= MAX_RASTER_POINTS


class TestWithData:
    def test_the_gauge_shows_the_latest_cost_and_goes_red_over_budget(self, panels):
        _live(panels, rates=[1.0], costs=[5.0, 6.0, 12.3])
        assert any("12.3" in text.get_text() for text in panels.budget_axes.texts)
        over = panels._budget.get_color()
        _live(panels, rates=[1.0], costs=[5.0, 6.0, 4.1])
        assert panels._budget.get_color() != over, "the gauge never goes back to green"

    def test_both_backends_get_their_own_line(self, panels):
        panels.update_sweep(
            [
                _result("brian2", 1_000, 35.9),
                _result("brian2", 3_000, 54.5),
                _result("torch:cpu", 1_000, 5.5),
                _result("torch:cpu", 3_000, 12.2),
            ],
            [],
        )
        labels = {line.get_label() for line in panels.backend_axes.lines}
        assert {"brian2", "torch:cpu"} <= labels

    def test_a_backend_keeps_one_colour_across_panels(self):
        """Five panels, and no room for five legends."""
        assert BACKEND_COLOURS["brian2"] != BACKEND_COLOURS["torch"]

    def test_the_timestep_panel_plots_cost_and_rate_together(self, panels):
        """Speed bought and accuracy spent belong on one panel, or the timestep looks
        free."""
        panels.update_sweep([], [_result("brian2", 3_000, 54.5, dt_ms=dt) for dt in (0.1, 0.5)])
        assert len(panels._timestep_cost.get_xdata()) == 2
        assert len(panels._timestep_rate.get_xdata()) == 2

    def test_the_rate_trace_shows_the_current_value(self, panels):
        _live(panels, rates=[1.0, 2.5])
        assert any("2.50" in text.get_text() for text in panels.rate_axes.texts)

    def test_the_raster_window_follows_the_clock(self, panels):
        _live(panels, times=[1400.0], neurons=[3], now=1500.0)
        left, right = panels.raster_axes.get_xlim()
        assert left == pytest.approx(1000.0)
        assert right == pytest.approx(1500.0)
