"""Panels for watching a fly brain run (GH-23).

Drawing only, so the viewer in ``scripts/brain_live.py`` stays wiring -- the same split
``fly_driver.analysis.hex_plots`` and ``scripts/flyvis_eye_live.py`` use for the eye.

**Artists are created once and updated in place.** The obvious way to write this is a
draw function per panel that clears its axes and replots, and it will lock up the
machine it runs on: matplotlib keeps everything ever added to a figure, so clearing and
replotting five panels every frame is unbounded work, and ``twinx()`` inside a redraw
adds a whole new axes on every call. Hence :class:`BrainPanels`, which builds every
artist in its constructor and afterwards only ever calls ``set_data``.

matplotlib is imported lazily, so ``import fly_driver.analysis`` works without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from fly_driver.brains.benchmark import BIOLOGICAL_MS_PER_FRAME, ThroughputResult

__all__ = ["BACKEND_COLOURS", "MAX_RASTER_POINTS", "BrainPanels"]

#: One colour per backend, used in every panel so a backend can be followed between
#: them without reading a legend twice.
BACKEND_COLOURS = {"brian2": "#3b7dd8", "torch": "#d8733b"}

#: Most spikes the raster will draw. A busy network produces far more than a screen can
#: show, and drawing them all is the difference between a viewer and a freeze.
MAX_RASTER_POINTS = 12_000

_OVER_BUDGET = "#c8443c"
_UNDER_BUDGET = "#3fa14a"
_SPIKE = "#e8e8ec"
_RATE = "#8fd17a"
_MUTED = "#8a8a94"


def _colour_for(label: str) -> str:
    """The colour for a backend label like ``brian2`` or ``torch:cpu``."""
    return BACKEND_COLOURS.get(label.split(":")[0], "#888888")


class BrainPanels:
    """The five panels, built once and updated in place.

    Args:
        figure: The matplotlib figure to lay out on.
        num_neurons: Network size, for the raster's y axis.
        budget_ms: Wall-clock milliseconds per frame the brain may use.
        window_ms: How much simulation history the live panels show.
    """

    def __init__(
        self, figure: Any, *, num_neurons: int, budget_ms: float, window_ms: float
    ) -> None:
        self.figure = figure
        self.num_neurons = num_neurons
        self.budget_ms = budget_ms
        self.window_ms = window_ms

        grid = figure.add_gridspec(2, 3)
        self.raster_axes = figure.add_subplot(grid[0, 0:2])
        self.budget_axes = figure.add_subplot(grid[0, 2])
        self.rate_axes = figure.add_subplot(grid[1, 0])
        self.backend_axes = figure.add_subplot(grid[1, 1])
        self.timestep_axes = figure.add_subplot(grid[1, 2])

        self._build_raster()
        self._build_budget()
        self._build_rate()
        self._build_backend()
        self._build_timestep()
        self._cap_ticks()
        self.status = figure.suptitle("", fontsize=11)

    def _cap_ticks(self) -> None:
        """Ask for few ticks, because tick rendering is most of a redraw.

        Measured on this figure: 77 ms per full redraw with matplotlib's default tick
        density, 57 ms with four per axis. The panels are read at a glance -- none of
        them is a plot anybody measures off -- so the ticks are the cheapest thing to
        give up and the one that buys the most.
        """
        from matplotlib.ticker import MaxNLocator

        for axes in (
            self.raster_axes,
            self.budget_axes,
            self.rate_axes,
            self.backend_axes,
            self.timestep_axes,
            self.rate_axis,
        ):
            for axis in (axes.xaxis, axes.yaxis):
                if axis.get_scale() == "linear":
                    axis.set_major_locator(MaxNLocator(4))

    # -- construction ----------------------------------------------------------

    def _build_raster(self) -> None:
        axes = self.raster_axes
        axes.set_facecolor("#15161a")
        (self._raster,) = axes.plot(
            [], [], linestyle="none", marker=".", markersize=1.4, color=_SPIKE, alpha=0.85
        )
        axes.set_ylim(0, max(1, self.num_neurons))
        axes.set_xlim(0, self.window_ms)
        axes.set_ylabel("neuron")
        axes.set_title("spikes", loc="left", fontsize=10)
        axes.tick_params(labelbottom=False)

    def _build_budget(self) -> None:
        axes = self.budget_axes
        axes.axhspan(0, self.budget_ms, color=_UNDER_BUDGET, alpha=0.15, lw=0)
        axes.axhline(self.budget_ms, color=_UNDER_BUDGET, lw=1.2, ls="--")
        axes.axhline(BIOLOGICAL_MS_PER_FRAME, color=_MUTED, lw=0.9, ls=":")
        (self._budget,) = axes.plot([], [], color=_UNDER_BUDGET, lw=1.3)
        self._budget_text = axes.annotate(
            "",
            xy=(0.97, 0.86),
            xycoords="axes fraction",
            ha="right",
            fontsize=15,
            color=_UNDER_BUDGET,
        )
        axes.annotate(
            f"budget {self.budget_ms:.1f} ms",
            xy=(0.97, 0.72),
            xycoords="axes fraction",
            ha="right",
            fontsize=9,
            color=_MUTED,
        )
        axes.set_ylim(0, BIOLOGICAL_MS_PER_FRAME * 2)
        axes.set_ylabel("ms per frame")
        axes.set_title("cost vs budget", loc="left", fontsize=10)
        axes.tick_params(labelbottom=False)

    def _build_rate(self) -> None:
        axes = self.rate_axes
        axes.axhspan(0.1, 20.0, color="#2a7d3f", alpha=0.13, lw=0)
        (self._rate,) = axes.plot([], [], color=_RATE, lw=1.3)
        self._rate_text = axes.annotate(
            "",
            xy=(0.97, 0.88),
            xycoords="axes fraction",
            ha="right",
            fontsize=11,
            color=_RATE,
        )
        axes.set_xlim(0, self.window_ms)
        axes.set_ylim(0, 10.0)
        axes.set_xlabel("simulation time (ms)")
        axes.set_ylabel("rate (Hz)")
        axes.set_title("population rate", loc="left", fontsize=10)

    def _build_backend(self) -> None:
        axes = self.backend_axes
        axes.axhline(self.budget_ms, color=_UNDER_BUDGET, lw=1.1, ls="--")
        self._backend_lines: dict[str, Any] = {}
        self._backend_hint = axes.annotate(
            "press m to measure",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            color=_MUTED,
            fontsize=9,
        )
        axes.set_xscale("log")
        axes.set_yscale("log")
        axes.set_xlabel("neurons")
        axes.set_ylabel("ms per frame")
        axes.set_title("backend vs network size", loc="left", fontsize=10)

    def _build_timestep(self) -> None:
        axes = self.timestep_axes
        # Created once. twinx() inside a redraw adds a new axes on every call, and a
        # few hundred of those is what turns this window into a frozen machine.
        self.rate_axis = axes.twinx()
        (self._timestep_cost,) = axes.plot(
            [], [], marker="o", markersize=4, lw=1.4, color="#3b7dd8"
        )
        (self._timestep_rate,) = self.rate_axis.plot(
            [], [], marker="s", markersize=4, lw=1.2, ls="--", color="#c8a13c"
        )
        self._timestep_hint = axes.annotate(
            "press m to measure",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            color=_MUTED,
            fontsize=9,
        )
        axes.set_xlabel("dt (ms)")
        axes.set_ylabel("ms per frame", color="#3b7dd8")
        axes.tick_params(axis="y", labelcolor="#3b7dd8")
        self.rate_axis.set_ylabel("rate (Hz)", color="#c8a13c")
        self.rate_axis.tick_params(axis="y", labelcolor="#c8a13c")
        axes.set_title("timestep: speed bought, accuracy spent", loc="left", fontsize=10)

    # -- updates ---------------------------------------------------------------

    def update_live(
        self,
        *,
        spike_times_ms: np.ndarray,
        spike_neurons: np.ndarray,
        rate_times_ms: Sequence[float],
        rates_hz: Sequence[float],
        wall_ms: Sequence[float],
        now_ms: float,
    ) -> None:
        """Refresh the three per-frame panels. Called on every draw tick."""
        if len(spike_times_ms) > MAX_RASTER_POINTS:
            keep = np.linspace(0, len(spike_times_ms) - 1, MAX_RASTER_POINTS).astype(int)
            spike_times_ms, spike_neurons = spike_times_ms[keep], spike_neurons[keep]
        self._raster.set_data(spike_times_ms, spike_neurons)
        left = max(0.0, now_ms - self.window_ms)
        self.raster_axes.set_xlim(left, max(left + self.window_ms, now_ms))

        self._rate.set_data(list(rate_times_ms), list(rates_hz))
        self.rate_axes.set_xlim(left, max(left + self.window_ms, now_ms))
        if rates_hz:
            self._rate_text.set_text(f"{rates_hz[-1]:.2f} Hz")
            self.rate_axes.set_ylim(0, max(10.0, max(rates_hz) * 1.2))

        costs = list(wall_ms)
        if costs:
            self._budget.set_data(range(len(costs)), costs)
            latest = costs[-1]
            colour = _UNDER_BUDGET if latest <= self.budget_ms else _OVER_BUDGET
            self._budget.set_color(colour)
            self._budget_text.set_text(f"{latest:.1f} ms")
            self._budget_text.set_color(colour)
            self.budget_axes.set_xlim(0, max(50, len(costs)))
            self.budget_axes.set_ylim(0, max(BIOLOGICAL_MS_PER_FRAME * 1.4, max(costs) * 1.2))

    def update_sweep(
        self,
        size_results: Sequence[ThroughputResult],
        dt_results: Sequence[ThroughputResult],
    ) -> None:
        """Refresh the two measured panels. Only called when a measurement lands."""
        if size_results:
            self._backend_hint.set_text("")
            grouped: dict[str, list[ThroughputResult]] = {}
            for result in size_results:
                grouped.setdefault(result.backend, []).append(result)
            for label, results in grouped.items():
                ordered = sorted(results, key=lambda item: item.num_neurons)
                line = self._backend_lines.get(label)
                if line is None:
                    (line,) = self.backend_axes.plot(
                        [],
                        [],
                        marker="o",
                        markersize=4,
                        lw=1.4,
                        color=_colour_for(label),
                        label=label,
                    )
                    self._backend_lines[label] = line
                    self.backend_axes.legend(fontsize=8, frameon=False)
                line.set_data(
                    [item.num_neurons for item in ordered],
                    [item.wall_ms_per_frame for item in ordered],
                )
            self.backend_axes.relim()
            self.backend_axes.autoscale_view()

        if dt_results:
            self._timestep_hint.set_text("")
            ordered = sorted(dt_results, key=lambda item: item.dt_ms)
            steps = [item.dt_ms for item in ordered]
            self._timestep_cost.set_data(steps, [item.wall_ms_per_frame for item in ordered])
            self._timestep_rate.set_data(steps, [item.spike_rate_hz for item in ordered])
            for axes in (self.timestep_axes, self.rate_axis):
                axes.relim()
                axes.autoscale_view()

    def set_status(self, text: str) -> None:
        """Set the line above the panels."""
        self.status.set_text(text)
