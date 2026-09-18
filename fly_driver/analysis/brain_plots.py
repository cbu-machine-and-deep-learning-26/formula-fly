"""Panels for watching a fly brain run (GH-23).

Drawing only. Each function takes a matplotlib ``Axes`` and the data to put in it, so
the viewer in ``scripts/brain_live.py`` stays wiring and the panels can be tested
without a window -- the same split ``fly_driver.analysis.hex_plots`` and
``scripts/flyvis_eye_live.py`` already use for the eye.

matplotlib is imported lazily, so ``import fly_driver.analysis`` works without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from fly_driver.brains.benchmark import BIOLOGICAL_MS_PER_FRAME, ThroughputResult

__all__ = [
    "BACKEND_COLOURS",
    "draw_backend_comparison",
    "draw_budget",
    "draw_dt_sweep",
    "draw_raster",
    "draw_rate",
]

#: One colour per backend, used everywhere so the eye can follow a backend between
#: panels without reading a legend twice.
BACKEND_COLOURS = {"brian2": "#3b7dd8", "torch": "#d8733b"}

_OVER_BUDGET = "#c8443c"
_UNDER_BUDGET = "#3fa14a"
_SPIKE = "#e8e8ec"


def _colour_for(label: str) -> str:
    """The colour for a backend label like ``brian2`` or ``torch:cpu``."""
    return BACKEND_COLOURS.get(label.split(":")[0], "#888888")


def draw_raster(
    axes: Any,
    neuron_indices: Sequence[np.ndarray],
    spike_times_ms: Sequence[np.ndarray],
    *,
    num_neurons: int,
    window_ms: float,
    now_ms: float,
) -> None:
    """Scrolling spike raster: one dot per spike, neuron against time.

    This is the panel that makes a dead network obvious. A brain that has saturated
    goes solid, one that has fallen silent goes empty, and both look exactly like a
    healthy one in a table of milliseconds.

    Args:
        axes: Where to draw.
        neuron_indices: Per-frame arrays of spiking neuron indices.
        spike_times_ms: Per-frame arrays of spike times, same shapes.
        num_neurons: Network size, for the y axis.
        window_ms: How much history to show.
        now_ms: Current simulation time, the right-hand edge.
    """
    axes.clear()
    axes.set_facecolor("#15161a")
    if neuron_indices:
        indices = np.concatenate(list(neuron_indices))
        times = np.concatenate(list(spike_times_ms))
        keep = times >= now_ms - window_ms
        axes.plot(
            times[keep],
            indices[keep],
            linestyle="none",
            marker=".",
            markersize=1.4,
            color=_SPIKE,
            alpha=0.85,
        )
    axes.set_xlim(max(0.0, now_ms - window_ms), max(window_ms, now_ms))
    axes.set_ylim(0, max(1, num_neurons))
    axes.set_ylabel("neuron")
    axes.set_title("spikes", loc="left", fontsize=10)
    axes.tick_params(labelbottom=False)


def draw_rate(
    axes: Any,
    times_ms: Sequence[float],
    rates_hz: Sequence[float],
    *,
    window_ms: float,
    now_ms: float,
) -> None:
    """Population firing rate over time, with the plausible band shaded.

    `AGENTS.md` §11 asks for spike rates in plausible ranges. A fly's neurons idle at
    a few hertz; the shaded band is 0.1-20 Hz, and a trace that leaves it means the
    numbers beside it are measuring something that is not a fly.
    """
    axes.clear()
    axes.axhspan(0.1, 20.0, color="#2a7d3f", alpha=0.13, lw=0)
    if times_ms:
        axes.plot(times_ms, rates_hz, color="#8fd17a", lw=1.3)
        axes.annotate(
            f"{rates_hz[-1]:.2f} Hz",
            xy=(0.99, 0.9),
            xycoords="axes fraction",
            ha="right",
            fontsize=11,
            color="#8fd17a",
        )
    axes.set_xlim(max(0.0, now_ms - window_ms), max(window_ms, now_ms))
    axes.set_ylim(bottom=0)
    axes.set_xlabel("simulation time (ms)")
    axes.set_ylabel("rate (Hz)")
    axes.set_title("population rate", loc="left", fontsize=10)


def draw_budget(
    axes: Any,
    wall_ms: Sequence[float],
    *,
    budget_ms: float,
    window: int = 200,
) -> None:
    """Wall-clock cost per frame against the budget the brain has to fit.

    The whole decision in one panel: the frame is 20 ms, the eye and the practice
    track have already spent theirs, and what is left is ``budget_ms``. Green under
    it, red over it.
    """
    axes.clear()
    recent = list(wall_ms)[-window:]
    axes.axhspan(0, budget_ms, color=_UNDER_BUDGET, alpha=0.15, lw=0)
    axes.axhline(budget_ms, color=_UNDER_BUDGET, lw=1.2, ls="--")
    axes.axhline(BIOLOGICAL_MS_PER_FRAME, color="#8a8a94", lw=0.9, ls=":", label="whole frame")
    if recent:
        latest = recent[-1]
        colour = _UNDER_BUDGET if latest <= budget_ms else _OVER_BUDGET
        axes.plot(range(len(recent)), recent, color=colour, lw=1.3)
        axes.annotate(
            f"{latest:.1f} ms",
            xy=(0.99, 0.88),
            xycoords="axes fraction",
            ha="right",
            fontsize=15,
            color=colour,
        )
        axes.annotate(
            f"budget {budget_ms:.1f} ms",
            xy=(0.99, 0.74),
            xycoords="axes fraction",
            ha="right",
            fontsize=9,
            color="#a8a8b2",
        )
    axes.set_xlim(0, max(window, len(recent)))
    axes.set_ylim(0, max(BIOLOGICAL_MS_PER_FRAME * 1.4, max(recent, default=0) * 1.2))
    axes.set_ylabel("ms per frame")
    axes.set_title("cost vs budget", loc="left", fontsize=10)
    axes.tick_params(labelbottom=False)


def draw_backend_comparison(
    axes: Any, results: Sequence[ThroughputResult], *, budget_ms: float
) -> None:
    """Both backends as the network grows.

    The two backends have opposite scaling, and which one wins depends entirely on how
    it is driven. Brian2 propagates from the neurons that spiked, so its cost tracks
    activity and it wins on big networks -- but it charges a fixed 33-52 ms of setup per
    ``run()`` call, which is the whole cost when something steps it one frame at a time.
    The PyTorch version multiplies the whole weight matrix every step, so its cost tracks
    size and runs away on a big network, but it charges nothing per call.
    """
    axes.clear()
    axes.axhline(budget_ms, color=_UNDER_BUDGET, lw=1.1, ls="--")
    if not results:
        axes.annotate(
            "measuring...",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            color="#8a8a94",
            fontsize=10,
        )
    else:
        by_backend: dict[str, list[ThroughputResult]] = {}
        for result in results:
            by_backend.setdefault(result.backend, []).append(result)
        for label, group in by_backend.items():
            group = sorted(group, key=lambda item: item.num_neurons)
            axes.plot(
                [item.num_neurons for item in group],
                [item.wall_ms_per_frame for item in group],
                marker="o",
                markersize=4,
                lw=1.4,
                color=_colour_for(label),
                label=label,
            )
        axes.legend(fontsize=8, frameon=False)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("neurons")
    axes.set_ylabel("ms per frame")
    axes.set_title("backend vs network size", loc="left", fontsize=10)


def draw_dt_sweep(axes: Any, results: Sequence[ThroughputResult]) -> None:
    """What coarsening the timestep buys, and what it costs.

    Cost is proportional to steps per frame, so ``dt`` is a bigger lever than either
    backend -- but the synaptic delay is 1.8 ms and the refractory period 2.2 ms, so
    a coarse ``dt`` quantises both. The firing rate on the right axis is the price:
    when it starts climbing, the speed is no longer free.
    """
    axes.clear()
    if not results:
        axes.annotate(
            "measuring...",
            xy=(0.5, 0.5),
            xycoords="axes fraction",
            ha="center",
            color="#8a8a94",
            fontsize=10,
        )
        axes.set_title("timestep", loc="left", fontsize=10)
        return

    ordered = sorted(results, key=lambda item: item.dt_ms)
    steps = [item.dt_ms for item in ordered]
    axes.plot(
        steps,
        [item.wall_ms_per_frame for item in ordered],
        marker="o",
        markersize=4,
        lw=1.4,
        color="#3b7dd8",
        label="cost",
    )
    axes.set_xlabel("dt (ms)")
    axes.set_ylabel("ms per frame", color="#3b7dd8")
    axes.tick_params(axis="y", labelcolor="#3b7dd8")

    twin = axes.twinx()
    twin.plot(
        steps,
        [item.spike_rate_hz for item in ordered],
        marker="s",
        markersize=4,
        lw=1.2,
        ls="--",
        color="#c8a13c",
        label="rate",
    )
    twin.set_ylabel("rate (Hz)", color="#c8a13c")
    twin.tick_params(axis="y", labelcolor="#c8a13c")
    axes.set_title("timestep: speed bought, accuracy spent", loc="left", fontsize=10)
