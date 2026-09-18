"""Watch a fly brain run against the frame budget (GH-23).

A viewer, not a training component -- the brain-side counterpart to
``scripts/flyvis_eye_live.py``. Wiring only: the stepping lives in
:mod:`fly_driver.brains.stream` and the drawing in
:mod:`fly_driver.analysis.brain_plots`, per AGENTS.md §14.3.

Five panels. Three update every frame:

* **spikes** -- a scrolling raster. The panel that makes a dead network obvious: a
  saturated brain goes solid, a silent one goes empty, and in a table of milliseconds
  both look exactly like a healthy one.
* **population rate** -- with the plausible 0.1-20 Hz band shaded (`AGENTS.md` §11).
* **cost vs budget** -- wall-clock milliseconds to simulate each 20 ms frame, against
  the ~10 ms the brain gets once the eye's 7.9 ms and the practice track's 2.2 ms come
  out of the frame. Green under, red over.

Two fill in only when you ask, with **m**, because each measurement blocks the window
while it runs: **backend vs network size** and **timestep**. They are measured stepped
one frame at a time, like the gauge beside them, so their numbers are several times the
batched ones ``scripts/brain_throughput.py`` prints.

Run it in the brain environment (see ``docs/running-the-stacks.md``)::

    python scripts/brain_live.py                          # torch, 1,000 neurons: fits
    python scripts/brain_live.py --backend brian2         # the one that does not
    python scripts/brain_live.py --neurons 3000
    python scripts/brain_live.py --no-display --frames 100   # headless check

The defaults are the configuration that actually holds the budget. **Brian2 costs
36-59 ms per frame whatever the network size**, because it re-prepares itself on every
``run()`` call, so ``--backend brian2`` is worth looking at once and is not a pleasant
thing to leave running.

Keys: **space** pauses, **m** takes one measurement, **r** restarts the brain, **q**
quits.

It exits ``0`` with a ``SKIP:`` line when matplotlib, a simulator, or the connectome
data is missing, so base CI needs none of them.
"""

from __future__ import annotations

import argparse
from collections import deque

import numpy as np

from fly_driver.brains.benchmark import BenchmarkConfig
from fly_driver.brains.connectome import ConnectomeNotFoundError, load_subnetwork

#: Network sizes the backend panel walks through, one per press of ``m``.
SWEEP_SIZES = (1_000, 3_000, 10_000)

#: Timesteps the timestep panel walks through.
SWEEP_TIMESTEPS_MS = (0.1, 0.25, 0.5, 1.0)

#: Biology per measurement. Short, because the window is frozen while one runs.
SWEEP_BIOLOGICAL_MS = 60.0

#: Share of each redraw interval spent simulating. The rest is the redraw itself, which
#: is the larger half and is matplotlib's, not ours. Measured here: ~70 ms to draw the
#: five panels, against 5.5 ms to simulate one frame of a 1,000-neuron brain on torch.
SIMULATION_SHARE = 0.3


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectome-dir", default=None, help="overrides FLY_CONNECTOME_DIR")
    parser.add_argument("--release", default="630", choices=("630", "783"))
    parser.add_argument("--neurons", type=int, default=1_000, help="network size to run")
    parser.add_argument("--backend", choices=("brian2", "torch"), default="torch")
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--dt", type=float, default=0.5, help="timestep in ms")
    parser.add_argument(
        "--window-ms", type=float, default=500.0, help="how much history the raster shows"
    )
    parser.add_argument(
        "--frames-per-draw",
        type=int,
        default=0,
        help=(
            "simulation frames per redraw. 0 picks a count that keeps simulated time "
            "roughly level with the wall clock"
        ),
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=150,
        help=(
            "milliseconds between redraws. A full redraw of these five panels costs "
            "~70 ms here, most of it tick labels, so the viewer costs roughly half a "
            "core at this rate. Raise it to spend less, lower it for smoother scrolling"
        ),
    )
    parser.add_argument("--frames", type=int, default=0, help="stop after this many frames")
    parser.add_argument("--no-display", action="store_true", help="run headless")
    parser.add_argument("--figsize", default="12,6.5", help="figure size, W,H in inches")
    parser.add_argument("--dpi", type=int, default=80, help="figure dpi; lower is cheaper")
    return parser.parse_args(argv)


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - a viewer is a wiring loop
    args = _parse_args(argv)

    try:
        import matplotlib
    except ImportError:
        return _skip("this viewer needs matplotlib: pip install matplotlib")
    if args.no_display:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        # Loaded once. Every subset below is sliced from this in memory; re-reading the
        # 86 MB parquet per measurement is what made the first version of this viewer
        # unusable.
        whole = load_subnetwork(args.connectome_dir, release=args.release)
    except (ConnectomeNotFoundError, ImportError) as exc:
        return _skip(str(exc))
    network = whole.head(args.neurons)

    from fly_driver.analysis.brain_plots import BrainPanels
    from fly_driver.brains.stream import SpikeStream, measure_in_loop

    config = BenchmarkConfig(dt_ms=args.dt)
    try:
        stream = SpikeStream(network, args.backend, config=config, device=args.device)
    except (ImportError, RuntimeError) as exc:
        return _skip(str(exc))
    stream.warm_up()

    width, height = (float(value) for value in args.figsize.split(","))
    plt.style.use("dark_background")
    # No constrained layout: it re-solves the whole arrangement on every draw, which
    # measured 87 ms a frame against 71 ms without it. One tight_layout at the end does.
    figure = plt.figure(figsize=(width, height), dpi=args.dpi)
    if figure.canvas.manager is not None:
        figure.canvas.manager.set_window_title("fly brain, live")
    panels = BrainPanels(
        figure,
        num_neurons=network.num_neurons,
        budget_ms=config.budget_ms,
        window_ms=args.window_ms,
    )

    buffers = {
        "times": np.empty(0, dtype=float),
        "neurons": np.empty(0, dtype=np.int64),
    }
    rate_times: deque = deque(maxlen=400)
    rate_values: deque = deque(maxlen=400)
    wall_ms: deque = deque(maxlen=300)
    size_results: list = []
    dt_results: list = []
    size_plan = deque((size, name) for size in SWEEP_SIZES for name in ("brian2", "torch"))
    dt_plan = deque(SWEEP_TIMESTEPS_MS)
    state = {"paused": False, "frames": 0, "note": "press m to measure"}

    def measure_one() -> None:
        """Take one measurement. Blocks the window, which is why it is on a key."""
        if size_plan:
            size, backend = size_plan.popleft()
            subset, dt_ms, kind = whole.head(size), args.dt, "size"
        elif dt_plan:
            dt_ms, backend, kind = dt_plan.popleft(), args.backend, "dt"
            subset = network
        else:
            state["note"] = "all measured"
            return
        state["note"] = f"measuring {backend} at {subset.num_neurons:,} neurons, dt {dt_ms}"
        panels.set_status(state["note"])
        figure.canvas.draw_idle()
        figure.canvas.flush_events()
        try:
            point = BenchmarkConfig(dt_ms=dt_ms, biological_ms=SWEEP_BIOLOGICAL_MS)
            result = measure_in_loop(subset, backend, config=point, device=args.device)
        except Exception as exc:  # pragma: no cover - a failed point is not fatal
            state["note"] = f"measurement failed: {exc}"
            return
        (size_results if kind == "size" else dt_results).append(result)
        panels.update_sweep(size_results, dt_results)
        remaining = len(size_plan) + len(dt_plan)
        state["note"] = f"m for the next of {remaining}" if remaining else "all measured"

    def on_key(event) -> None:
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "m":
            measure_one()
        elif event.key == "r":
            stream.reset()
            stream.warm_up()
            buffers["times"] = np.empty(0, dtype=float)
            buffers["neurons"] = np.empty(0, dtype=np.int64)
            rate_times.clear()
            rate_values.clear()
        elif event.key in ("q", "escape"):
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)

    def frames_this_tick() -> int:
        """How many simulation frames to run before the next redraw.

        Zero means auto: enough that simulated time keeps pace with the wall clock, so
        the raster scrolls at roughly life speed rather than at whatever the backend
        happens to manage. Capped, or a fast backend spends the whole interval
        simulating and the window stops answering.
        """
        if args.frames_per_draw > 0:
            return args.frames_per_draw
        recent = list(wall_ms)[-10:]
        if not recent:
            return 1
        average = sum(recent) / len(recent)
        # Only part of the interval is the brain's: a full redraw of five panels costs
        # ~70 ms here, and sizing the simulation to the whole interval and then drawing
        # on top of it is how a viewer ends up using more than a core.
        return max(1, min(20, int(SIMULATION_SHARE * args.interval_ms / max(average, 0.5))))

    def update(_tick: int) -> None:
        if state["paused"]:
            return
        for _ in range(frames_this_tick()):
            activity = stream.advance()
            state["frames"] += 1
            if activity.num_spikes:
                buffers["times"] = np.concatenate([buffers["times"], activity.spike_times_ms])
                buffers["neurons"] = np.concatenate([buffers["neurons"], activity.neuron_indices])
            rate_times.append(activity.sim_time_ms)
            rate_values.append(activity.rate_hz)
            wall_ms.append(activity.wall_ms)

        # Drop what has scrolled off, or the raster grows without bound and the viewer
        # gets slower the longer it is watched.
        if buffers["times"].size:
            keep = buffers["times"] >= stream.sim_time_ms - args.window_ms
            buffers["times"] = buffers["times"][keep]
            buffers["neurons"] = buffers["neurons"][keep]

        panels.update_live(
            spike_times_ms=buffers["times"],
            spike_neurons=buffers["neurons"],
            rate_times_ms=rate_times,
            rates_hz=rate_values,
            wall_ms=wall_ms,
            now_ms=stream.sim_time_ms,
        )
        panels.set_status(
            f"{stream.label}  |  {stream.num_neurons:,} neurons, "
            f"{network.num_synapses:,} synapses  |  dt {args.dt} ms  |  "
            f"frame {state['frames']}  |  {state['note']}"
        )

    if args.no_display:
        for tick in range(max(1, args.frames)):
            update(tick)
        measure_one()
        cost = sum(wall_ms) / len(wall_ms) if wall_ms else 0.0
        rate = sum(rate_values) / len(rate_values) if rate_values else 0.0
        verdict = "inside" if cost <= config.budget_ms else "over"
        print(
            f"{stream.label}: {state['frames']} frames, {cost:.1f} ms/frame "
            f"({verdict} the {config.budget_ms:.1f} ms budget), {rate:.2f} Hz/neuron"
        )
        return 0

    from matplotlib.animation import FuncAnimation

    animation = FuncAnimation(
        figure,
        update,
        interval=args.interval_ms,
        cache_frame_data=False,
        save_count=0,
        frames=args.frames or None,
    )
    figure.tight_layout()
    figure._fly_animation = animation  # keep a reference or it is garbage collected
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
