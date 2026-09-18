"""Watch a fly brain run against the frame budget (GH-23).

A viewer, not a training component -- the brain-side counterpart to
``scripts/flyvis_eye_live.py``. Wiring only: the stepping lives in
:mod:`fly_driver.brains.stream` and the panels in
:mod:`fly_driver.analysis.brain_plots`, per AGENTS.md §14.3.

Five panels, two of them live every frame and three filling in as the sweep runs:

* **spikes** -- a scrolling raster. The panel that makes a dead network obvious: a
  saturated brain goes solid, a silent one goes empty, and in a table of milliseconds
  both look exactly like a healthy one.
* **population rate** -- with the plausible 0.1-20 Hz band shaded (`AGENTS.md` §11).
* **cost vs budget** -- wall-clock milliseconds to simulate each 20 ms frame, against
  the ~10 ms the brain gets once the eye's 7.9 ms and the practice track's 2.2 ms come
  out of the frame. Green under, red over.
* **backend vs network size** -- Brian2 and PyTorch as the network grows.
* **timestep** -- what coarsening ``dt`` buys, with the firing rate beside it as the
  accuracy price.

The three sweep panels are measured *while you watch*, one point at a time, on short
runs -- so they are noisier than ``scripts/brain_throughput.py``. They are also measured
**stepped one frame at a time**, like the gauge beside them and like the real loop, which
is why their numbers are several times higher than the batched ones that script prints.

Run it in the brain environment (see ``docs/running-the-stacks.md``)::

    python scripts/brain_live.py
    python scripts/brain_live.py --neurons 10000 --backend torch
    python scripts/brain_live.py --no-display --frames 100   # headless check

Keys: space pauses, ``r`` restarts the brain, ``q`` quits.

It exits ``0`` with a ``SKIP:`` line when matplotlib, a simulator, or the connectome
data is missing, so base CI needs none of them.
"""

from __future__ import annotations

import argparse
from collections import deque

from fly_driver.brains.benchmark import BenchmarkConfig
from fly_driver.brains.connectome import ConnectomeNotFoundError, load_subnetwork

#: Network sizes the live sweep walks through for the backend panel.
SWEEP_SIZES = (1_000, 3_000, 10_000)

#: Timesteps the live sweep walks through for the timestep panel.
SWEEP_TIMESTEPS_MS = (0.1, 0.25, 0.5, 1.0)

#: Biology per live sweep point. Short, because the viewer stalls while one runs.
SWEEP_BIOLOGICAL_MS = 60.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectome-dir", default=None, help="overrides FLY_CONNECTOME_DIR")
    parser.add_argument("--release", default="630", choices=("630", "783"))
    parser.add_argument("--neurons", type=int, default=3_000, help="network size to run")
    parser.add_argument("--backend", choices=("brian2", "torch"), default="brian2")
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument("--dt", type=float, default=0.5, help="timestep in ms")
    parser.add_argument(
        "--window-ms", type=float, default=600.0, help="how much history the raster shows"
    )
    parser.add_argument(
        "--measure-every",
        type=int,
        default=25,
        help="frames between live sweep points; 0 leaves the sweep panels empty",
    )
    parser.add_argument("--frames", type=int, default=0, help="stop after this many frames")
    parser.add_argument("--no-display", action="store_true", help="run headless")
    parser.add_argument("--figsize", default="15,8", help="figure size, W,H in inches")
    return parser.parse_args(argv)


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def _sweep_plan(neurons: int, backend: str) -> deque:
    """Measurements to take while the viewer runs, cheapest first."""
    plan: deque = deque()
    for size in SWEEP_SIZES:
        for name in ("brian2", "torch"):
            plan.append(("size", size, None, name))
    for dt_ms in SWEEP_TIMESTEPS_MS:
        plan.append(("dt", neurons, dt_ms, backend))
    return plan


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
        network = load_subnetwork(
            args.connectome_dir, release=args.release, num_neurons=args.neurons
        )
    except (ConnectomeNotFoundError, ImportError) as exc:
        return _skip(str(exc))

    from fly_driver.analysis import brain_plots
    from fly_driver.brains.stream import SpikeStream, measure_in_loop

    config = BenchmarkConfig(dt_ms=args.dt)
    try:
        stream = SpikeStream(network, args.backend, config=config, device=args.device)
    except (ImportError, RuntimeError) as exc:
        return _skip(str(exc))
    stream.warm_up()

    width, height = (float(value) for value in args.figsize.split(","))
    plt.style.use("dark_background")
    figure = plt.figure(figsize=(width, height), layout="constrained")
    figure.canvas.manager.set_window_title("fly brain, live")
    grid = figure.add_gridspec(2, 3)
    axes = {
        "raster": figure.add_subplot(grid[0, 0:2]),
        "budget": figure.add_subplot(grid[0, 2]),
        "rate": figure.add_subplot(grid[1, 0]),
        "backend": figure.add_subplot(grid[1, 1]),
        "dt": figure.add_subplot(grid[1, 2]),
    }

    history_frames = max(1, int(args.window_ms / 20.0))
    spikes_i: deque = deque(maxlen=history_frames)
    spikes_t: deque = deque(maxlen=history_frames)
    rate_times: deque = deque(maxlen=history_frames)
    rate_values: deque = deque(maxlen=history_frames)
    wall_ms: deque = deque(maxlen=400)
    size_results: list = []
    dt_results: list = []
    plan = _sweep_plan(args.neurons, args.backend) if args.measure_every else deque()
    state = {"paused": False, "frames": 0, "measuring": ""}

    def on_key(event) -> None:
        if event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "r":
            stream.__post_init__()
            stream.warm_up()
            spikes_i.clear()
            spikes_t.clear()
        elif event.key in ("q", "escape"):
            plt.close(figure)

    figure.canvas.mpl_connect("key_press_event", on_key)

    def take_one_measurement() -> None:
        """Run a single sweep point. Blocks the viewer, which is why they are short."""
        kind, size, dt_ms, backend = plan.popleft()
        state["measuring"] = f"{backend} at {size:,} neurons"
        try:
            subset = load_subnetwork(args.connectome_dir, release=args.release, num_neurons=size)
            point = BenchmarkConfig(
                dt_ms=dt_ms if dt_ms else args.dt, biological_ms=SWEEP_BIOLOGICAL_MS
            )
            # measure_in_loop, not shiu.measure: the sweep panels have to agree with
            # the gauge beside them, and the gauge times one frame at a time.
            result = measure_in_loop(subset, backend, config=point, device=args.device)
        except Exception as exc:  # pragma: no cover - a failed point is not fatal
            print(f"sweep point failed ({backend}, {size}): {exc}")
            state["measuring"] = ""
            return
        (size_results if kind == "size" else dt_results).append(result)
        state["measuring"] = ""

    def update(_frame: int) -> None:
        if state["paused"]:
            return
        activity = stream.advance()
        spikes_i.append(activity.neuron_indices)
        spikes_t.append(activity.spike_times_ms)
        rate_times.append(activity.sim_time_ms)
        rate_values.append(activity.rate_hz)
        wall_ms.append(activity.wall_ms)
        state["frames"] += 1

        if plan and args.measure_every and state["frames"] % args.measure_every == 0:
            take_one_measurement()

        brain_plots.draw_raster(
            axes["raster"],
            spikes_i,
            spikes_t,
            num_neurons=stream.num_neurons,
            window_ms=args.window_ms,
            now_ms=activity.sim_time_ms,
        )
        brain_plots.draw_rate(
            axes["rate"],
            list(rate_times),
            list(rate_values),
            window_ms=args.window_ms,
            now_ms=activity.sim_time_ms,
        )
        brain_plots.draw_budget(axes["budget"], list(wall_ms), budget_ms=config.budget_ms)
        brain_plots.draw_backend_comparison(
            axes["backend"], size_results, budget_ms=config.budget_ms
        )
        brain_plots.draw_dt_sweep(axes["dt"], dt_results)

        pending = f"  |  measuring {state['measuring']}" if state["measuring"] else ""
        figure.suptitle(
            f"{stream.label}  |  {stream.num_neurons:,} neurons, "
            f"{network.num_synapses:,} synapses  |  dt {args.dt} ms  |  "
            f"frame {state['frames']}{pending}",
            fontsize=11,
        )

    if args.no_display:
        for index in range(max(1, args.frames)):
            update(index)
        rate = sum(rate_values) / len(rate_values) if rate_values else 0.0
        cost = sum(wall_ms) / len(wall_ms) if wall_ms else 0.0
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
        interval=1,
        cache_frame_data=False,
        save_count=0,
        frames=args.frames or None,
    )
    figure._fly_animation = animation  # keep a reference or it is garbage collected
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
