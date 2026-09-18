"""Measure whether a fly brain can keep up with the frame budget (GH-23).

Wiring only: every measurement lives in :mod:`fly_driver.brains`, per AGENTS.md §14.3.

Run it::

    python scripts/brain_throughput.py
    python scripts/brain_throughput.py --neurons 3000 10000 --dt 0.25 0.5 1.0
    python scripts/brain_throughput.py --device cpu cuda      # the lab comparison

It needs the FlyWire connectivity, which is a 370 MB download kept outside this
repository and pointed at with ``FLY_CONNECTOME_DIR``; see
``docs/running-the-stacks.md``. It exits ``0`` with a ``SKIP:`` line when the data or
the optional simulators are absent, the way ``scripts/flyvis_smoke.py`` does, so CI on
the base install stays green.

Two things it prints that matter as much as the timings:

**The Brian2 code generation target.** Brian2 falls back from Cython to plain numpy when
there is no C++ compiler, with only a warning, and the fallback is several times slower
than the backend can actually go.

**The resolved torch device, by name.** ``--device cpu cuda`` measures both on the *same*
machine, which is the only way the comparison means anything -- a CPU number from one box
against a GPU number from another confounds the card with the whole computer.

Each network is timed two ways, because they answer different questions. **stepped** is
one 20 ms frame at a time, the way a brain between an eye and a policy is actually driven,
and it is the number the GH-23 decision rests on. **batched** asks for the whole run in one
call, which is what the lesion sweep and other offline work get. For Brian2 the two differ
by an order of magnitude.
"""

from __future__ import annotations

import argparse
import importlib.util

from fly_driver.brains.benchmark import BenchmarkConfig, ThroughputResult
from fly_driver.brains.connectome import ConnectomeNotFoundError, load_subnetwork

DEFAULT_NEURON_COUNTS = (3_000, 10_000, 40_000)
DEFAULT_TIMESTEPS_MS = (0.1, 0.5)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--connectome-dir", default=None, help="overrides FLY_CONNECTOME_DIR")
    parser.add_argument("--release", default="630", choices=("630", "783"))
    parser.add_argument(
        "--neurons",
        type=int,
        nargs="+",
        default=list(DEFAULT_NEURON_COUNTS),
        help="network sizes to measure; 0 means the whole brain",
    )
    parser.add_argument(
        "--dt",
        type=float,
        nargs="+",
        default=list(DEFAULT_TIMESTEPS_MS),
        help="integration timesteps in ms",
    )
    parser.add_argument("--backend", choices=("brian2", "torch", "both"), default="both")
    parser.add_argument(
        "--device",
        nargs="+",
        default=["auto"],
        help=(
            "torch devices to measure, e.g. `--device cpu cuda`. 'auto' prefers CUDA when "
            "present. Naming cuda explicitly fails rather than silently using the CPU"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("stepped", "batched", "both"),
        default="both",
        help="stepped is one frame at a time (the in-loop cost); batched is one long run",
    )
    parser.add_argument(
        "--biological-ms", type=float, default=100.0, help="biology to time per run"
    )
    return parser.parse_args(argv)


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from fly_driver.brains.stream import measure_in_loop

    modes = ("stepped", "batched") if args.mode == "both" else (args.mode,)
    runs: list[tuple[str, str, str | None]] = []  # (label, backend, device)
    if args.backend in ("brian2", "both"):
        try:
            import brian2  # noqa: F401

            runs.append(("brian2", "brian2", None))
        except ImportError:  # pragma: no cover - depends on the local install
            pass
    if args.backend in ("torch", "both") and importlib.util.find_spec("torch") is not None:
        try:
            from fly_driver.brains.torch_lif import describe_device, resolve_device

            for name in args.device:
                requested = None if name == "auto" else name
                try:
                    resolved = resolve_device(requested)
                except RuntimeError as exc:
                    print(f"skipping --device {name}: {exc}")
                    continue
                print(f"torch device {name}: {describe_device(resolved)}")
                runs.append((f"torch:{resolved.type}", "torch", str(resolved)))
        except ImportError:  # pragma: no cover
            pass
    if not runs:
        return _skip("neither brian2 nor torch is installed")

    try:
        whole_brain = load_subnetwork(args.connectome_dir, release=args.release)
    except ConnectomeNotFoundError as exc:
        return _skip(str(exc))
    except ImportError as exc:
        return _skip(str(exc))

    print(f"{whole_brain!r}")
    try:
        from fly_driver.brains.shiu import codegen_target

        print(f"brian2 codegen target: {codegen_target()}")
    except Exception:  # pragma: no cover - brian2 absent or unprobeable
        pass
    print()

    def measure_one(
        backend: str, device: str | None, network: object, config: BenchmarkConfig, mode: str
    ) -> ThroughputResult:
        if mode == "stepped":
            return measure_in_loop(network, backend, config=config, device=device)
        if backend == "brian2":
            from fly_driver.brains import shiu

            return shiu.measure(network, config=config)
        from fly_driver.brains import torch_lif

        return torch_lif.measure(network, config=config, device=device)

    results: list[ThroughputResult] = []
    for count in args.neurons:
        size = None if count <= 0 else count
        network = load_subnetwork(args.connectome_dir, release=args.release, num_neurons=size)
        for dt_ms in args.dt:
            config = BenchmarkConfig(dt_ms=dt_ms, biological_ms=args.biological_ms)
            for mode in modes:
                for label, backend, device in runs:
                    try:
                        result = measure_one(backend, device, network, config, mode)
                    except Exception as exc:  # pragma: no cover - backend-specific
                        print(f"{label} {mode}: failed at {network.num_neurons:,}: {exc}")
                        continue
                    if mode == "stepped":
                        results.append(result)
                    print(f"{mode:>8}  {result}", f" {result.notes}" if result.notes else "")

    if not results:
        return _skip("no backend produced a measurement")

    if not results:
        return _skip("no stepped measurement was produced")
    budget = results[0].budget_ms
    print(f"\nbudget: {budget:.1f} ms of wall clock per 20 ms of biology")
    print("the verdict below is on the STEPPED numbers, which is how the loop drives it")
    fitting = [result for result in results if result.fits_budget]
    if fitting:
        best = max(fitting, key=lambda result: result.num_neurons)
        print(
            f"largest network inside it: {best.num_neurons:,} neurons "
            f"({best.backend}, dt={best.dt_ms} ms, {best.wall_ms_per_frame:.1f} ms)"
        )
        # The central complex plus descending neurons is roughly this many, and whether
        # it fits decides whether RQ2 is reachable with a real circuit.
        if best.num_neurons >= 4_300:
            print("that covers a central-complex + descending-neuron circuit (~4,300)")
        else:
            print("short of a central-complex + descending-neuron circuit (~4,300)")
    else:
        print("nothing measured fits the budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
