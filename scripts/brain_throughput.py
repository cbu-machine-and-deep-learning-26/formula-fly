"""Measure whether a fly brain can keep up with the frame budget (GH-23).

Wiring only: every measurement lives in :mod:`fly_driver.brains`, per AGENTS.md §14.3.

Run it::

    python scripts/brain_throughput.py
    python scripts/brain_throughput.py --neurons 3000 10000 --dt 0.25 0.5 1.0
    python scripts/brain_throughput.py --backend torch --device cuda

It needs the FlyWire connectivity, which is a 370 MB download kept outside this
repository and pointed at with ``FLY_CONNECTOME_DIR``; see
``docs/running-the-stacks.md``. It exits ``0`` with a ``SKIP:`` line when the data or
the optional simulators are absent, the way ``scripts/flyvis_smoke.py`` does, so CI on
the base install stays green.

**Check the reported code generation target before quoting a Brian2 number.** Brian2
falls back from Cython to plain numpy when there is no C++ compiler, with only a
warning, and the fallback is several times slower than the backend can actually go.
"""

from __future__ import annotations

import argparse

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
    parser.add_argument("--device", default=None, help="torch device, e.g. cpu or cuda")
    parser.add_argument(
        "--biological-ms", type=float, default=100.0, help="biology to time per run"
    )
    return parser.parse_args(argv)


def _skip(reason: str) -> int:
    print(f"SKIP: {reason}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    backends = []
    if args.backend in ("brian2", "both"):
        try:
            from fly_driver.brains import shiu

            backends.append(("brian2", lambda net, cfg: shiu.measure(net, config=cfg)))
        except ImportError:  # pragma: no cover - depends on the local install
            pass
    if args.backend in ("torch", "both"):
        try:
            from fly_driver.brains import torch_lif

            backends.append(
                (
                    "torch",
                    lambda net, cfg: torch_lif.measure(net, config=cfg, device=args.device),
                )
            )
        except ImportError:  # pragma: no cover
            pass
    if not backends:
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

    results: list[ThroughputResult] = []
    for count in args.neurons:
        size = None if count <= 0 else count
        network = load_subnetwork(args.connectome_dir, release=args.release, num_neurons=size)
        for dt_ms in args.dt:
            config = BenchmarkConfig(dt_ms=dt_ms, biological_ms=args.biological_ms)
            for name, run in backends:
                try:
                    result = run(network, config)
                except Exception as exc:  # pragma: no cover - backend-specific
                    print(f"{name}: failed at {network.num_neurons:,} neurons: {exc}")
                    continue
                results.append(result)
                print(result, f" {result.notes}" if result.notes else "")

    if not results:
        return _skip("no backend produced a measurement")

    budget = results[0].budget_ms
    fitting = [r for r in results if r.fits_budget]
    print(f"\nbudget: {budget:.1f} ms of wall clock per 20 ms of biology")
    if fitting:
        best = max(fitting, key=lambda r: r.num_neurons)
        print(
            f"largest network inside it: {best.num_neurons:,} neurons "
            f"({best.backend}, dt={best.dt_ms} ms, {best.wall_ms_per_frame:.1f} ms)"
        )
    else:
        print("nothing measured fits the budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
