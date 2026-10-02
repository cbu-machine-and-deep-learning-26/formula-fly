"""Neuron-count versus frame-time for the central-complex module.

Times :meth:`CentralComplexBrain.step` and prints the trainable-parameter counts
next to a dense blank net of the same width. Exits 0 with a SKIP line when torch
is not installed, so a machine on the base extra can run it.

The budget is one environment frame (20 ms at 50 Hz). It is the whole frame, not
a budget the brain may spend alone; a width that already misses it cannot be live.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from fly_driver.brains.sweep import frame_budget_ms, sweep_frame_times


def main(argv: list[str] | None = None) -> int:
    """Sweep neuron counts and optionally write the curve to CSV.

    Args:
        argv: Arguments, excluding the program name. ``None`` reads ``sys.argv``.

    Returns:
        Process exit code. ``0`` on a finished sweep and when torch is absent.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--counts",
        default="256,1024,4096,16384,65536,131072,262144,393216",
        help="Comma-separated neuron counts. The default runs up to the 20 ms frame.",
    )
    parser.add_argument("--input-dim", type=int, default=576)
    parser.add_argument("--output-dim", type=int, default=32)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=None, help="CSV path. Omit to only print.")
    args = parser.parse_args(argv)

    try:
        import torch  # noqa: F401  presence check; the modules import it themselves
    except ImportError:
        print("SKIP: torch is not installed; the central-complex sweep needs it.")
        return 0

    counts = tuple(int(part) for part in args.counts.split(",") if part.strip())
    points = sweep_frame_times(
        counts,
        input_dim=args.input_dim,
        output_dim=args.output_dim,
        steps=args.steps,
        warmup=args.warmup,
        seed=args.seed,
        device=args.device,
    )
    budget = frame_budget_ms()
    header = (
        "neuron_count",
        "output_dim",
        "constrained_trainable",
        "blank_trainable",
        "median_frame_ms",
        "p95_frame_ms",
        "frame_budget_ms",
        "meets_frame_budget",
    )
    rows = [
        {
            "neuron_count": point.neuron_count,
            "output_dim": point.output_dim,
            "constrained_trainable": point.constrained_trainable,
            "blank_trainable": point.blank_trainable,
            "median_frame_ms": f"{point.median_frame_ms:.4f}",
            "p95_frame_ms": f"{point.p95_frame_ms:.4f}",
            "frame_budget_ms": f"{point.frame_budget_ms:.4f}",
            "meets_frame_budget": str(point.meets_frame_budget).lower(),
        }
        for point in points
    ]
    print(f"frame budget {budget:.1f} ms ({1000.0 / budget:.0f} Hz), brain step only")
    print(",".join(header))
    for row in rows:
        print(",".join(str(row[column]) for column in header))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(header))
            writer.writeheader()
            writer.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
