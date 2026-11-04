"""Write a copy of a policy checkpoint with its outputs scaled down (GH-104).

Example:
    python scripts/rescale_checkpoint.py --source weights/best/policy_best.pt \
        --target weights/variable_start/policy_best.pt --steer 0.3 --pedals 0.7

The source isn't changed. Its brain is copied beside the target. It needs torch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.policies.rescale import rescale_checkpoint


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--steer", type=float, default=1.0)
    parser.add_argument("--pedals", type=float, default=1.0, help="gain on throttle and brake")
    parser.add_argument("--squash-mean", action="store_true", help="keep the mean inside +-1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _parse_args(argv)
    written = rescale_checkpoint(
        args.source,
        args.target,
        steer=args.steer,
        throttle=args.pedals,
        brake=args.pedals,
        squash_mean=args.squash_mean,
    )
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
