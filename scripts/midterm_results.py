"""Regenerate the GH-27 midterm table, figure and robustness results from run directories.

For every finished seed under RUNS_DIR without ``robustness/summary.json``, runs the
evaluation protocol (``fly_driver/configs/eval_default.yaml``'s perturbations, episodes
capped by the env, not the harness) on ``policy_final.pt``. Then aggregates every seed into
``docs/results/`` and ``docs/figures/midterm_learning_curves.png``. See ``docs/midterm.md``.

Examples:
    python scripts/midterm_results.py runs                   # robustness if missing, then report
    python scripts/midterm_results.py runs --robustness-only --seed 0  # GPU box, after training
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fly_driver.analysis.midterm import (  # noqa: E402
    ROBUSTNESS_DIRNAME,
    find_seed_runs,
    write_results,
)
from fly_driver.training.evaluation import EvalConfig  # noqa: E402


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs_dir", type=Path, help="holds <condition>/seed_<k>/ run directories")
    parser.add_argument("--episodes", type=int, default=3, help="robustness episodes per condition")
    parser.add_argument(
        "--max-steps", type=int, help="harness cap per robustness episode (default: the env's)"
    )
    parser.add_argument("--robustness-only", action="store_true", help="skip the report")
    parser.add_argument(
        "--condition", action="append", help="robustness only for this run name (repeatable)"
    )
    parser.add_argument(
        "--seed", type=int, action="append", help="robustness only for this seed (repeatable)"
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "docs/results")
    parser.add_argument(
        "--figure", type=Path, default=ROOT / "docs/figures/midterm_learning_curves.png"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _parse_args(argv)
    protocol = replace(
        EvalConfig.from_yaml(ROOT / "fly_driver/configs/eval_default.yaml"),
        episodes=args.episodes,
        max_steps=args.max_steps,
    )
    for run in find_seed_runs(args.runs_dir):
        output = run.run_dir / ROBUSTNESS_DIRNAME
        if (output / "summary.json").exists():
            continue
        if args.condition and run.run_dir.parent.name not in args.condition:
            continue
        if args.seed is not None and run.seed not in args.seed:
            continue
        from fly_driver.analysis.robustness import evaluate_checkpoint

        print(f"robustness: {run.run_dir}", flush=True)
        evaluate_checkpoint(run.run_dir, replace(protocol, output_dir=str(output)))
    if args.robustness_only:
        return 0
    print(write_results(args.runs_dir, args.results_dir, args.figure))
    print(f"wrote {args.results_dir} and {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
