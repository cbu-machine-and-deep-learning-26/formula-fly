"""Have Claude review a training run's reward and propose the next recipes (GH-27).

    python scripts/reward_coach.py runs/lab_20261104_g3                # one review now
    python scripts/reward_coach.py runs/lab_20261104_g3 --dry-run      # digest + prompts only
    python scripts/reward_coach.py runs/lab_20261104_g3 --every-minutes 60 --note "Vale test"

The API key comes from ``ANTHROPIC_API_KEY``, or from ``ANTHROPIC_API_KEY=...`` in a ``.env``
file in the current directory (git-ignored). Reviews land in ``<run>/coach/<timestamp>/``:
``digest.md``, ``review.md``, ``response.json`` and one ``proposal_<k>_*.yaml`` per proposal
the guardrails accepted. This script reads runs and writes reviews; it never starts training.
See docs/reward-coach.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from fly_driver.coach.digest import build_digest
from fly_driver.coach.review import DEFAULT_MODEL, load_api_key, review_run


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path, help="run or seed directories to pool")
    parser.add_argument("--note", help="context for Claude, e.g. what this run is testing")
    parser.add_argument("--coach-dir", type=Path, help="where reviews go; default <run>/coach")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env-file", type=Path, help="a .env file holding ANTHROPIC_API_KEY")
    parser.add_argument("--dry-run", action="store_true", help="write the prompts, call nothing")
    parser.add_argument(
        "--every-minutes",
        type=float,
        default=0.0,
        help="review again on this interval while the run trains; 0 = once",
    )
    parser.add_argument(
        "--min-new-drives",
        type=int,
        default=200,
        help="in --every-minutes mode, skip a review until this many new drives have finished",
    )
    return parser.parse_args()


def _once(args: argparse.Namespace) -> int:
    result = review_run(
        args.runs,
        coach_dir=args.coach_dir,
        note=args.note,
        model=args.model,
        dry_run=args.dry_run,
    )
    print(f"wrote {result.directory}")
    if args.dry_run:
        return 0
    print(f"verdict: {result.review['verdict']}")
    for title, path in result.accepted.items():
        print(f"  accepted: {title} -> {path}")
    for title, problems in result.rejected.items():
        print(f"  rejected: {title}: {'; '.join(problems)}")
    usage = result.usage
    print(
        f"tokens: in {usage.get('input_tokens', '?')}, "
        f"cache read {usage.get('cache_read_input_tokens', '?')}, "
        f"out {usage.get('output_tokens', '?')}"
    )
    return len(build_digest(args.runs).drives)


def main() -> None:
    """Review once, or on an interval."""
    args = _parse()
    load_api_key(args.env_file)
    if not args.dry_run:
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set: export it or put it in .env (see --env-file)")
    if args.every_minutes <= 0:
        _once(args)
        return
    reviewed_at = -args.min_new_drives
    while True:
        drives = len(build_digest(args.runs).drives)
        if drives - reviewed_at >= args.min_new_drives:
            try:
                reviewed_at = _once(args)
            except Exception as error:  # noqa: BLE001  a failed review must not end the watch
                print(json.dumps({"review_failed": str(error)}), flush=True)
        else:
            print(f"{drives - reviewed_at} new drives since the last review; waiting", flush=True)
        time.sleep(args.every_minutes * 60)


if __name__ == "__main__":
    main()
