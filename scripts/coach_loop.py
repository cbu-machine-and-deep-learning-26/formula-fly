"""Train for a whole GPU job in segments, with Claude reviewing the reward between them.

    python scripts/coach_loop.py --config fly_driver/configs/train_lab.yaml \\
        --prefix coached_20261105 --hours 9.3 --num-envs 8

Each segment warm-starts from the best checkpoint so far, which lands in
``<output>/<prefix>/best/``. ``<output>/<prefix>/coach_loop.jsonl`` logs every segment,
score and review. To change the recipe of a running job, write a full recipe to
``<output>/<prefix>/override.yaml``; the next segment uses it. See :mod:`fly_driver.coach.loop`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.coach.loop import SegmentPlan, run_coached  # noqa: E402
from fly_driver.coach.review import load_api_key  # noqa: E402


def main() -> None:
    """Parse the command line and run the coached job."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, required=True, help="the first recipe")
    parser.add_argument("--prefix", required=True, help="run name prefix")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--hours", type=float, required=True, help="total wall clock")
    parser.add_argument("--first-segment-hours", type=float, default=2.5)
    parser.add_argument("--segment-hours", type=float, default=1.75)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps-per-env", type=int, default=512)
    parser.add_argument("--best-dir", type=Path, help="default <output>/<prefix>/best")
    parser.add_argument("--no-review", action="store_true", help="never ask Claude")
    parser.add_argument("--note", help="context for every review")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--coach-dir", type=Path, help="default <output>/<prefix>/coach")
    args = parser.parse_args()

    load_api_key(args.env_file)
    recipe = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    plan = SegmentPlan(
        prefix=args.prefix,
        output_dir=args.output,
        num_envs=args.num_envs,
        steps_per_env=args.steps_per_env,
        first_segment_s=args.first_segment_hours * 3600,
        segment_s=args.segment_hours * 3600,
        deadline=time.time() + args.hours * 3600,
    )
    run_coached(
        recipe,
        plan,
        best_dir=args.best_dir or args.output / args.prefix / "best",
        review=not args.no_review,
        note=args.note,
        coach_dir=args.coach_dir,
    )


if __name__ == "__main__":
    main()
