"""Run the evaluation harness on a baseline agent from a YAML config.

Example:
    python scripts/evaluate.py --agent random --output runs/eval_random
    python scripts/evaluate.py --config fly_driver/configs/eval_default.yaml --record-video
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.envs import DummyTrackEnv
from fly_driver.policies import ConstantAgent, RandomAgent
from fly_driver.training import EvalConfig, EvalReport, evaluate

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "fly_driver/configs/eval_default.yaml"

ENVS: dict[str, Callable[[], DummyTrackEnv]] = {"dummy": DummyTrackEnv}
AGENTS: dict[str, Callable[[], object]] = {
    "random": RandomAgent,
    "constant": lambda: ConstantAgent(steer=0.0, throttle=1.0, brake=0.0),
}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env", choices=sorted(ENVS), default="dummy")
    parser.add_argument("--agent", choices=sorted(AGENTS), default="random")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, help="overrides output_dir")
    parser.add_argument("--record-video", action="store_true")
    return parser.parse_args(argv)


def _print_report(report: EvalReport) -> None:
    print(f"actions_digest: {report.actions_digest}")
    print(f"video_backend: {report.video_backend}")
    header = f"{'condition':<32}{'return':>10}{'laps':>7}{'lap_time_s':>12}{'ratio':>8}"
    print(header)
    for item in report.conditions:
        lap_time = "-" if item.mean_lap_time_s is None else f"{item.mean_lap_time_s:.2f}"
        ratio = "-" if item.return_ratio_to_clean is None else f"{item.return_ratio_to_clean:.2f}"
        print(
            f"{item.condition:<32}{item.mean_return:>10.2f}"
            f"{item.lap_completion_rate:>7.2f}{lap_time:>12}{ratio:>8}"
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _parse_args(argv)
    config = EvalConfig.from_yaml(args.config)
    overrides = {
        key: value
        for key, value in (
            ("episodes", args.episodes),
            ("seed", args.seed),
            ("output_dir", None if args.output is None else str(args.output)),
        )
        if value is not None
    }
    if args.record_video:
        overrides["record_video"] = True
    config = replace(config, **overrides)
    report = evaluate(ENVS[args.env], AGENTS[args.agent](), config)
    _print_report(report)
    if config.output_dir is not None:
        print(f"wrote {Path(config.output_dir) / 'episodes.csv'} and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
