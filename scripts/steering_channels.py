"""Override one control channel at run time, to find which one limits the car (GH-104).

Example:
    python scripts/steering_channels.py --conditions own gov_0.6 gov_0.85 vale_85 script_own

See :mod:`fly_driver.analysis.channel_overrides` for the conditions. Writes the same files as
``scripts/steering_chatter.py``. It needs torch and a flyvis install.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.analysis.channel_overrides import ChannelOverride, ChannelSpec, WatchedEnv
from fly_driver.analysis.steering_report import StartSet, measure, print_summary, write_results
from fly_driver.envs.braking_line import speed_profile
from fly_driver.envs.segments import find_segments
from fly_driver.training import TrainConfig
from fly_driver.training.train import build_env, load_driver

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/best/policy_best.pt"))
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "fly_driver/configs/train_lab.yaml"
    )
    parser.add_argument("--conditions", nargs="+", default=["own", "gov_0.6", "script_gov_0.85"])
    parser.add_argument("--uniform-starts", type=int, default=30)
    parser.add_argument("--zone-starts", type=int, default=10)
    parser.add_argument("--zone", type=float, nargs=2, default=(4000.0, 5200.0))
    parser.add_argument("--max-steps", type=int, default=7500)
    parser.add_argument("--output", type=Path, default=Path("runs/steering_channels"))
    parser.add_argument(
        "--eye-device", default="cpu", help="cpu repeats bit for bit; cuda does not"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _parse_args(argv)
    run_config = TrainConfig.from_yaml(args.config)
    eye_params = {**run_config.eye.params, "device": args.eye_device}
    run_config = replace(run_config, eye=replace(run_config.eye, params=eye_params))
    board: dict[str, object] = {}
    grid_env = WatchedEnv(build_env(run_config.env), board)
    random_env = WatchedEnv(build_env(run_config.env, training=True), board)
    try:
        segments = find_segments(grid_env.centerline)
        profile = speed_profile(grid_env.centerline)
        starts = StartSet.choose(
            random_env,
            uniform=args.uniform_starts,
            zone_count=args.zone_starts,
            zone=tuple(args.zone),
        )
        driver = load_driver(run_config, args.checkpoint, grid_env.env)
        results = []
        for condition in args.conditions:
            agent = ChannelOverride(
                driver,
                board,
                ChannelSpec.parse(condition),
                profile.arclength_m,
                profile.speed_mps,
            )
            results.extend(
                measure(
                    condition,
                    agent,
                    grid_env=grid_env,
                    random_env=random_env,
                    starts=starts,
                    segments=segments,
                    max_steps=args.max_steps,
                )
            )
    finally:
        grid_env.close()
        random_env.close()
    write_results(results, args.output)
    print_summary(results)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
