"""Measure steering chatter on a trained driver, on the practice track (GH-104).

Example:
    python scripts/steering_chatter.py --checkpoint weights/best/policy_best.pt
    python scripts/steering_chatter.py --conditions noise_off --uniform-starts 2 --zone-starts 0
    python scripts/steering_chatter.py --conditions noise_off smooth_0.5 smooth_0.2 gain_0.5

Writes ``episodes.csv``, ``summary.csv``, ``straight_exits.csv`` and one ``.npz`` trace per drive
to ``--output``. The checkpoint needs its brain beside it. It needs torch and a flyvis install.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.analysis.steering_chatter import SmoothedSteering
from fly_driver.analysis.steering_report import (
    ConditionSpec,
    StartSet,
    measure,
    print_summary,
    write_results,
)
from fly_driver.envs.segments import find_segments
from fly_driver.policies.sampled_policy import SampledPolicy
from fly_driver.policies.scaled_steering_policy import ScaledSteeringPolicy
from fly_driver.training import TrainConfig
from fly_driver.training.train import build_env, load_driver

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/best/policy_best.pt"))
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "fly_driver/configs/train_lab.yaml"
    )
    parser.add_argument("--conditions", nargs="+", default=["noise_off", "noise_on"])
    parser.add_argument("--uniform-starts", type=int, default=20)
    parser.add_argument("--zone-starts", type=int, default=8)
    parser.add_argument("--zone", type=float, nargs=2, default=(4000.0, 5200.0))
    parser.add_argument("--max-steps", type=int, default=7500)
    parser.add_argument("--output", type=Path, default=Path("runs/steering_chatter"))
    parser.add_argument(
        "--eye-device",
        default="cpu",
        help="cpu repeats bit for bit. On cuda the eye differs by about 1e-6 between runs, and "
        "the closed loop grows that to a full steering lock within about 4 s.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _parse_args(argv)
    run_config = TrainConfig.from_yaml(args.config)
    eye_params = {**run_config.eye.params, "device": args.eye_device}
    run_config = replace(run_config, eye=replace(run_config.eye, params=eye_params))
    grid_env = build_env(run_config.env)
    random_env = build_env(run_config.env, training=True)
    try:
        segments = find_segments(grid_env.centerline)
        starts = StartSet.choose(
            random_env,
            uniform=args.uniform_starts,
            zone_count=args.zone_starts,
            zone=tuple(args.zone),
        )
        driver = load_driver(run_config, args.checkpoint, grid_env)
        mean_policy = driver.policy
        results = []
        for condition in args.conditions:
            spec = ConditionSpec.parse(condition)
            if spec.noise_on:
                driver.policy = SampledPolicy(mean_policy)
            elif spec.gain is not None:
                driver.policy = ScaledSteeringPolicy(mean_policy, spec.gain)
            else:
                driver.policy = mean_policy
            agent = driver if spec.blend is None else SmoothedSteering(driver, spec.blend)
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
