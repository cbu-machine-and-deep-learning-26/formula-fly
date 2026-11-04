"""Where a driver's steering chatter comes from, stage by stage (GH-104).

Example:
    python scripts/steering_source.py --checkpoint weights/best/policy_best.pt

See :mod:`fly_driver.analysis.steering_source`. It needs torch and a flyvis install.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.analysis.steering_source import (
    StageRecorder,
    run_closed_loop,
    run_frozen_frame,
    run_shadow,
    stage_report,
)
from fly_driver.training import TrainConfig
from fly_driver.training.train import build_env, load_driver

REPO_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 1005)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=Path("weights/best/policy_best.pt"))
    parser.add_argument(
        "--config", type=Path, default=REPO_ROOT / "fly_driver/configs/train_lab.yaml"
    )
    parser.add_argument("--max-frames", type=int, default=3000)
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
    env = build_env(run_config.env, training=True)
    try:
        recorder = StageRecorder(load_driver(run_config, args.checkpoint, env))
        rate = float(env.frame_rate_hz)

        def report(label: str) -> None:
            fields = asdict(stage_report(recorder, label, rate))
            print(
                ", ".join(
                    f"{key} {value:.3f}" if isinstance(value, float) else str(value)
                    for key, value in fields.items()
                ),
                flush=True,
            )

        for seed in SEEDS:
            run_closed_loop(env, recorder, seed=seed, max_frames=args.max_frames)
            report(f"closed loop, seed {seed}")
        run_frozen_frame(env, recorder, seed=0, drive_first=0, frames=500)
        report("frozen frame, at rest")
        run_frozen_frame(env, recorder, seed=0, drive_first=400, frames=500)
        report("frozen frame, from a moving car")
        for seed in SEEDS:
            run_shadow(env, recorder, seed=seed, max_frames=1500)
            report(f"shadow, seed {seed}")
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
