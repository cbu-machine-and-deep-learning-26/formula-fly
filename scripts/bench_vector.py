"""How many tracks should one trainer drive on this machine? Measure it (GH-27).

    python scripts/bench_vector.py                                   # 1, 8, 16, 32 tracks
    python scripts/bench_vector.py --num-envs 16 24 32 48 --updates 3

Runs a few PPO updates of a config (default: the lab recipe, without its warm start) at each
``num_envs`` and prints env steps per second from the last update, so worker start-up and
the first CUDA calls are not counted. Runs go to a temporary directory and are deleted.
Pick the smallest ``num_envs`` past which steps/s stops rising, then set it, and
``ppo.rollout_steps`` as a multiple of it, in your config.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml

from fly_driver.training.config import TrainConfig
from fly_driver.training.train import train_seed


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("fly_driver/configs/train_lab.yaml"))
    parser.add_argument("--num-envs", type=int, nargs="+", default=[1, 8, 16, 32])
    parser.add_argument("--steps-per-env", type=int, default=128, help="rollout length per track")
    parser.add_argument("--updates", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    """Time each ``num_envs`` and print a table."""
    args = _parse()
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results = []
    with tempfile.TemporaryDirectory() as scratch:
        for num_envs in args.num_envs:
            rollout = num_envs * args.steps_per_env
            data = dict(base)
            data.update(
                name=f"bench_{num_envs}",
                output_dir=scratch,
                seeds=[0],
                init_from=None,
                num_envs=num_envs,
                total_env_steps=rollout * args.updates,
            )
            data["ppo"] = {**base["ppo"], "rollout_steps": rollout, "minibatches": 8}
            data["eval"] = {**base.get("eval", {}), "every_updates": 0}
            data["logging"] = {**base.get("logging", {}), "checkpoint_every_updates": 0}
            config = TrainConfig.from_dict(data)
            records = []
            train_seed(config, 0, progress=records.append, overwrite=True)
            speed = records[-1].steps_per_second
            results.append((num_envs, speed))
            print(f"num_envs {num_envs:>3}: {speed:8.0f} env steps/s", flush=True)
    baseline = results[0][1]
    print("\nnum_envs  steps/s  vs first")
    for num_envs, speed in results:
        print(f"{num_envs:>8}  {speed:7.0f}  {speed / baseline:6.1f}x")


if __name__ == "__main__":
    main()
