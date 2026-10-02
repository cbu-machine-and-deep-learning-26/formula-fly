"""Train a driver on the practice track from a YAML config (GH-17).

The config is the experiment: eye type, brain, frozen or fine-tuned, seeds. This script only
loads it, applies the command-line overrides, and calls into ``fly_driver.training``.

Examples:
    python scripts/train.py                          # default: flyvis, frozen, seeds 0 1 2
    python scripts/train.py --seed 0                 # one seed; the others on other machines
    python scripts/train.py --env dummy --eye pixels --total-env-steps 20000 --name smoke
    python scripts/train.py --config my_condition.yaml --wandb

Needs torch. On a machine with the flyvis stack, run it from that virtualenv; without
flyvis, ``--eye pixels`` still exercises the whole loop. See ``docs/training.md``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.training.config import (  # noqa: E402
    KNOWN_BRAINS,
    KNOWN_ENV_TYPES,
    KNOWN_EYE_TYPES,
    TrainConfig,
)
from fly_driver.training.loggers import UpdateRecord  # noqa: E402

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "fly_driver/configs/train_default.yaml"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="condition YAML")
    parser.add_argument("--name", help="run name; runs land in <output>/<name>/seed_<k>")
    parser.add_argument("--output", type=Path, help="overrides output_dir")
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help="run only this seed; repeat for several (default: the config's seeds)",
    )
    parser.add_argument("--eye", choices=KNOWN_EYE_TYPES, help="overrides eye.type")
    frozen = parser.add_mutually_exclusive_group()
    frozen.add_argument("--frozen", dest="frozen", action="store_true", default=None)
    frozen.add_argument("--fine-tune", dest="frozen", action="store_false")
    parser.add_argument("--brain", choices=KNOWN_BRAINS, help="overrides brain")
    parser.add_argument("--env", choices=KNOWN_ENV_TYPES, help="overrides env.type")
    parser.add_argument("--total-env-steps", type=int, help="overrides total_env_steps")
    parser.add_argument("--device", help="overrides device (auto, cpu, cuda, cuda:1...)")
    parser.add_argument("--wandb", action="store_true", help="enable Weights & Biases logging")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run dir")
    return parser.parse_args(argv)


def _apply_overrides(config: TrainConfig, args: argparse.Namespace) -> TrainConfig:
    top = {
        key: value
        for key, value in (
            ("name", args.name),
            ("output_dir", None if args.output is None else str(args.output)),
            ("total_env_steps", args.total_env_steps),
            ("device", args.device),
            ("brain", args.brain),
        )
        if value is not None
    }
    if args.seeds:
        top["seeds"] = tuple(args.seeds)
    eye = {
        key: value
        for key, value in (("type", args.eye), ("frozen", args.frozen))
        if value is not None
    }
    if eye:
        top["eye"] = replace(config.eye, **eye)
    if args.env is not None:
        # The practice-track keys in env.params mean nothing to the dummy, and vice versa.
        top["env"] = replace(config.env, type=args.env, params={})
    if args.wandb:
        top["logging"] = replace(config.logging, wandb=replace(config.logging.wandb, enabled=True))
    return replace(config, **top) if top else config


def _print_progress(record: UpdateRecord) -> None:
    mean_return = "-" if record.mean_episode_return is None else f"{record.mean_episode_return:.2f}"
    print(
        f"seed {record.seed}  update {record.update:5d}  env_steps {record.env_steps:9d}  "
        f"episodes {record.episodes_completed:3d}  return {mean_return:>9}  "
        f"kl {record.approx_kl:7.4f}  ev {record.explained_variance:6.2f}  "
        f"{record.steps_per_second:7.0f} steps/s",
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _parse_args(argv)
    config = _apply_overrides(TrainConfig.from_yaml(args.config), args)
    try:
        from fly_driver.training.train import train
    except ImportError as error:
        print(
            f"training needs torch ({error}). Run this from the flyvis virtualenv "
            "(docs/running-the-stacks.md) or install torch into a training venv; see "
            "docs/training.md.",
            file=sys.stderr,
        )
        return 2

    print(
        f"condition {config.name}: eye={config.eye.type} "
        f"({'frozen' if config.eye.frozen else 'fine-tuned'}) brain={config.brain} "
        f"env={config.env.type} seeds={list(config.seeds)} "
        f"steps/seed={config.total_env_steps} updates/seed={config.num_updates}"
    )
    print(f"logs: {config.run_root}")
    results = train(config, progress=_print_progress, overwrite=args.overwrite)
    print(f"{'seed':>4} {'env_steps':>10} {'episodes':>9} {'last return':>12}  checkpoint")
    for result in results:
        last = result.last_update
        last_return = (
            "-"
            if last is None or last.mean_episode_return is None
            else f"{last.mean_episode_return:.2f}"
        )
        print(
            f"{result.seed:>4} {result.env_steps:>10} {result.episodes:>9} {last_return:>12}  "
            f"{result.checkpoint}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
