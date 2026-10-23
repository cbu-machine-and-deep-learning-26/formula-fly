"""Behaviour-clone the fly driver on human laps from a BC config (GH-29).

    python scripts/bc_train.py --config fly_driver/configs/bc_flyvis_cx_ac.yaml

See ``fly_driver/training/bc.py`` for what trains and ``docs/assetto-corsa-training.md``
§ Behaviour cloning for the whole path from a human drive to a PPO warm start.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fly_driver.training.bc import BcConfig, train_bc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--epochs", type=int, help="override the config's epochs")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing run")
    args = parser.parse_args()

    config = BcConfig.from_yaml(args.config)
    if args.epochs is not None:
        config = BcConfig.from_dict({**config.__dict__, "epochs": args.epochs})
    train_bc(config, overwrite=args.overwrite)
    print(f"done: {config.run_dir / 'policy_final.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
