"""Car side of the Assetto Corsa bridge: step AC and trade frames for controls with the fly.

Runs on the Windows box in AssettoCorsaGym's Python 3.9 env, after scripts/ac_fly.py is
listening. See docs/assetto-corsa-bridge.md.

Example:
    python scripts/ac_car.py --gym-root C:/src/assetto_corsa_gym --fly-host 127.0.0.1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.bridge.assetto import AssettoCorsaCar
from fly_driver.bridge.car_side import FlyLink, run_episode
from fly_driver.bridge.wire import DEFAULT_PORT
from fly_driver.interface import FRAME_RATE_HZ, FRAME_SHAPE


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, connect to the fly, and run one episode in AC."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gym-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="defaults to <gym-root>/config.yml")
    parser.add_argument("--track", default="ks_silverstone-gp")
    parser.add_argument("--car", default="dallara_f317")
    parser.add_argument("--fly-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--deadline-ms", type=float, default=15.0)
    parser.add_argument("--seconds", type=float, default=180.0, help="episode cap")
    parser.add_argument("--work-dir", type=Path, default=Path("ac_runs"))
    parser.add_argument("--log", type=Path, default=Path("ac_bridge_timing.csv"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    car = AssettoCorsaCar(
        args.gym_root,
        args.config or args.gym_root / "config.yml",
        args.track,
        args.car,
        args.work_dir,
    )
    link = FlyLink(args.fly_host, args.port, args.deadline_ms / 1000.0)
    try:
        stats = run_episode(car, link, FRAME_SHAPE, int(args.seconds * FRAME_RATE_HZ))
    finally:
        link.close()
        car.close()
    stats.write_csv(args.log)
    print(stats.summary())


if __name__ == "__main__":
    main()
