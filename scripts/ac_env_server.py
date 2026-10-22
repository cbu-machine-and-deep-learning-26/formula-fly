"""Serve Assetto Corsa to the trainer as a training env: reset and step over TCP.

Runs on the AC box in AssettoCorsaGym's Python 3.9 env, in the same desktop session as AC,
with AC already in a Hotlap at the configured track and car. Start it before the trainer;
it serves every trainer that connects until stopped. ``--fake`` serves
fly_driver.bridge.fake_ac instead, anywhere, with no AC. See docs/assetto-corsa-training.md.

Examples:
    python scripts/ac_env_server.py --gym-root C:/src/assetto_corsa_gym
    python scripts/ac_env_server.py --fake --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.bridge.env_link import serve_env
from fly_driver.bridge.fly_side import open_listener
from fly_driver.bridge.wire import DEFAULT_ENV_PORT
from fly_driver.interface import FRAME_RATE_HZ, FRAME_SHAPE


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, build the car, and serve it until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gym-root", type=Path, help="AssettoCorsaGym checkout")
    parser.add_argument("--config", type=Path, help="defaults to <gym-root>/config.yml")
    parser.add_argument("--track", default="ks_silverstone-gp")
    parser.add_argument("--car", default="dallara_f317")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 for a remote trainer")
    parser.add_argument("--port", type=int, default=DEFAULT_ENV_PORT)
    parser.add_argument("--work-dir", type=Path, default=Path("ac_runs"))
    parser.add_argument("--fake", action="store_true", help="serve FakeAc, no AC needed")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.fake:
        from fly_driver.bridge.fake_ac import FakeAc

        car = FakeAc()
    else:
        if args.gym_root is None:
            parser.error("--gym-root is required unless --fake")
        from fly_driver.bridge.assetto import AssettoCorsaCar

        car = AssettoCorsaCar(
            args.gym_root,
            args.config or args.gym_root / "config.yml",
            args.track,
            args.car,
            args.work_dir,
            upstream_termination=False,
        )
    hello = {
        "track": args.track,
        "car": args.car,
        "frame_shape": list(FRAME_SHAPE),
        "frame_rate_hz": FRAME_RATE_HZ,
    }
    listener = open_listener(args.host, args.port)
    logging.info("env server for %s/%s on %s:%d", args.track, args.car, args.host, args.port)
    try:
        serve_env(car, listener, hello)
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()
        car.close()


if __name__ == "__main__":
    main()
