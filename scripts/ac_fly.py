"""Fly side of the Assetto Corsa bridge: serve a driver to scripts/ac_car.py over TCP.

Start this first; it waits for the car. See docs/assetto-corsa-bridge.md.

Examples:
    python scripts/ac_fly.py --throttle 0.3
    python scripts/ac_fly.py --checkpoint <run_dir>/policy_final.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.bridge.fly_side import (
    ConstantDriver,
    load_trained_driver,
    open_listener,
    serve,
)
from fly_driver.bridge.wire import DEFAULT_PORT
from fly_driver.interface import FRAME_SHAPE, ControlVector


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, build the driver, and serve one car connection."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0", help="127.0.0.1 for same-host only")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--checkpoint", type=Path, help="trained MlpPolicy; omit for scripted")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steer", type=float, default=0.0, help="scripted driver only")
    parser.add_argument("--throttle", type=float, default=0.3, help="scripted driver only")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    if args.checkpoint is not None:
        driver = load_trained_driver(args.checkpoint, args.device)
    else:
        driver = ConstantDriver(ControlVector(args.steer, args.throttle, 0.0), FRAME_SHAPE)
    with open_listener(args.host, args.port) as listener:
        logging.info("fly side listening on %s:%d", args.host, args.port)
        serve(driver, listener)


if __name__ == "__main__":
    main()
