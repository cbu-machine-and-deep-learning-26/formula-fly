"""The fly-body half of drive_fly_live.py's live hand-off. No GL context, ever.

Runs as a plain-Python subprocess: steps CombinedBody continuously and writes the
realised control out atomically (temp file + os.replace) so the mjpython viewer process
never reads a torn write. Kept in its own file, not a function in drive_fly_live.py,
so it is unambiguous that it never imports mujoco.viewer or opens a render context --
that separation is the entire fix for the GL-context corruption drive_fly.py's
docstring documents.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from fly_driver.body import CombinedBody, FlybodyWingBody, TrackballLegBody
from fly_driver.interface import ControlVector

CONTROL_HZ = 50


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--period-s", type=float, default=4.0, help="scripted steer weave period")
    parser.add_argument("--throttle", type=float, default=0.5)
    args = parser.parse_args()

    body = CombinedBody(FlybodyWingBody(), TrackballLegBody())
    body.reset()

    tmp_path = args.out.with_suffix(".tmp.npy")
    t = 0.0
    print("fly body process: streaming controls", flush=True)
    while True:
        step_start = time.perf_counter()
        t += 1.0 / CONTROL_HZ
        steer = float(np.sin(2 * np.pi * t / args.period_s))
        intent = ControlVector(steer=steer, throttle=args.throttle, brake=0.0)
        realised = body.actuate(intent)

        np.save(tmp_path, np.array([realised.steer, realised.throttle, realised.brake]))
        os.replace(tmp_path, args.out)

        remaining = (1.0 / CONTROL_HZ) - (time.perf_counter() - step_start)
        if remaining > 0:
            time.sleep(remaining)


if __name__ == "__main__":
    raise SystemExit(main())
