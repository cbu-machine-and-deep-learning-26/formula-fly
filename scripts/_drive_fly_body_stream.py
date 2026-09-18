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
from scripts.drive import KeyboardInput

CONTROL_HZ = 50


class ScriptedWeave:
    """The autopilot: a steady left-right weave and constant throttle."""

    name = "scripted weave"

    def __init__(self, period_s: float, throttle: float) -> None:
        self.period_s = period_s
        self.throttle = throttle
        self._t = 0.0

    def control(self, speed_mps: float) -> ControlVector:
        del speed_mps
        self._t += 1.0 / CONTROL_HZ
        steer = float(np.sin(2 * np.pi * self._t / self.period_s))
        return ControlVector(steer=steer, throttle=self.throttle, brake=0.0)

    def stop(self) -> None:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--input",
        choices=("scripted", "keyboard"),
        default="scripted",
        help="scripted: autopilot weave. keyboard: your arrow keys, same as scripts/drive.py -- "
        "needs Accessibility permission granted to your terminal app (System Settings > "
        "Privacy & Security > Accessibility), or held keys are silently never seen",
    )
    parser.add_argument("--period-s", type=float, default=4.0, help="scripted steer weave period")
    parser.add_argument("--throttle", type=float, default=0.5)
    parser.add_argument("--raw-steer", action="store_true", help="keyboard: no speed-based assist")
    args = parser.parse_args()

    if args.input == "keyboard":
        source = KeyboardInput(raw_steer=args.raw_steer)
        source.start()
    else:
        source = ScriptedWeave(period_s=args.period_s, throttle=args.throttle)

    body = CombinedBody(FlybodyWingBody(), TrackballLegBody())
    body.reset()

    tmp_path = args.out.with_suffix(".tmp.npy")
    speed_estimate = 0.0  # only the keyboard's own speed-assist reads this; see below
    print(f"fly body process: streaming controls (input: {source.name})", flush=True)
    try:
        while True:
            step_start = time.perf_counter()
            intent = source.control(speed_estimate)
            realised = body.actuate(intent)
            # The keyboard's steering_gain wants the car's own speed, which this process
            # never sees (the car lives in the other process). realised.throttle is the
            # closest local proxy; wrong in magnitude, right in trend, and only softens
            # keyboard steering -- scripted mode does not use it at all.
            speed_estimate = realised.throttle * 30.0

            np.save(tmp_path, np.array([realised.steer, realised.throttle, realised.brake]))
            os.replace(tmp_path, args.out)

            remaining = (1.0 / CONTROL_HZ) - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        source.stop()


if __name__ == "__main__":
    raise SystemExit(main())
