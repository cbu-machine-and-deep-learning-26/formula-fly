"""Does a steering wobble alone stop a car on a straight? (GH-104)

Example:
    python scripts/steering_wobble.py
    python scripts/steering_wobble.py --recorded runs/steering_chatter/noise_off_uniform_19.npz

No policy runs, so it needs neither torch nor flyvis. See
:mod:`fly_driver.analysis.steering_wobble`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.analysis.steering_wobble import (
    drive_straight,
    longest_straights,
    recorded_wobble,
    square_wobble,
)
from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.envs.segments import find_segments

SQUARE_CASES = [
    (flip_frames, amplitude) for flip_frames in (5, 10, 20) for amplitude in (1.0, 0.5, 0.3)
]
RECORDED_OFFSETS = (0, 2000, 4000, 6000)
RECORDED_GAINS = (1.0, 0.5, 0.3)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--recorded", type=Path, nargs="*", default=[], help="traces from steering_chatter.py"
    )
    parser.add_argument("--straights", type=int, default=3)
    return parser.parse_args(argv)


def _cell(end_speed_kmh: float, crashed: bool) -> str:
    return f"{'crash ' if crashed else ''}{end_speed_kmh:.0f}"


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _parse_args(argv)
    env = PracticeTrack(random_start=True, max_steps=4000)
    try:
        env.reset(seed=0)
        straights = longest_straights(find_segments(env.centerline), args.straights)
        print(f"{'wobble':<28}" + "".join(f"{item.name + ' km/h':>16}" for item in straights))
        run = [drive_straight(env, item, lambda frame: 0.0) for item in straights]
        print(
            f"{'none':<28}"
            + "".join(f"{_cell(item.end_speed_kmh, item.crashed):>16}" for item in run)
        )
        for flip_frames, amplitude in SQUARE_CASES:
            wobble = square_wobble(flip_frames, amplitude)
            run = [drive_straight(env, item, wobble) for item in straights]
            cells = "".join(f"{_cell(item.end_speed_kmh, item.crashed):>16}" for item in run)
            print(f"{f'flip/{flip_frames} frames, amp {amplitude}':<28}{cells}", flush=True)
        if args.recorded:
            steer = np.concatenate([np.load(path)["steer"] for path in args.recorded])
            for gain in RECORDED_GAINS:
                cells = ""
                for straight in straights:
                    runs = [
                        drive_straight(
                            env, straight, recorded_wobble(steer, offset=item, gain=gain)
                        )
                        for item in RECORDED_OFFSETS
                    ]
                    crashes = [item.end_speed_kmh for item in runs if item.crashed]
                    if crashes:
                        text = f"crash {len(crashes)}/{len(runs)} {np.mean(crashes):.0f}"
                    else:
                        text = f"{np.mean([item.end_speed_kmh for item in runs]):.0f}"
                    cells += f"{text:>16}"
                print(f"{f'recorded x{gain}':<28}{cells}", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
