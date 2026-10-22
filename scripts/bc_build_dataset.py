"""Pair a re-rendered replay's frames with the driver's controls: one BC drive (GH-29).

Reads the replay (inputs, pose) and ``scripts/bc_replay_capture.py``'s output, checks that
every frame was rendered at the replay instant it claims, adds the sidebar training frames
carry, and writes a
:class:`~fly_driver.training.bc_data.HumanLaps` ``.npz``. ``--strip`` also saves frames
from the hardest corners with their steer, throttle and brake, to check the alignment by eye.

    python scripts/bc_build_dataset.py --replay <drive>.acreplay \\
        --capture <drive> --out data/human_laps/<drive>.npz --strip <drive>_strip.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from fly_driver.interface import FRAME_RATE_HZ
from fly_driver.training.bc_data import (
    HumanLaps,
    add_ac_sidebar,
    fit_render_offset,
    parse_replay,
    replay_controls,
)

# A frame whose pose offset strays further than this from the median was rendered at the
# wrong time (fit_render_offset): 0.5 m is 6 ms at 300 km/h, under a third of a step.
MAX_POSE_ERROR_M = 0.5


def save_strip(laps: HumanLaps, path: Path, count: int = 6) -> None:
    """Save the frames at the ``count`` largest steering inputs, captioned with the controls."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steer = laps.controls[:, 0]
    order = np.argsort(-np.abs(steer))
    picked: list[int] = []
    for index in order:  # one frame per corner: at least 3 s apart
        if all(abs(int(index) - other) > 3 * FRAME_RATE_HZ for other in picked):
            picked.append(int(index))
        if len(picked) == count:
            break
    figure, axes = plt.subplots(2, count, figsize=(2.2 * count, 4.6))
    for column, index in enumerate(sorted(picked)):
        for row, offset in enumerate((-25, 0)):  # half a second before, and the peak
            frame = max(index + offset, 0)
            axis = axes[row, column]
            axis.imshow(laps.frames[frame])
            steer_value, throttle, brake = laps.controls[frame]
            axis.set_title(
                f"#{frame} s{steer_value:+.2f}\nt{throttle:.2f} b{brake:.2f}", fontsize=8
            )
            axis.axis("off")
    figure.suptitle("top: 0.5 s before; bottom: steering peak (s>0 = right)", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=110)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--capture", type=Path, required=True, help="bc_replay_capture --out")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strip", type=Path)
    args = parser.parse_args()

    replay = parse_replay(args.replay)
    frames = np.load(f"{args.capture}.frames.npy")
    poses = np.load(f"{args.capture}.pose.npy")
    offset_s, error_m = fit_render_offset(replay, poses[:, 1:])
    bad = np.flatnonzero(error_m > MAX_POSE_ERROR_M)
    print(
        f"{len(frames)} frames; AC drew each {offset_s * 1000:+.0f} ms from its instant; pose "
        f"deviation median {np.median(error_m):.3f} m, max {error_m.max():.3f} m, "
        f"{len(bad)} over {MAX_POSE_ERROR_M} m"
    )
    if len(bad):
        print(f"first misaligned frames: {bad[:10].tolist()}", file=sys.stderr)
        return 1

    laps = HumanLaps.from_replay(replay, add_ac_sidebar(frames), time_offset_s=offset_s)
    laps.save(args.out)
    per_lap = {
        int(lap): int(n) for lap, n in zip(*np.unique(laps.laps, return_counts=True), strict=True)
    }
    controls = replay_controls(replay, np.arange(len(frames)) / FRAME_RATE_HZ + offset_s)
    print(
        f"wrote {args.out}: {len(laps)} of {len(frames)} frames kept, frames per lap {per_lap}, "
        f"{len(laps.segment_bounds())} segments; |steer| max {np.abs(controls[:, 0]).max():.2f}"
    )
    if args.strip:
        save_strip(laps, args.strip)
        print(f"wrote {args.strip}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
