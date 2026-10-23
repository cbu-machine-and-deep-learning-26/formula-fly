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

# A frame whose pose strays further than this, in time (deviation over speed), was rendered
# at the wrong instant: half a 20 ms step. AC blends replay frames a little differently from
# linear interpolation; on the first drive that left up to 6.7 ms at 300 km/h.
MAX_TIMING_ERROR_S = 0.01
# The pose gap wanders by ~0.2 m whatever the timing (the car pitches and rolls about its
# reference point), so only at speed does it resolve time: 0.2 m at 30 m/s is 7 ms. Every
# frame is also checked by the replay frame index AC reports.
MIN_SPEED_MPS = 30.0


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
    steps = np.arange(len(frames))
    wanted = np.minimum(
        np.floor(steps / FRAME_RATE_HZ / replay.interval_s + 1e-9), len(replay.time_s) - 2
    )
    # AC reports the frame it holds with a constant offset that depends on the recording
    # rate (0 at 30 ms, +1 at 15 ms); the pose fit below measures the instant actually drawn.
    # What must not happen is the offset changing from one frame to the next.
    gap = poses[:, 0] - wanted
    values, counts = np.unique(gap, return_counts=True)
    index_offset = values[np.argmax(counts)]
    clamped = wanted == len(replay.time_s) - 2
    wrong_index = np.flatnonzero((gap != index_offset) & ~clamped)
    if len(wrong_index):
        print(
            f"AC held the wrong replay frame at steps {wrong_index[:10].tolist()}", file=sys.stderr
        )
        return 1
    offset_s, error_m = fit_render_offset(replay, poses[:, 1:], min_speed_mps=MIN_SPEED_MPS)
    times = np.arange(len(frames)) / FRAME_RATE_HZ + offset_s
    speed = np.interp(times, replay.time_s, replay.speed_mps)
    moving = speed >= MIN_SPEED_MPS
    error_s = np.where(moving, error_m / np.maximum(speed, MIN_SPEED_MPS), 0.0)
    bad = np.flatnonzero(error_s > MAX_TIMING_ERROR_S)
    print(
        f"{len(frames)} frames, all at the replay frame asked for "
        f"(AC's index {index_offset:+.0f}); AC drew each "
        f"{offset_s * 1000:+.0f} ms from its instant; timing error over 30 m/s: median "
        f"{np.median(error_s[moving]) * 1000:.1f} ms, "
        f"max {error_s.max() * 1000:.1f} ms, {len(bad)} over {MAX_TIMING_ERROR_S * 1000:.0f} ms"
    )
    if len(bad):
        print(f"first misaligned frames: {bad[:10].tolist()}", file=sys.stderr)
        return 1

    # The shift can carry the last frame or so past the replay's end; those have no inputs.
    within = int(np.sum(times <= replay.duration_s))
    laps = HumanLaps.from_replay(replay, add_ac_sidebar(frames[:within]), time_offset_s=offset_s)
    laps.save(args.out)
    per_lap = {
        int(lap): int(n) for lap, n in zip(*np.unique(laps.laps, return_counts=True), strict=True)
    }
    controls = replay_controls(replay, times[:within])
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
