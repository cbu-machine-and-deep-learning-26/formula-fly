"""Drive the practice track with the training teacher alone: no eye, no policy (GH-27).

    python scripts/teacher_lap.py                    # 85% of the centreline speed profile
    python scripts/teacher_lap.py --fraction 0.9 --start-m 500

The teacher reads the same env info the trainer gives it (arclength, speed, lateral offset,
heading error), so a lap here checks the teacher, the env's info and their sign conventions
together. On Nov 5, 2026 it lapped Silverstone in 135.6 s at 0.85 and 128.7 s at 0.9.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fly_driver.envs.practice_track import PracticeTrack  # noqa: E402
from fly_driver.interface import ControlVector  # noqa: E402
from fly_driver.training.teacher import DrivingTeacher  # noqa: E402


def main() -> None:
    """Drive until a lap, a crash or the step cap, printing every 10 s."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fraction", type=float, default=0.85)
    parser.add_argument("--lookahead-s", type=float, default=0.3)
    parser.add_argument("--start-m", type=float, help="start here instead of the grid")
    parser.add_argument("--max-steps", type=int, default=15000)
    args = parser.parse_args()

    kwargs: dict = {"max_steps": args.max_steps, "track_limit": 0.15}
    if args.start_m is not None:
        zone = [[args.start_m, args.start_m + 0.5]]
        kwargs.update(random_start=True, start_focus={"share": 0.999, "zones": zone})
    env = PracticeTrack(**kwargs)
    teacher = DrivingTeacher.for_centerline(
        env.centerline, fraction=args.fraction, lookahead_s=args.lookahead_s
    )
    _, info = env.reset(seed=0)
    for step in range(args.max_steps):
        state = [info[key] or 0.0 for key in ("arclength_m", "speed_mps", "lateral_m")]
        state.append(info["heading_error_rad"])
        control = ControlVector.clipped(*(float(x) for x in teacher.targets(*state)[0]))
        _, _, terminated, truncated, info = env.step(control)
        if step % 500 == 0:
            print(
                f"t={step / 50:6.1f}s s={state[0]:7.1f} m v={state[1] * 3.6:5.0f} km/h "
                f"lateral={state[2]:+5.2f} m steer={control.steer:+.2f}"
            )
        if info["lap_complete"]:
            print(f"LAP COMPLETE in {info['lap_time']:.1f} s")
            return
        if terminated or truncated:
            print(
                f"ENDED at {info['arclength_m']:.0f} m, {info['speed_mps'] * 3.6:.0f} km/h, "
                f"crashed={info['crashed']} stalled={info['stalled']}"
            )
            return


if __name__ == "__main__":
    main()
