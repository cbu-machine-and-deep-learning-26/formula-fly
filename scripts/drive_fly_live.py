"""Live: watch the fly body drive the car in MuJoCo's own window, in real time.

**Must be run under `mjpython`, directly from your own terminal.** macOS requires
`mujoco.viewer.launch_passive`'s window to be created by a process that is itself the
direct foreground process of a real terminal session. Launched any other way --
including everything this repo's automation can do -- it crashes with
``NSWindow should only be instantiated on the main thread``, reproduced with every
other variable removed. Nothing to route around; run it yourself::

    ./.venv-flybody/bin/mjpython scripts/drive_fly_live.py
    ./.venv-flybody/bin/mjpython scripts/drive_fly_live.py --input keyboard

Default is the scripted autopilot weave. ``--input keyboard`` puts your held arrow keys
(same as `scripts/drive.py`) in as the *intent*; the fly's body still turns that into
whatever the cockpit actually reads, same as it does the autopilot's intent -- your input
does not bypass the body, it goes through it.

ESC/SPACE/BACKSPACE/[/] are the viewer's own bindings, same as `scripts/drive.py`.
Close the window or Ctrl-C to stop.

**No longer real time, and that is the correct trade, not a regression.** The body used
to call flybody's ``env.step()`` once per frame; that was 1/100th of a frame's worth of
simulated time (see ``_substeps_per_frame``'s docstring), fast but simulating a different,
wrong-speed fly. Fixed, it costs what a fly actually costs: ~1.8 s of wall clock per
20 ms frame on this machine, ~90x too slow for 50 Hz. The car will visibly crawl relative
to the fly body process, which is exactly the same category of problem the brain track
(#23) already found and measured for its own reasons -- correctness and a real-time
budget are different constraints, and nothing here has been sped up to hide the gap.
Watch `scripts/drive_fly.py`'s recorded video for something that plays back at a normal
pace; use this script to watch the live hand-off actually happen, slowly.

**Two processes, live instead of record-then-replay.** `scripts/drive_fly.py`'s docstring
has the full story: flybody's dm_control physics and the car's raw `mujoco.viewer`
corrupt each other's frames when both are alive in one process. There, the fix was
compute all the controls first, then render in a fresh process. Here neither phase can
finish before the other starts -- it has to be live -- so instead a background
subprocess (plain Python, opens no GL context at all) steps the fly body continuously
and writes the realised control out; this process (mjpython, the live viewer) reads the
latest one every frame. The file write is atomic (write to a temp path, then
`os.replace`) so the viewer never reads a torn write, only ever a complete one that may
be a frame or two stale.
"""

from __future__ import annotations

import argparse
import atexit
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import mujoco.viewer
import numpy as np

from fly_driver.envs.car import CarConfig
from fly_driver.envs.scene import SceneConfig
from fly_driver.interface import ControlVector
from scripts.drive import CONTROL_HZ, build, reset_to_start

CONTROL_FILE = Path(__file__).resolve().parent / ".drive_fly_live_control.npy"


def _read_latest_control() -> ControlVector:
    """The viewer's side of the hand-off: read whatever the body process wrote last."""
    try:
        values = np.load(CONTROL_FILE)
        return ControlVector(
            steer=float(values[0]), throttle=float(values[1]), brake=float(values[2])
        )
    except (FileNotFoundError, ValueError, OSError):
        # Not written yet, or read mid-replace on a filesystem where that races anyway.
        return ControlVector.neutral()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        choices=("scripted", "keyboard"),
        default="scripted",
        help="scripted: autopilot weave. keyboard: your arrow keys -- needs Accessibility "
        "permission granted to your terminal app first (System Settings > Privacy & "
        "Security > Accessibility), or held keys are silently never seen",
    )
    args = parser.parse_args()

    try:
        import mujoco.viewer as _  # noqa: F401 -- fails clearly here, not deep in launch_passive
    except ImportError:
        print("mujoco.viewer is unavailable; this needs a desktop with OpenGL.", file=sys.stderr)
        return 1

    body_process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve().parent / "_drive_fly_body_stream.py"),
            "--out",
            str(CONTROL_FILE),
            "--input",
            args.input,
        ]
    )
    atexit.register(body_process.terminate)

    car = CarConfig()
    scene = SceneConfig()
    centerline, model = build(car, scene)
    data = mujoco.MjData(model)
    reset_to_start(model, data)

    from fly_driver.envs.car import CarDynamics

    dynamics = CarDynamics(model, car)
    substeps = max(1, int(round((1.0 / CONTROL_HZ) / model.opt.timestep)))
    del centerline, scene  # only the dynamics and the rendered model are needed live

    print(__doc__)
    print("waiting for the fly body process to start writing controls...")
    for _ in range(100):
        if CONTROL_FILE.exists():
            break
        time.sleep(0.05)

    try:
        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            last_report = 0.0
            while viewer.is_running():
                step_start = time.perf_counter()

                control = _read_latest_control()
                dynamics.step(control, data, substeps)
                viewer.sync()

                now = time.perf_counter()
                if now - last_report > 0.2:
                    last_report = now
                    speed = dynamics.speed_mps(data)
                    print(
                        f"\rsteer {control.steer:+.2f} thr {control.throttle:.2f} "
                        f"brk {control.brake:.2f} | {speed * 2.237:5.1f} mph",
                        end="",
                        flush=True,
                    )

                remaining = (1.0 / CONTROL_HZ) - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        body_process.terminate()
        CONTROL_FILE.unlink(missing_ok=True)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
