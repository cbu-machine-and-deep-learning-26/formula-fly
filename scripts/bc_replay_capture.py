"""Re-render an AC replay through the nose camera, one 96x96 frame per 20 ms (GH-29).

Runs on the AC box, in AC's desktop session and AssettoCorsaGym's Python 3.9 env (it
needs ``mss``, ``cv2`` and ``pywin32``, which that env has). AC must already be playing the
replay, with ``scripts/bc_replay_app`` installed as ``apps/lua/fly_bc_replay``. For each
replay step ``k`` this asks the Lua app to hold the replay at ``k * 20 ms``, waits until
that instant is on screen, and grabs the window with the ``sensors_par`` plugin's own
``capture_and_process``, so the frames match what the env server sends in training.

Writes ``<out>.frames.npy`` (``(n, 96, 96, 3)`` uint8 RGB, frame ``k`` at ``k / 50`` s) and
``<out>.pose.npy`` (``(n, 4)``: the replay frame index AC reports and the car's x, y, z),
for steps ``--start`` onwards (0 for a dataset),
which ``scripts/bc_build_dataset.py`` checks against the replay file. See
``docs/assetto-corsa-training.md`` § Behaviour cloning.
"""

from __future__ import annotations

import argparse
import mmap
import struct
import sys
import time
from pathlib import Path

import numpy as np

STEP_S = 0.02
# int want, shown, frames, current; float frameMs; int replay; float x, y, z;
# int heartbeat, camera, drivable
LAYOUT = struct.Struct("<4if i3f3i")
DRIVABLE, DASH = 2, 4


def _read(shared: mmap.mmap) -> tuple:
    return LAYOUT.unpack_from(shared, 0)


def _wait(shared: mmap.mmap, ready, timeout_s: float, what: str) -> tuple:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        state = _read(shared)
        if ready(state):
            return state
        time.sleep(0.002)
    raise TimeoutError(f"no {what} after {timeout_s} s; last state {_read(shared)}")


def _wait_for_foreground(win32gui, window: int, step: int, timeout_s: float = 600.0) -> None:
    """Block while another window is in front of AC, e.g. someone using the console."""
    if win32gui.GetForegroundWindow() == window:
        return
    print(f"step {step}: AC is not the foreground window; waiting", flush=True)
    deadline = time.monotonic() + timeout_s
    while win32gui.GetForegroundWindow() != window:
        if time.monotonic() > deadline:
            raise TimeoutError(f"AC was not the foreground window for {timeout_s} s")
        time.sleep(0.5)
    time.sleep(0.5)  # let AC redraw over whatever was there


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True, help="output prefix")
    parser.add_argument(
        "--ac-root",
        type=Path,
        default=Path(r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa"),
    )
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--start", type=int, default=0, help="first step (a test)")
    parser.add_argument("--limit", type=int, help="stop after this many frames (a test)")
    args = parser.parse_args()

    sys.path.insert(0, str(args.ac_root / "apps" / "python" / "sensors_par"))
    import screen_capture  # the plugin's own grab and resize
    import win32gui

    window = win32gui.FindWindow(None, "Assetto Corsa")
    if not window:
        raise RuntimeError("the AC window was not found")

    shared = mmap.mmap(-1, LAYOUT.size, tagname="Local\\fly_bc_replay")
    state = _wait(
        shared,
        lambda s: s[5] == 1 and s[2] > 1 and s[10] == DRIVABLE and s[11] == DASH,
        120.0,
        "replay on the nose camera (is the Lua app installed and AC in the replay?)",
    )
    frame_count, frame_ms = state[2], state[4]
    steps = int((frame_count - 1) * frame_ms / 1000.0 / STEP_S) + 1 - args.start
    if args.limit:
        steps = min(steps, args.limit)
    print(f"replay: {frame_count} frames of {frame_ms} ms -> {steps} steps", flush=True)

    frames = np.zeros((steps, args.size, args.size, 3), np.uint8)
    poses = np.zeros((steps, 4), np.float64)
    started = time.monotonic()
    for k in range(steps):
        step = args.start + k
        struct.pack_into("<i", shared, 0, step + 1)
        state = _wait(shared, lambda s, want=step + 1: s[1] == want, 10.0, f"step {step} on screen")
        # The grab is of the screen where AC's window is: anything on top of it is captured.
        _wait_for_foreground(win32gui, window, step)
        image, _ = screen_capture.capture_and_process(args.size, args.size, color_mode="bgr")
        if image is None:
            raise RuntimeError("the AC window was not found")
        frames[k] = image[..., ::-1]  # BGR -> RGB, as AssettoCorsaCar does
        poses[k] = (state[3], state[6], state[7], state[8])
        if k % 500 == 0:
            rate = (k + 1) / (time.monotonic() - started)
            print(f"step {k}/{steps} replay frame {state[3]} at {rate:.1f} steps/s", flush=True)
    struct.pack_into("<i", shared, 0, 0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(f"{args.out}.frames.npy", frames)
    np.save(f"{args.out}.pose.npy", poses)
    print(f"wrote {steps} frames in {time.monotonic() - started:.0f} s to {args.out}.*", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
