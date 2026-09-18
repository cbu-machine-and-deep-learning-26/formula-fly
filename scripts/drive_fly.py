"""Watch the tethered fly body (GH-21) drive the practice-track car.

The scripted intent is what a keyboard or a trained policy would send straight to the
car in `scripts/drive.py`. Here it goes through `CombinedBody` first: flybody's real
wing and leg physics turn that intent into whatever the cockpit actually reads, and
that realised control is what drives the car. This is "the fly is the car" made literal
and watchable, not a metaphor.

Run it::

    ./.venv/Scripts/python.exe scripts/drive_fly.py --seconds 20 --out fly_driving.mp4

**Two subprocesses, on purpose.** flybody's dm_control physics (for the wings and legs)
and the car's raw `mujoco.Renderer` corrupt each other's frames -- not a metaphor either,
a reproduced finding -- when both are alive in the same process: recording video while
stepping the body produced solid static, reproducible and not fixed by the width-padding
red herring imageio warns about (macro_block_size). Isolating them fixed it: one process
steps the body and writes out the realised controls, a second, separate process (no
flybody import at all) replays those controls through the car and renders. This script
is the orchestrator; `--phase controls` and `--phase render` are its own two halves, run
as subprocesses rather than inline so the GPU/GL context each opens is never shared.

Also **not** `mujoco.viewer`'s live window: macOS's `mjpython` (required for that window,
see `scripts/drive.py`) needs to be the direct foreground process of a real terminal
session, and crashes (``NSWindow should only be instantiated on the main thread``)
launched any other way -- confirmed by removing every other variable and still hitting
the identical crash. Not something to route around from here; use `scripts/drive.py`'s
live viewer directly from a real terminal if that is what you want, and this recorded
video for what the fly is actually doing to the car in the meantime.

Needs `.venv-flybody` (or flybody importable some other way) for `--phase controls`, and
`imageio[ffmpeg]` for `--phase render` writing `.mp4`/`.gif`/`.webm` (already a `dev`
extra for the eval harness's own video writer) -- falls back to a directory of PNG
frames for any other `--out` suffix, or if imageio has no ffmpeg plugin.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CONTROL_HZ = 50

#: mujoco.Renderer's own default is 854x480 (16:9 at 480p); 848 is the nearest width
#: divisible by imageio's ffmpeg writer's macro_block_size (16), which sidesteps its
#: automatic padding resize. Not the fix for the GL-context corruption above -- that is
#: the two-process split -- but worth keeping since the padding path is untrusted here.
FRAME_WIDTH = 848
FRAME_HEIGHT = 480


def _run_controls_phase(seconds: float, out_path: Path, track_limit: float) -> None:
    import numpy as np

    from fly_driver.body import CombinedBody, FlybodyWingBody, TrackballLegBody
    from fly_driver.interface import ControlVector

    body = CombinedBody(FlybodyWingBody(), TrackballLegBody())
    body.reset()

    n_steps = int(round(seconds * CONTROL_HZ))
    controls = np.empty((n_steps, 3), dtype=np.float32)
    period_s = 4.0
    for step in range(n_steps):
        t = step / CONTROL_HZ
        steer = float(np.sin(2 * np.pi * t / period_s))
        intent = ControlVector(steer=steer, throttle=0.5, brake=0.0)
        realised = body.actuate(intent)
        controls[step] = (realised.steer, realised.throttle, realised.brake)
        print(f"\rbody step {step + 1}/{n_steps}", end="", flush=True)
    print()

    del track_limit  # excursion handling lives in the render phase, which owns the car
    np.save(out_path, controls)


def _run_render_phase(controls_path: Path, out_path: Path, fps: int, track_limit: float) -> None:
    import mujoco
    import numpy as np

    from fly_driver.envs.car import CarConfig, CarDynamics
    from fly_driver.envs.centerline import off_track_fraction
    from fly_driver.envs.scene import SceneConfig
    from fly_driver.interface import ControlVector
    from scripts.drive import build, reset_to_start

    controls = np.load(controls_path)

    car = CarConfig()
    scene = SceneConfig()
    centerline, model = build(car, scene)
    data = mujoco.MjData(model)
    reset_to_start(model, data)
    dynamics = CarDynamics(model, car)
    car_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
    substeps = max(1, int(round((1.0 / CONTROL_HZ) / model.opt.timestep)))

    renderer = mujoco.Renderer(model, height=FRAME_HEIGHT, width=FRAME_WIDTH)
    camera = mujoco.MjvCamera()
    camera.trackbodyid = car_body_id
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.distance = 12.0
    camera.azimuth = 135.0
    camera.elevation = -20.0

    frame_every = max(1, round(CONTROL_HZ / fps))
    frames: list[np.ndarray] = []
    excursions = 0

    for step, (steer, throttle, brake) in enumerate(controls):
        dynamics.step(ControlVector(float(steer), float(throttle), float(brake)), data, substeps)

        position = data.xpos[car_body_id]
        projection = centerline.project(float(position[0]), float(position[1]))
        if track_limit > 0.0:
            beyond = off_track_fraction(
                projection, car_width_m=car.overall_width_m, kerb_width_m=scene.kerb_width_m
            )
            if beyond > track_limit:
                excursions += 1
                reset_to_start(model, data)
                dynamics.reset()

        if step % frame_every == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
            print(f"\rrender step {step + 1}/{len(controls)}", end="", flush=True)
    print(f"\ncaptured {len(frames)} frames, excursions {excursions}")

    if out_path.suffix.lower() in (".mp4", ".gif", ".webm"):
        try:
            import imageio.v3 as iio

            iio.imwrite(out_path, frames, fps=fps)
            print(f"wrote {out_path}")
            return
        except Exception as error:  # imageio without an ffmpeg plugin, etc.
            print(f"video write failed ({error}); writing PNG frames instead", file=sys.stderr)
            out_path = out_path.with_suffix("")

    out_path.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(out_path / f"frame_{i:04d}.png")
    print(f"wrote {len(frames)} PNGs to {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=15.0, help="how much sim time to record")
    parser.add_argument("--out", type=Path, default=Path("fly_driving.mp4"))
    parser.add_argument("--fps", type=int, default=25, help="output video frame rate")
    parser.add_argument(
        "--track-limit",
        type=float,
        default=0.0,
        help="restart on excursion past this kerb fraction; 0 (default) just keeps recording",
    )
    parser.add_argument(
        "--phase",
        choices=("controls", "render"),
        help=argparse.SUPPRESS,  # internal: this is how the two subprocesses invoke themselves
    )
    parser.add_argument("--controls-path", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.phase == "controls":
        _run_controls_phase(args.seconds, args.controls_path, args.track_limit)
        return 0
    if args.phase == "render":
        _run_render_phase(args.controls_path, args.out, args.fps, args.track_limit)
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        controls_path = Path(tmp) / "controls.npy"
        print("phase 1/2: stepping the fly body (own process)")
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--phase",
                "controls",
                "--seconds",
                str(args.seconds),
                "--controls-path",
                str(controls_path),
            ],
            check=True,
        )
        print("phase 2/2: rendering the car (own process)")
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--phase",
                "render",
                "--controls-path",
                str(controls_path),
                "--out",
                str(args.out),
                "--fps",
                str(args.fps),
                "--track-limit",
                str(args.track_limit),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
