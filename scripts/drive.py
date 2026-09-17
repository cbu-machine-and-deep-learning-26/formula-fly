"""Drive the practice-track car by hand in the MuJoCo viewer.

A development tool, not part of the training loop. It exists because several GH-16 bugs
were invisible in the numbers and only showed up by looking -- kerbs rendered back-faced,
the near clipping plane cutting off everything within 85 m, the camera buried in the
bodywork. Driving the thing yourself is the fastest way to catch the next one.

Run it::

    ./.venv/Scripts/python.exe scripts/drive.py

Controls (press, not hold -- the viewer reports key presses, not key state):

===========  ==================================================================
  W / S      throttle up / down, in steps. Behaves like cruise control: it holds
             where you leave it.
  A / D      steer left / right. Recentres on its own when you stop pressing.
  SPACE      brake. Applies for a short pulse, then releases.
  TAB        cycle cameras -- press it to sit in the fly's head camera.
  R          reset to the start line.
  ESC        quit.
===========  ==================================================================

``--export model.xml`` writes the generated MJCF instead of launching, so you can open it
with ``python -m mujoco.viewer --mjcf=model.xml`` and drag the raw actuator sliders.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from fly_driver.envs.car import (
    CarConfig,
    car_actuators_xml,
    car_assets_xml,
    car_body_xml,
    control_to_ctrl,
)
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.scene import SceneConfig, build_scene_xml

# GLFW key codes. Spelled out rather than imported so this file does not depend on glfw.
KEY_ESCAPE, KEY_SPACE = 256, 32
KEY_A, KEY_D, KEY_R, KEY_S, KEY_W = 65, 68, 82, 83, 87
KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN = 263, 262, 265, 264

THROTTLE_STEP = 0.15
STEER_STEP = 0.25
#: Fraction of steering kept each control step when no key is pressed. Below 1.0 the
#: wheels drift back to centre, which makes press-only input feel like a real wheel.
STEER_RECENTRE = 0.90
#: Control steps a SPACE press keeps the brake applied.
BRAKE_PULSE_STEPS = 25
CONTROL_HZ = 50


class DriverState:
    """Mutable control state shared between the key callback and the sim loop."""

    def __init__(self) -> None:
        self.steer = 0.0
        self.throttle = 0.0
        self.brake_steps = 0
        self.reset_requested = False
        self.quit_requested = False

    def on_key(self, keycode: int) -> None:
        if keycode in (KEY_W, KEY_UP):
            self.throttle = min(1.0, self.throttle + THROTTLE_STEP)
        elif keycode in (KEY_S, KEY_DOWN):
            self.throttle = max(0.0, self.throttle - THROTTLE_STEP)
        elif keycode in (KEY_A, KEY_LEFT):
            self.steer = max(-1.0, self.steer - STEER_STEP)
        elif keycode in (KEY_D, KEY_RIGHT):
            self.steer = min(1.0, self.steer + STEER_STEP)
        elif keycode == KEY_SPACE:
            self.brake_steps = BRAKE_PULSE_STEPS
        elif keycode == KEY_R:
            self.reset_requested = True
        elif keycode == KEY_ESCAPE:
            self.quit_requested = True

    def settle(self) -> None:
        """Apply per-step decay. Steering recentres; the brake pulse counts down."""
        self.steer *= STEER_RECENTRE
        if abs(self.steer) < 1e-3:
            self.steer = 0.0
        self.brake_steps = max(0, self.brake_steps - 1)

    @property
    def brake(self) -> float:
        return 1.0 if self.brake_steps > 0 else 0.0


def build(car: CarConfig, scene: SceneConfig) -> tuple[Centerline, mujoco.MjModel]:
    centerline = Centerline.load()
    position, yaw = centerline.pose_at(0.0)
    xml = build_scene_xml(
        centerline,
        scene,
        extra_assets=car_assets_xml(car),
        extra_bodies=car_body_xml(position, yaw, car),
        extra_actuators=car_actuators_xml(car),
    )
    return centerline, mujoco.MjModel.from_xml_string(xml)


def reset_to_start(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the car back on the start line. qpos0 holds the pose baked into the MJCF."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="write the MJCF here instead of driving")
    parser.add_argument("--fovy", type=float, default=None, help="camera vertical FOV, degrees")
    parser.add_argument(
        "--no-walls", action="store_true", help="disable track walls (they are off by default)"
    )
    parser.add_argument("--walls", action="store_true", help="add collidable walls at the edges")
    args = parser.parse_args(argv)

    car = CarConfig(camera_fovy_deg=args.fovy) if args.fovy else CarConfig()
    scene = SceneConfig(include_walls=args.walls and not args.no_walls)
    centerline, model = build(car, scene)

    if args.export:
        position, yaw = centerline.pose_at(0.0)
        args.export.write_text(
            build_scene_xml(
                centerline,
                scene,
                extra_assets=car_assets_xml(car),
                extra_bodies=car_body_xml(position, yaw, car),
                extra_actuators=car_actuators_xml(car),
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.export}")
        print(f"open it with:  python -m mujoco.viewer --mjcf={args.export}")
        return 0

    try:
        import mujoco.viewer
    except ImportError:  # pragma: no cover - depends on the local install
        print("mujoco.viewer is unavailable; this needs a desktop with OpenGL.", file=sys.stderr)
        return 1

    data = mujoco.MjData(model)
    reset_to_start(model, data)

    actuator_ids = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in range(model.nu)
    }
    car_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
    state = DriverState()
    substeps = max(1, int(round((1.0 / CONTROL_HZ) / model.opt.timestep)))

    print(__doc__)
    print(f"lap length {centerline.length:.0f} m\n")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=state.on_key, show_left_ui=False, show_right_ui=False
    ) as viewer:
        last_report = 0.0
        while viewer.is_running() and not state.quit_requested:
            step_start = time.perf_counter()

            if state.reset_requested:
                reset_to_start(model, data)
                state.reset_requested = False

            commands = control_to_ctrl(state.steer, state.throttle, state.brake, car)
            for name, value in commands.items():
                data.ctrl[actuator_ids[name]] = value

            for _ in range(substeps):
                mujoco.mj_step(model, data)
            state.settle()
            viewer.sync()

            now = time.perf_counter()
            if now - last_report > 0.5:
                last_report = now
                position = data.xpos[car_body]
                speed = float(np.linalg.norm(data.cvel[car_body][3:5]))
                projection = centerline.project(float(position[0]), float(position[1]))
                where = "on track" if projection.is_on_track else "OFF"
                print(
                    f"\r{speed * 3.6:6.1f} km/h | "
                    f"lap {projection.arclength / centerline.length * 100:5.1f}% | "
                    f"{projection.lateral:+6.2f} m {where:>8} | "
                    f"thr {state.throttle:4.2f} steer {state.steer:+5.2f}",
                    end="",
                    flush=True,
                )

            # Keep the sim near wall-clock so it feels like driving rather than a fast-forward.
            remaining = (1.0 / CONTROL_HZ) - (time.perf_counter() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
