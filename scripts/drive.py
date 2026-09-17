"""Drive the practice-track car by hand in the MuJoCo viewer.

A development tool, not part of the training loop. It exists because several GH-16 bugs
were invisible in the numbers and only showed up by looking -- kerbs rendered back-faced,
the near clipping plane cutting off everything within 85 m, the camera buried in the
bodywork. Driving the thing yourself is the fastest way to catch the next one.

Run it::

    ./.venv/Scripts/python.exe scripts/drive.py

Controls -- **arrow keys only**, by design:

===========  ==================================================================
  UP / DOWN  one longitudinal axis. UP adds throttle a step at a time. DOWN
             lifts off completely in one press; a second press is half brake
             and a third is full brake. It holds where you leave it, like
             cruise control, because the viewer reports presses, not held keys.
  LEFT/RIGHT steer. Recentres on its own when you stop pressing, and a press
             means less lock the faster you go (see ``steering_gain``).
===========  ==================================================================

Everything else is MuJoCo's own viewer binding, and those take precedence:

===========  ==================================================================
  [ and ]    cycle cameras -- press ] to sit in the fly_head camera.
  ESC        back to the free camera (orbit with the mouse).
  SPACE      pause / resume the physics.
  BACKSPACE  reset the simulation to the start line.
  F1         the viewer's full shortcut list.
===========  ==================================================================

WASD is deliberately unused. The viewer binds letter keys to render flags -- W
toggles wireframe, and number keys toggle geom-group visibility -- so a driving
control on a letter fires the render flag *as well as* the control. An earlier
version of this script bound throttle to W and the brake to SPACE, which meant
braking also paused the simulation.

``--export model.xml`` writes the generated MJCF instead of launching, so you can open it
with ``python -m mujoco.viewer --mjcf=model.xml`` and drag the raw actuator sliders.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco

from fly_driver.envs.car import (
    CarConfig,
    CarDynamics,
    car_actuators_xml,
    car_assets_xml,
    car_body_xml,
)
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.scene import SceneConfig, build_scene_xml
from fly_driver.interface import ControlVector

# GLFW key codes. Spelled out rather than imported so this file does not depend on glfw.
# Only the arrows are used: every letter and digit is already a viewer render-flag toggle.
KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN = 263, 262, 265, 264

#: Throttle added per UP press.
PEDAL_STEP = 0.15
#: Brake added per DOWN press. Deliberately much coarser than the throttle: from coasting,
#: two presses is full brake. At the throttle's step it took seven presses to get there,
#: which is what "the brakes barely work" turned out to mean -- the car was fine, the
#: pedal was slow.
BRAKE_STEP = 0.5
STEER_STEP = 0.35
#: Fraction of steering kept per control step with no key pressed. At 50 Hz this sets a
#: time constant, and it has been wrong in both directions. 0.90 decayed in 0.2 s, so a
#: press was gone before the car could respond ("turning is like 5 degrees"). 0.995 held
#: for four seconds, so the car would not straighten up after a corner. 0.96 gives about
#: half a second, which is roughly how fast a real wheel self-centres when released.
STEER_RECENTRE = 0.96
CONTROL_HZ = 50

#: Speed-sensitive steering, for the keyboard and only the keyboard. A key press gives a
#: fixed STEER_STEP of lock; at 300 km/h that is far more than any driver would apply and
#: the car snaps sideways. The gain scales the *input* down with speed, so a press is a
#: small correction at racing speed and a full turn-in at walking pace. It is not part of
#: the car and is never applied to anyone else's ControlVector: a wheel, or the fly's
#: wingbeat asymmetry (GH-21), are analog and can be gentle on their own.
STEER_GAIN_REFERENCE_MPS = 40.0
STEER_GAIN_FLOOR = 0.2


def steering_gain(speed_mps: float) -> float:
    """Multiplier on keyboard steering: 1 at rest, falling to a floor at speed."""
    ratio = max(0.0, speed_mps) / STEER_GAIN_REFERENCE_MPS
    return max(STEER_GAIN_FLOOR, 1.0 / (1.0 + ratio * ratio))


class DriverState:
    """Mutable control state shared between the key callback and the sim loop.

    Throttle and brake are one signed ``pedal`` axis rather than two controls. That is
    what lets the whole thing run on four arrow keys, which is the point: every letter
    and digit the viewer sees is already bound to a render flag.
    """

    def __init__(self) -> None:
        self.steer = 0.0
        self.pedal = 0.0

    def on_key(self, keycode: int) -> None:
        if keycode == KEY_UP:
            # From the brakes, UP releases them; only then does it add throttle.
            self.pedal = 0.0 if self.pedal < 0.0 else min(1.0, self.pedal + PEDAL_STEP)
        elif keycode == KEY_DOWN:
            # From the throttle, DOWN is a full lift in one press; from coasting it brakes.
            self.pedal = 0.0 if self.pedal > 0.0 else max(-1.0, self.pedal - BRAKE_STEP)
        elif keycode == KEY_LEFT:
            self.steer = max(-1.0, self.steer - STEER_STEP)
        elif keycode == KEY_RIGHT:
            self.steer = min(1.0, self.steer + STEER_STEP)

    def settle(self) -> None:
        """Apply per-step decay. Steering recentres; the pedal axis holds."""
        self.steer *= STEER_RECENTRE
        if abs(self.steer) < 1e-3:
            self.steer = 0.0

    @property
    def throttle(self) -> float:
        return max(0.0, self.pedal)

    @property
    def brake(self) -> float:
        return max(0.0, -self.pedal)


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

    # Everything goes through CarDynamics, which is the only path that applies
    # aerodynamics. Setting data.ctrl directly here would let you hand-drive a car with no
    # downforce while the trained policy drove a different one.
    dynamics = CarDynamics(model, car)
    car_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
    state = DriverState()
    substeps = max(1, int(round((1.0 / CONTROL_HZ) / model.opt.timestep)))

    print(__doc__)
    print(f"lap length {centerline.length:.0f} m\n")

    with mujoco.viewer.launch_passive(
        model, data, key_callback=state.on_key, show_left_ui=False, show_right_ui=False
    ) as viewer:
        last_report = 0.0
        while viewer.is_running():
            step_start = time.perf_counter()

            speed = dynamics.speed_mps(data)
            control = ControlVector.clipped(
                steer=state.steer * steering_gain(speed),
                throttle=state.throttle,
                brake=state.brake,
            )
            dynamics.step(control, data, substeps)
            state.settle()
            viewer.sync()

            now = time.perf_counter()
            if now - last_report > 0.5:
                last_report = now
                position = data.xpos[car_body]
                projection = centerline.project(float(position[0]), float(position[1]))
                where = "on track" if projection.is_on_track else "OFF"
                print(
                    f"\r{speed * 3.6:6.1f} km/h | "
                    f"lap {projection.arclength / centerline.length * 100:5.1f}% | "
                    f"{projection.lateral:+6.2f} m {where:>8} | "
                    f"gear {dynamics.gear + 1} | "
                    f"thr {state.throttle:4.2f} brk {state.brake:4.2f} "
                    f"steer {state.steer:+5.2f}",
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
