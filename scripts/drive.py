"""Drive the practice-track car by hand in the MuJoCo viewer.

A development tool, not part of the training loop. It exists because several GH-16 bugs
were invisible in the numbers and only showed up by looking -- kerbs rendered back-faced,
the near clipping plane cutting off everything within 85 m, the camera buried in the
bodywork. Driving the thing yourself is the fastest way to catch the next one.

Run it::

    ./.venv/Scripts/python.exe scripts/drive.py
    ./.venv/Scripts/python.exe scripts/drive.py --input keyboard
    ./.venv/Scripts/python.exe scripts/drive.py --raw-steer

The first uses a gamepad if one is plugged in and the keyboard otherwise. ``--raw-steer``
gives the keyboard literal full lock at any speed -- what the fly gets for ``steer=1``.

The car always receives a :class:`~fly_driver.interface.ControlVector`, and that vector is
**analog** -- steer in [-1, 1], throttle and brake in [0, 1]. Whoever drives decides how
much of the range they use. The fly (GH-21) and a gamepad can command 50%; a keyboard key
cannot, so it commands all or nothing. The car does not know the difference.

Keyboard -- arrows only, **held** keys:

===========  ==================================================================
  UP         100% throttle while held, 0 when released.
  DOWN       100% brake while held.
  LEFT/RIGHT full steering lock while held. Scaled down with speed unless
             ``--raw-steer``, because on a keyboard there is no such thing as
             a small correction at 250 km/h (see ``steering_gain``).
===========  ==================================================================

Gamepad (Xbox or PlayStation, through GLFW's built-in mappings) -- analog. Steering is on the
right stick, Payton's preference after driving with the left:

===========  ==================================================================
  right stick X   steering, with a small deadzone so a resting stick is centred.
  right trigger   throttle, 0 to 1.
  left trigger    brake, 0 to 1.
===========  ==================================================================

A telemetry panel sits in the bottom-left of the viewer window: speed, gear, an rpm bar
with a row of shift lights (the box shifts for itself; the lights show where it will),
throttle and brake as bars, steering as a bar running -1 to +1 as the ControlVector does,
and the travel of each coilover. ``--no-hud`` suppresses it.

Everything else is MuJoCo's own viewer binding, and those take precedence:

===========  ==================================================================
  [ and ]    cycle cameras -- press ] to sit in the fly_head camera.
  ESC        back to the free camera (orbit with the mouse).
  SPACE      pause / resume the physics.
  BACKSPACE  reset the simulation to the start line.
  F1         the viewer's full shortcut list.
===========  ==================================================================

Why a separate keyboard listener: the viewer's own ``key_callback`` reports *presses* only
-- no releases and no held state -- which is what forced every earlier version of this
script into stepped inputs and decay constants that were wrong in both directions. pynput
reports both edges, so a held key can mean what it says. It listens globally, whichever
window has focus; fine for a dev tool, worth knowing.

``--export model.xml`` writes the generated MJCF instead of launching, so you can open it
with ``python -m mujoco.viewer --mjcf=model.xml`` and drag the raw actuator sliders.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Protocol

import mujoco
import numpy as np

from fly_driver.envs.car import CarConfig, CarDynamics, assemble_model_xml
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.scene import SceneConfig
from fly_driver.hud import Telemetry, ViewerHUD
from fly_driver.interface import ControlVector

CONTROL_HZ = 50

#: Speed-sensitive steering, for the keyboard and only the keyboard. A held key is full
#: lock; at 300 km/h that is far more than any driver would apply and the car snaps
#: sideways. The gain scales the *keyboard* input down with speed, so a held key is a
#: small correction at racing speed and a full turn-in at walking pace. It is not part of
#: the car and never touches anyone else's ControlVector: a gamepad stick, or the fly's
#: wingbeat asymmetry (GH-21), are analog and can be gentle on their own.
STEER_GAIN_REFERENCE_MPS = 40.0
STEER_GAIN_FLOOR = 0.2

#: Keyboard keys used, by pynput name. Arrows only: the viewer owns letters and digits
#: (W toggles wireframe, digits toggle geom groups) and handles them as well as passing
#: them on, so a driving control on a letter fires a render flag too.
KEYS = ("up", "down", "left", "right")

#: GLFW gamepad axis indices. Spelled out so the mapping is testable without GLFW; a test
#: checks them against the real constants.
AXIS_STEER = 2  # right stick, X
AXIS_LEFT_TRIGGER = 4
AXIS_RIGHT_TRIGGER = 5
#: Stick travel below this is treated as centred. A resting stick rarely reads exactly 0,
#: and without a deadzone the car creeps sideways with nobody touching anything.
GAMEPAD_DEADZONE = 0.08


def steering_gain(speed_mps: float) -> float:
    """Multiplier on keyboard steering: 1 at rest, falling to a floor at speed."""
    ratio = max(0.0, speed_mps) / STEER_GAIN_REFERENCE_MPS
    return max(STEER_GAIN_FLOOR, 1.0 / (1.0 + ratio * ratio))


class InputSource(Protocol):
    name: str

    def control(self, speed_mps: float) -> ControlVector: ...
    def stop(self) -> None: ...


class KeyboardInput:
    """Held arrow keys, all or nothing.

    ``held`` is the set of key names currently down. The pynput listener mutates it from
    its own thread; :meth:`control` reads it. Tests set it directly, which is why the
    mapping does not depend on pynput being installed.

    Args:
        raw_steer: Command literal full lock regardless of speed, which is exactly what
            the fly gets for ``steer=1.0``. Off by default because on a keyboard it makes
            anything above ~100 km/h untestable by hand.
    """

    name = "keyboard"

    def __init__(self, *, raw_steer: bool = False) -> None:
        self.held: set[str] = set()
        self.raw_steer = raw_steer
        self._listener = None

    def start(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as exc:  # pragma: no cover - depends on the local install
            raise SystemExit(
                "keyboard driving needs pynput: "
                "./.venv/Scripts/python.exe -m pip install -e '.[dev]'"
            ) from exc

        names = {
            keyboard.Key.up: "up",
            keyboard.Key.down: "down",
            keyboard.Key.left: "left",
            keyboard.Key.right: "right",
        }

        def on_press(key):
            name = names.get(key)
            if name:
                self.held.add(name)

        def on_release(key):
            name = names.get(key)
            if name:
                self.held.discard(name)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    def control(self, speed_mps: float) -> ControlVector:
        steer = float("right" in self.held) - float("left" in self.held)
        if not self.raw_steer:
            steer *= steering_gain(speed_mps)
        return ControlVector.clipped(
            steer=steer,
            throttle=float("up" in self.held),
            brake=float("down" in self.held),
        )


def gamepad_axes_to_control(
    axes: list[float] | np.ndarray, *, deadzone: float = GAMEPAD_DEADZONE
) -> ControlVector:
    """Map GLFW gamepad axes onto a ControlVector.

    GLFW reports every axis in [-1, 1], including the triggers, where -1 is released and
    +1 is fully pressed -- hence the ``(x + 1) / 2``. The stick is rescaled past the
    deadzone so the edge of the deadzone is 0 and full deflection is still 1; a plain cut
    would leave the first 8% of travel dead and the rest unreachable.
    """
    if len(axes) <= AXIS_RIGHT_TRIGGER:
        raise ValueError(f"expected at least {AXIS_RIGHT_TRIGGER + 1} axes, got {len(axes)}")
    if not 0.0 <= deadzone < 1.0:
        raise ValueError(f"deadzone must be in [0, 1), got {deadzone}")

    raw = float(axes[AXIS_STEER])
    if abs(raw) < deadzone:
        steer = 0.0
    else:
        steer = float(np.sign(raw)) * (abs(raw) - deadzone) / (1.0 - deadzone)
    throttle = (float(axes[AXIS_RIGHT_TRIGGER]) + 1.0) / 2.0
    brake = (float(axes[AXIS_LEFT_TRIGGER]) + 1.0) / 2.0
    return ControlVector.clipped(steer=steer, throttle=throttle, brake=brake)


class GamepadInput:
    """An analog controller through GLFW's gamepad API, which the viewer already ships.

    Must be created on the main thread before the viewer launches: GLFW's joystick
    functions are main-thread only, and the passive viewer runs its own GLFW work on a
    background thread.
    """

    name = "gamepad"

    def __init__(self, joystick_id: int, label: str) -> None:
        self._jid = joystick_id
        self.name = f"gamepad ({label})"

    @classmethod
    def detect(cls) -> GamepadInput | None:
        """The first connected controller GLFW recognises as a gamepad, or None."""
        import glfw

        if not glfw.init():
            return None
        for jid in range(glfw.JOYSTICK_1, glfw.JOYSTICK_LAST + 1):
            if glfw.joystick_present(jid) and glfw.joystick_is_gamepad(jid):
                label = glfw.get_gamepad_name(jid)
                if isinstance(label, bytes):
                    label = label.decode(errors="replace")
                return cls(jid, str(label))
        return None

    def control(self, speed_mps: float) -> ControlVector:
        del speed_mps  # analog input needs no speed assist
        import glfw

        state = glfw.get_gamepad_state(self._jid)
        if state is None:  # unplugged mid-run: coast rather than hold the last input
            return ControlVector.neutral()
        return gamepad_axes_to_control(list(state.axes))

    def stop(self) -> None:
        return None


def choose_input(mode: str, *, raw_steer: bool) -> InputSource:
    """``auto`` prefers a connected gamepad and falls back to the keyboard."""
    if mode in ("auto", "gamepad"):
        pad = GamepadInput.detect()
        if pad is not None:
            return pad
        if mode == "gamepad":
            raise SystemExit("no gamepad detected -- plug one in, or use --input keyboard")
    keyboard = KeyboardInput(raw_steer=raw_steer)
    keyboard.start()
    return keyboard


def build(car: CarConfig, scene: SceneConfig) -> tuple[Centerline, mujoco.MjModel]:
    centerline = Centerline.load()
    return centerline, mujoco.MjModel.from_xml_string(assemble_model_xml(centerline, scene, car))


def reset_to_start(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the car back on the start line. qpos0 holds the pose baked into the MJCF."""
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, help="write the MJCF here instead of driving")
    parser.add_argument("--fovy", type=float, default=None, help="camera vertical FOV, degrees")
    parser.add_argument(
        "--input",
        choices=("auto", "keyboard", "gamepad"),
        default="auto",
        help="auto uses a gamepad if one is plugged in, otherwise the keyboard",
    )
    parser.add_argument(
        "--raw-steer",
        action="store_true",
        help="keyboard: literal full lock at any speed, as the fly would get for steer=1",
    )
    parser.add_argument("--walls", action="store_true", help="add collidable walls at the edges")
    parser.add_argument("--no-hud", action="store_true", help="do not draw the telemetry panel")
    args = parser.parse_args(argv)

    car = CarConfig(camera_fovy_deg=args.fovy) if args.fovy else CarConfig()
    scene = SceneConfig(include_walls=args.walls)
    centerline, model = build(car, scene)

    if args.export:
        args.export.write_text(assemble_model_xml(centerline, scene, car), encoding="utf-8")
        print(f"wrote {args.export}")
        print(f"open it with:  python -m mujoco.viewer --mjcf={args.export}")
        return 0

    try:
        import mujoco.viewer
    except ImportError:  # pragma: no cover - depends on the local install
        print("mujoco.viewer is unavailable; this needs a desktop with OpenGL.", file=sys.stderr)
        return 1

    # Input is chosen before the viewer launches: gamepad detection must run on the main
    # thread before the viewer starts its own GLFW work.
    source = choose_input(args.input, raw_steer=args.raw_steer)

    data = mujoco.MjData(model)
    reset_to_start(model, data)

    # Everything goes through CarDynamics, which is the only path that applies
    # aerodynamics. Setting data.ctrl directly here would let you hand-drive a car with no
    # downforce while the trained policy drove a different one.
    dynamics = CarDynamics(model, car)
    car_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
    substeps = max(1, int(round((1.0 / CONTROL_HZ) / model.opt.timestep)))

    print(__doc__)
    print(f"input: {source.name}")
    print(f"lap length {centerline.length:.0f} m\n")

    try:
        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            hud = None
            if not args.no_hud:
                limiter_rpm = dynamics.powertrain.max_engine_rads * 60.0 / (2.0 * np.pi)
                hud = ViewerHUD(
                    viewer,
                    limiter_rpm=limiter_rpm,
                    shift_rpm=limiter_rpm * dynamics.powertrain.shift_up_fraction,
                )
            last_report = 0.0
            while viewer.is_running():
                step_start = time.perf_counter()

                speed = dynamics.speed_mps(data)
                control = source.control(speed)
                dynamics.step(control, data, substeps)
                viewer.sync()

                if hud is not None:
                    travel = dynamics.suspension_travel(data)
                    hud.update(
                        Telemetry(
                            speed_kmh=speed * 3.6,
                            gear=dynamics.gear + 1,
                            rpm=dynamics.engine_rpm(data),
                            throttle=control.throttle,
                            brake=control.brake,
                            steer=control.steer,
                            suspension=tuple(
                                travel[side] / car.suspension_travel_m
                                for side in ("fl", "fr", "rl", "rr")
                            ),
                        )
                    )

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
                        f"thr {control.throttle:4.2f} brk {control.brake:4.2f} "
                        f"steer {control.steer:+5.2f}",
                        end="",
                        flush=True,
                    )

                # Keep the sim near wall-clock so it feels like driving, not fast-forward.
                remaining = (1.0 / CONTROL_HZ) - (time.perf_counter() - step_start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        source.stop()

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
