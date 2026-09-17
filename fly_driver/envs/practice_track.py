"""The practice-track environment (GH-16): a camera frame in, a control vector out.

This is the file the rest of the project plugs into. `AGENTS.md` §3 fixes one interface for
the whole pipeline -- observation in, :class:`~fly_driver.interface.ControlVector` out -- and
everything else in ``fly_driver.envs`` is a piece of the world rather than a way to drive
through it. The track geometry, the car, the head camera, the lap timer and the track-limits
rule all already existed and are all tested; what was missing was the loop that renders a
frame, accepts an action, and steps the two in lockstep.

Deliberately **not** a ``gymnasium.Env``. gymnasium is not a dependency, and adding one to
the default install is exactly the x86-only-wheel risk `AGENTS.md` §7 tells us to find in
week 1 rather than week 5. The method names and the ``(obs, terminated, truncated, info)``
shape are Gym's, so the wrapper GH-17 may want is a dozen lines.

Connecting the eye and the brain
--------------------------------

This module never imports flyvis or torch, and it must stay that way: the optic lobe lives
in its own virtualenv (see ``docs/running-the-stacks.md``) because flyvis 1.2.0 needs
Python <3.13 and platform-specific CUDA wheels. What makes the stages connectable is not a
shared import, it is that the frame this env emits is exactly the frame the eye declares::

    from fly_driver.envs.practice_track import PracticeTrack
    from fly_driver.eyes import FlyvisEye          # in the .venv-flyvis environment

    env = PracticeTrack()
    eye = FlyvisEye(frame_shape=env.frame_shape, frame_rate_hz=env.frame_rate_hz)

    frame = env.reset()
    eye.reset()                                   # REQUIRED at every episode start:
                                                  # the optic lobe keeps state between
                                                  # frames, so a new episode must begin
                                                  # from the grey warm-up, not from
                                                  # whatever the last crash looked like.
    while True:
        features = eye.encode(frame)              # (5768,) float32
        control = policy.act(features)            # -> ControlVector
        result = env.step(control)
        frame = result.frame
        if result.terminated or result.truncated:
            break

Take the shape and the rate **from the env** rather than typing the numbers again. Both are
constructor arguments on :class:`~fly_driver.eyes.FlyvisEye`, and the failure mode if they
drift apart is silent: the eye integrates at the wrong ``dt`` and produces plausible
nonsense. :data:`~fly_driver.interface.FRAME_RATE_HZ` explains which of the two numbers is
negotiable.

Reward
------

There is none, and that is on purpose. GH-17 owns reward shaping and its per-term CSV
logging, so this env publishes the *terms* -- distance progressed, speed, how far off track,
lap times -- in :attr:`StepResult.info` and declines to pick the function that combines
them. Anything that needs a scalar computes it from ``info``.

What the fly sees
-----------------

The painted racing line is **on** by default and is therefore in shot. That is Payton's
call for hand driving and it is recorded here so it is not quietly reversed: a painted line
is a strong, unambiguous cue, and a policy that merely follows it is answering an easier
question than "does connectome wiring help". Pass ``SceneConfig(racing_line=False)`` for
the RQ1 comparisons, or leave it on deliberately and say so in the write-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from fly_driver.envs.car import CarConfig, CarDynamics, assemble_model_xml
from fly_driver.envs.centerline import Centerline, Projection, off_track_fraction
from fly_driver.envs.lap import LapTimer
from fly_driver.envs.scene import SceneConfig
from fly_driver.interface import (
    FRAME_RATE_HZ,
    FRAME_SHAPE,
    ControlVector,
    Frame,
    validate_frame,
)

__all__ = ["DEFAULT_CAMERA", "DEFAULT_TRACK_LIMIT", "PracticeTrack", "StepResult"]

#: The camera mounted where the fly's head sits, from :func:`~fly_driver.envs.car.car_body_xml`.
DEFAULT_CAMERA = "fly_head"

#: Fraction of the car's width that may hang past the kerb before the episode ends. The
#: same 0.15 ``scripts/drive.py`` throws a hand-driven lap away at, so the fly is judged
#: under the rule the humans drive under.
DEFAULT_TRACK_LIMIT = 0.15


@dataclass(frozen=True)
class StepResult:
    """One environment step.

    Args:
        frame: The camera image after the step, ``(height, width, 3)`` uint8. Freshly
            allocated each step, so holding onto it is safe.
        terminated: The episode ended because of what happened -- the car left the
            circuit, or the physics diverged.
        truncated: The episode was cut off by ``max_steps``, not by anything the car did.
            Kept separate because bootstrapping a value function treats the two
            differently.
        info: Raw signals, never a reward. See the module docstring.
    """

    frame: Frame
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        """Either kind of ending, for callers that do not care which."""
        return self.terminated or self.truncated


class PracticeTrack:
    """Silverstone, one car, one head camera, stepped at the optic lobe's frame rate.

    Args:
        car: Vehicle parameters. Defaults to the SF70H :class:`~fly_driver.envs.car.CarConfig`.
        scene: Track and rendering settings. Defaults to
            :class:`~fly_driver.envs.scene.SceneConfig`, which starts the car
            ``grid_offset_m`` behind the line so the first lap is timed from a real
            crossing rather than from wherever the simulator booted.
        seed: Seeds :attr:`rng`. Nothing in the env is stochastic yet; the generator exists
            so that when episode randomisation arrives it has one place to come from, and
            so that the seed-determinism check `AGENTS.md` §11 asks for can be written now
            instead of retrofitted.
        max_steps: Truncate the episode after this many steps. ``None`` runs forever, which
            is what a hand-driven session or a lap-time measurement wants.
        frame_shape: ``(height, width, 3)``. Must fit inside the scene's offscreen buffer.
        frame_rate_hz: Environment steps per second of simulated time. One step advances
            the physics by exactly ``1 / frame_rate_hz`` and produces exactly one frame.
        track_limit: Fraction of the car's width past the kerb that ends the episode.
            ``0`` turns the rule off and lets the car drive across the infield.
        camera: Name of the MJCF camera to render from.

    Raises:
        ValueError: If any argument is out of range, if the frame does not fit the
            offscreen buffer, or if the frame rate does not divide the physics timestep
            into a whole number of substeps.
    """

    def __init__(
        self,
        car: CarConfig | None = None,
        scene: SceneConfig | None = None,
        *,
        seed: int = 0,
        max_steps: int | None = None,
        frame_shape: tuple[int, int, int] = FRAME_SHAPE,
        frame_rate_hz: float = FRAME_RATE_HZ,
        track_limit: float = DEFAULT_TRACK_LIMIT,
        camera: str = DEFAULT_CAMERA,
    ) -> None:
        shape = tuple(int(value) for value in frame_shape)
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError(f"frame_shape must be (height, width, 3), got {frame_shape}")
        if shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"frame_shape must be positive, got {frame_shape}")
        if frame_rate_hz <= 0:
            raise ValueError(f"frame_rate_hz must be positive, got {frame_rate_hz}")
        if not 0.0 <= track_limit <= 1.0:
            raise ValueError(f"track_limit must be in [0, 1], got {track_limit}")
        if max_steps is not None and max_steps <= 0:
            raise ValueError(f"max_steps must be positive or None, got {max_steps}")

        self.car = car or CarConfig()
        self.scene = scene or SceneConfig()
        self._frame_shape: tuple[int, int, int] = (shape[0], shape[1], 3)
        self._frame_rate_hz = float(frame_rate_hz)
        self._track_limit = float(track_limit)
        self._max_steps = max_steps
        self._camera = camera
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # MuJoCo's offscreen framebuffer is sized in the MJCF and it *raises* when asked to
        # render anything larger, which is a confusing error a long way from its cause.
        if shape[0] > self.scene.offscreen_height or shape[1] > self.scene.offscreen_width:
            raise ValueError(
                f"frame_shape {self._frame_shape} does not fit the offscreen buffer "
                f"{self.scene.offscreen_width}x{self.scene.offscreen_height}; raise "
                "SceneConfig.offscreen_width/height"
            )

        self.centerline = Centerline.load()
        self._model = mujoco.MjModel.from_xml_string(
            assemble_model_xml(self.centerline, self.scene, self.car)
        )
        self._data = mujoco.MjData(self._model)

        # Everything that moves the car goes through CarDynamics, because aerodynamics are
        # not in the MJCF -- MuJoCo knows nothing about wings -- and writing data.ctrl here
        # would give the fly a car with no downforce while the hand-driven one had some.
        self._dynamics = CarDynamics(self._model, self.car)

        self._body = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "car")
        if self._body < 0:  # pragma: no cover - assemble_model_xml always adds it
            raise ValueError("model has no body named 'car'")
        if mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera) < 0:
            raise ValueError(f"model has no camera named {camera!r}")

        # One env step is a whole number of physics substeps. Anything else would make the
        # frame interval drift away from 1 / frame_rate_hz, and since one frame is one Euler
        # step of the optic lobe, a drifting interval means the eye integrates at a dt it
        # was never fitted at. Nothing downstream can detect that, so it is checked here.
        exact = (1.0 / self._frame_rate_hz) / self._model.opt.timestep
        self._substeps = int(round(exact))
        if self._substeps < 1 or abs(exact - self._substeps) > 1e-9:
            raise ValueError(
                f"frame_rate_hz={self._frame_rate_hz} needs {exact:.4f} physics steps of "
                f"{self._model.opt.timestep} s per frame, which is not a whole number; "
                "pick a rate that divides the timestep, or change SceneConfig.timestep"
            )

        self._lap = LapTimer(self.centerline)
        self._renderer: mujoco.Renderer | None = None
        self._closed = False
        self._steps = 0
        self._previous_s = 0.0
        self._frame: Frame | None = None

    # -- read-only view of the world -------------------------------------------------

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """``(height, width, 3)``. Hand this to the eye rather than a literal."""
        return self._frame_shape

    @property
    def frame_rate_hz(self) -> float:
        """Frames -- and environment steps -- per second of simulated time."""
        return self._frame_rate_hz

    @property
    def dt(self) -> float:
        """Simulated seconds per :meth:`step`."""
        return 1.0 / self._frame_rate_hz

    @property
    def substeps(self) -> int:
        """Physics steps per :meth:`step`."""
        return self._substeps

    @property
    def model(self) -> mujoco.MjModel:
        """The compiled model. For inspection; drive the car through :meth:`step`."""
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        """Live simulation state. For inspection; drive the car through :meth:`step`."""
        return self._data

    @property
    def dynamics(self) -> CarDynamics:
        """The car's force model, for telemetry a caller wants and ``info`` does not carry."""
        return self._dynamics

    @property
    def steps(self) -> int:
        """Steps taken since the last :meth:`reset`."""
        return self._steps

    @property
    def lap_timer(self) -> LapTimer:
        """Lap detection, shared with the hand-driving tool's semantics."""
        return self._lap

    # -- the loop ---------------------------------------------------------------------

    def reset(self) -> Frame:
        """Put the car back on the grid and return the first frame.

        Whatever drives this env must reset at the same moment -- in particular a
        :class:`~fly_driver.eyes.FlyvisEye` keeps its network state between frames and needs
        its own ``reset()`` here, or the new episode starts with the old one's afterimage.
        """
        self._require_open()
        # qpos0 holds the grid pose baked into the MJCF by assemble_model_xml.
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self._dynamics.reset()
        self._steps = 0

        projection = self._project()
        self._previous_s = projection.arclength
        self._lap.reset(projection.arclength, float(self._data.time))

        self._frame = self._render()
        return self._frame

    def step(self, control: ControlVector) -> StepResult:
        """Advance one frame.

        Args:
            control: Steering, throttle and brake. Already range-checked by its own
                constructor, so an out-of-range action fails where it was built rather
                than silently saturating here.

        Raises:
            TypeError: If ``control`` is not a :class:`~fly_driver.interface.ControlVector`.
            RuntimeError: If called before :meth:`reset`, or after :meth:`close`.
        """
        self._require_open()
        if not isinstance(control, ControlVector):
            raise TypeError(
                f"step() takes a ControlVector, got {type(control).__name__}. "
                "ControlVector.clipped(...) squashes a raw policy output."
            )
        if self._frame is None:
            raise RuntimeError("call reset() before step()")

        self._dynamics.step(control, self._data, self._substeps)
        self._steps += 1
        truncated = self._max_steps is not None and self._steps >= self._max_steps

        # A diverged integrator makes every number after it meaningless, and projecting a
        # NaN position onto the centerline would return a plausible-looking arclength.
        # End the episode and say why, rather than training on garbage (AGENTS.md §11).
        if not (np.all(np.isfinite(self._data.qpos)) and np.all(np.isfinite(self._data.qvel))):
            info = self._info(
                projection=None,
                progress_m=0.0,
                beyond=1.0,
                completed=None,
                diverged=True,
            )
            return StepResult(self._frame, True, False, info)

        projection = self._project()
        progress_m = self.centerline.progress_delta(self._previous_s, projection.arclength)
        self._previous_s = projection.arclength

        # Measured from the car's outer edge, so a kerb is fair game and the grass is not.
        beyond = off_track_fraction(
            projection,
            car_width_m=self.car.overall_width_m,
            kerb_width_m=self.scene.kerb_width_m,
        )
        completed = self._lap.update(projection.arclength, float(self._data.time))
        terminated = self._track_limit > 0.0 and beyond > self._track_limit

        self._frame = self._render()
        info = self._info(
            projection=projection,
            progress_m=progress_m,
            beyond=beyond,
            completed=completed,
            diverged=False,
        )
        return StepResult(self._frame, terminated, truncated, info)

    def close(self) -> None:
        """Release the offscreen renderer. Safe to call more than once."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self._closed = True

    def __enter__(self) -> PracticeTrack:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals --------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("this PracticeTrack is closed; build a new one")

    def _project(self) -> Projection:
        position = self._data.xpos[self._body]
        return self.centerline.project(float(position[0]), float(position[1]))

    def _render(self) -> Frame:
        """One camera image, checked against the contract on the way out.

        Built on first use rather than in ``__init__`` so that a headless machine can still
        construct the env, export the model and run every non-rendering test.
        """
        if self._renderer is None:
            height, width, _ = self._frame_shape
            self._renderer = mujoco.Renderer(self._model, height, width)
        self._renderer.update_scene(self._data, camera=self._camera)
        # Copied so the caller owns the array: the renderer is free to reuse its buffer,
        # and a policy comparing this frame with the last one would silently compare a
        # frame with itself.
        frame = np.array(self._renderer.render(), dtype=np.uint8, copy=True)
        # No silent resize: a frame that quietly changed size would land on the hex
        # resampler's fixed 721-column lattice and corrupt the retinal geometry rather
        # than merely look wrong.
        return validate_frame(frame, self._frame_shape)

    def _info(
        self,
        *,
        projection: Projection | None,
        progress_m: float,
        beyond: float,
        completed: float | None,
        diverged: bool,
    ) -> dict[str, Any]:
        """Raw signals for whoever builds a reward out of them (GH-17), never a reward."""
        return {
            "steps": self._steps,
            "sim_time": float(self._data.time),
            "progress_m": float(progress_m),
            "speed_mps": 0.0 if diverged else self._dynamics.speed_mps(self._data),
            "gear": self._dynamics.gear + 1,
            "arclength_m": self._previous_s,
            "lateral_m": 0.0 if projection is None else projection.lateral,
            "off_track_fraction": float(beyond),
            "on_track": False if projection is None else projection.is_on_track,
            "lap_time": self._lap.current_lap_time(float(self._data.time)),
            "lap_fraction": self._lap.lap_fraction,
            "lap_completed": completed,
            "lap_count": self._lap.completed,
            "on_out_lap": not self._lap.timing,
            "diverged": diverged,
        }
