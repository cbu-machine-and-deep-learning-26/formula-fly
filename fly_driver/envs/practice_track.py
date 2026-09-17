"""The practice-track environment (GH-16): a camera frame in, a control vector out.

This is the file the rest of the project plugs into. `AGENTS.md` §3 fixes one interface for
the whole pipeline -- observation in, :class:`~fly_driver.interface.ControlVector` out -- and
everything else in ``fly_driver.envs`` is a piece of the world rather than a way to drive
through it. The track geometry, the car, the head camera, the lap timer and the track-limits
rule all already existed and are all tested; what was missing was the loop that renders a
frame, accepts an action, and steps the two in lockstep.

The surface is Gymnasium's -- ``reset(seed=...) -> (frame, info)``,
``step(action) -> (frame, reward, terminated, truncated, info)``, ``render()`` -- because that
is what :mod:`fly_driver.training.evaluation` requires of an env, and what
:class:`~fly_driver.envs.dummy_track.DummyTrackEnv` already speaks. The two envs are
interchangeable to the evaluation harness, which is the whole point of having a dummy. It is
not a ``gymnasium.Env`` subclass: gymnasium is not a dependency, and adding one to the default
install is the x86-only-wheel risk `AGENTS.md` §7 says to find in week 1 rather than week 5.

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

    frame, info = env.reset(seed=0)
    eye.reset()                                   # REQUIRED at every episode start:
                                                  # the optic lobe keeps state between
                                                  # frames, so a new episode must begin
                                                  # from the grey warm-up, not from
                                                  # whatever the last crash looked like.
    while True:
        features = eye.encode(frame)              # (5768,) float32
        action = policy.act(features)             # ControlVector or a (3,) array
        frame, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

Take the shape and the rate **from the env** rather than typing the numbers again. Both are
constructor arguments on :class:`~fly_driver.eyes.FlyvisEye`, and the failure mode if they
drift apart is silent: the eye integrates at the wrong ``dt`` and produces plausible
nonsense. :data:`~fly_driver.interface.FRAME_RATE_HZ` explains which of the two numbers is
negotiable.

Reward
------

:class:`ProgressReward` is the default and carries the same terms and constants the dummy
track uses, so a score on one is comparable with a score on the other. It is an argument, not
a fixture: GH-17 owns reward shaping and passes its own. Every term is published separately
in ``info["reward_terms"]``, which is what GH-17's "writes per-term rewards to CSV" needs, and
what makes it possible to see an agent farming one term without progressing (`AGENTS.md` §11).

What the fly sees
-----------------

The painted racing line is **on** by default and is therefore in shot. That is Payton's
call for hand driving and it is recorded here so it is not quietly reversed: a painted line
is a strong, unambiguous cue, and a policy that merely follows it is answering an easier
question than "does connectome wiring help". Pass ``SceneConfig(racing_line=False)`` for
the RQ1 comparisons, or leave it on deliberately and say so in the write-up.

A note on reproducibility
-------------------------

The *physics* is bit-identical for a given seed and action sequence -- ``qpos`` matches
exactly, run after run. The *rendering* is not quite: MuJoCo hands rasterisation to the GPU,
and on this machine two identical runs occasionally disagree by one level out of 255 on a
handful of subpixels (measured: under 3% of frames, at most 7 of 27,648 subpixels, never
more than 1/255). That is far below anything the hex resampler's 13-pixel mean filter can
see, but it does mean **frames must not be hashed or compared for exact equality** in a
determinism check. Compare the physics, or compare frames with a tolerance.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

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

__all__ = [
    "DEFAULT_CAMERA",
    "DEFAULT_TRACK_LIMIT",
    "PracticeTrack",
    "ProgressReward",
    "RewardFunction",
]

#: The camera mounted where the fly's head sits, from :func:`~fly_driver.envs.car.car_body_xml`.
DEFAULT_CAMERA = "fly_head"

#: Fraction of the car's width that may hang past the kerb before the episode ends. The
#: same 0.15 ``scripts/drive.py`` throws a hand-driven lap away at, so the fly is judged
#: under the rule the humans drive under.
DEFAULT_TRACK_LIMIT = 0.15


class RewardFunction(Protocol):
    """Turns one step's raw signals into a scalar and its parts.

    The parts are not decoration: `AGENTS.md` §11 asks for per-term reward logging precisely
    so that an agent farming one term without progressing is visible rather than inferred.
    """

    def __call__(self, info: Mapping[str, Any], dt: float) -> tuple[float, dict[str, float]]:
        """Return ``(reward, terms)``; ``sum(terms.values())`` must equal ``reward``."""


@dataclass(frozen=True)
class ProgressReward:
    """Distance covered, less a penalty for running wide, plus a bonus for a full lap.

    The constants are :class:`~fly_driver.envs.dummy_track.DummyTrackEnv`'s, deliberately:
    the dummy exists so the evaluation harness can be exercised without MuJoCo, and a
    stand-in whose returns are on a different scale from the real thing is a poor stand-in.

    Args:
        lateral_penalty: Cost per metre off the centreline per second. Small, because the
            racing line is not the centreline and a fast lap is *supposed* to run wide.
        lap_bonus: One-off reward for completing a lap.
        off_track_penalty: One-off cost for leaving the circuit. Negative.
    """

    lateral_penalty: float = 0.1
    lap_bonus: float = 100.0
    off_track_penalty: float = -10.0

    def __call__(self, info: Mapping[str, Any], dt: float) -> tuple[float, dict[str, float]]:
        terms = {
            "progress": float(info["progress_m"]),
            "lateral": -self.lateral_penalty * abs(float(info["lateral_m"])) * dt,
            "lap_bonus": 0.0,
            "off_track": 0.0,
        }
        # elif, not a second if: a lap that completes is not also punished for the wheel
        # that was over the kerb as it crossed.
        if info["lap_complete"]:
            terms["lap_bonus"] = self.lap_bonus
        elif info["off_track"]:
            terms["off_track"] = self.off_track_penalty
        return float(sum(terms.values())), terms


class PracticeTrack:
    """Silverstone, one car, one head camera, stepped at the optic lobe's frame rate.

    Args:
        car: Vehicle parameters. Defaults to the SF70H :class:`~fly_driver.envs.car.CarConfig`.
        scene: Track and rendering settings. Defaults to
            :class:`~fly_driver.envs.scene.SceneConfig`, which starts the car
            ``grid_offset_m`` behind the line so the first lap is timed from a real
            crossing rather than from wherever the simulator booted.
        seed: Default seed, overridable per episode by :meth:`reset`. Nothing in the env is
            stochastic yet; the generator exists so that when episode randomisation arrives
            it has one place to come from, and so that the seed-determinism check
            `AGENTS.md` §11 asks for can be written now instead of retrofitted.
        max_steps: Truncate the episode after this many steps. ``None`` runs forever, which
            is what a hand-driven session or a lap-time measurement wants.
        frame_shape: ``(height, width, 3)``. Must fit inside the scene's offscreen buffer.
        frame_rate_hz: Environment steps per second of simulated time. One step advances
            the physics by exactly ``1 / frame_rate_hz`` and produces exactly one frame.
        track_limit: Fraction of the car's width past the kerb that ends the episode.
            ``0`` turns the rule off and lets the car drive across the infield.
        camera: Name of the MJCF camera to render from.
        reward: Scores each step. Defaults to :class:`ProgressReward`.

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
        reward: RewardFunction | None = None,
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
        self.reward = reward or ProgressReward()
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
        self._done = True
        self._steps = 0
        self._previous_s = 0.0
        self._frame: Frame | None = None

    # -- read-only view of the world -------------------------------------------------

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """``(height, width, 3)``. Hand this to the eye rather than a literal."""
        return self._frame_shape

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """Shape of the uint8 RGB frame from :meth:`reset` and :meth:`step`."""
        return self._frame_shape

    @property
    def action_shape(self) -> tuple[int]:
        """Shape of the ``(steer, throttle, brake)`` action."""
        return (3,)

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

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Frame, dict[str, Any]]:
        """Put the car back on the grid and return the first frame and info.

        Whatever drives this env must reset at the same moment -- in particular a
        :class:`~fly_driver.eyes.FlyvisEye` keeps its network state between frames and needs
        its own ``reset()`` here, or the new episode starts with the old one's afterimage.

        Args:
            seed: Re-seeds :attr:`rng` for this episode. ``None`` keeps the current one.
            options: Unused; present because the harness's protocol allows it.
        """
        del options  # accepted for Gymnasium compatibility, nothing to configure yet
        self._require_open()
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)

        # qpos0 holds the grid pose baked into the MJCF by assemble_model_xml.
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        self._dynamics.reset()
        self._steps = 0
        self._done = False

        projection = self._project()
        self._previous_s = projection.arclength
        self._lap.reset(projection.arclength, float(self._data.time))

        self._frame = self._render()
        info = self._info(
            projection=projection,
            progress_m=0.0,
            beyond=0.0,
            lap_complete=False,
            diverged=False,
            terms={},
        )
        return self._frame, info

    def step(self, action: object) -> tuple[Frame, float, bool, bool, dict[str, Any]]:
        """Advance one frame.

        Args:
            action: ``(steer, throttle, brake)``, as a
                :class:`~fly_driver.interface.ControlVector`, a length-3 array-like, or
                anything with a ``to_array()``. Out-of-range values raise; nothing is
                clipped silently, because a saturated action that looks deliberate is the
                kind of plausible-wrong `AGENTS.md` §11 is about. Use
                :meth:`~fly_driver.interface.ControlVector.clipped` to squash on purpose.

        Returns:
            ``(frame, reward, terminated, truncated, info)``, in Gymnasium order.

        Raises:
            TypeError: If ``action`` is not a real number triple.
            ValueError: If a component is non-finite or out of range.
            RuntimeError: Before the first :meth:`reset`, after the episode has ended, or
                after :meth:`close`.
        """
        self._require_open()
        if self._done or self._frame is None:
            raise RuntimeError("call reset() before step(); the episode has ended")

        control = _as_control(action)
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
                lap_complete=False,
                diverged=True,
                terms={},
            )
            self._done = True
            return self._frame, 0.0, True, False, info

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

        info = self._info(
            projection=projection,
            progress_m=progress_m,
            beyond=beyond,
            lap_complete=completed is not None,
            diverged=False,
            terms={},
            lap_time=completed,
        )
        reward, terms = self.reward(info, self.dt)
        info["reward_terms"] = terms

        self._frame = self._render()
        self._done = terminated or truncated
        return self._frame, float(reward), terminated, truncated, info

    def render(self) -> Frame:
        """The current frame, identical to the last observation.

        A copy, so a video writer holding onto it is unaffected by the next step.
        """
        self._require_open()
        if self._frame is None:
            raise RuntimeError("call reset() before render()")
        return self._frame.copy()

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
        lap_complete: bool,
        diverged: bool,
        terms: dict[str, float],
        lap_time: float | None = None,
    ) -> dict[str, Any]:
        """Raw signals, plus the two keys the evaluation harness reads by name.

        ``lap_complete`` and ``lap_time`` carry the harness's meaning, not the intuitive
        one: ``lap_time`` is the **completed** lap's time and is ``None`` on every other
        step. The harness reads it only when ``lap_complete`` is true, and silently
        substitutes ``steps / frame_rate_hz`` when it is ``None``, so putting a running
        clock here would fabricate lap times that look entirely reasonable. The running
        clock is ``lap_elapsed_s``.
        """
        return {
            # the evaluation harness's contract
            "lap_complete": lap_complete,
            "lap_time": lap_time,
            "off_track": beyond > 0.0,
            "reward_terms": terms,
            # the terms a reward function is built from
            "progress_m": float(progress_m),
            "speed_mps": 0.0 if diverged else self._dynamics.speed_mps(self._data),
            "lateral_m": 0.0 if projection is None else projection.lateral,
            "off_track_fraction": float(beyond),
            # telemetry
            "steps": self._steps,
            "sim_time": float(self._data.time),
            "gear": self._dynamics.gear + 1,
            "arclength_m": self._previous_s,
            "on_track": False if projection is None else projection.is_on_track,
            "lap_elapsed_s": self._lap.current_lap_time(float(self._data.time)),
            "lap_fraction": self._lap.lap_fraction,
            "lap_count": self._lap.completed,
            "on_out_lap": not self._lap.timing,
            "diverged": diverged,
        }


def _as_control(action: object) -> ControlVector:
    """Accept what the harness sends, what a policy returns, or the typed thing itself.

    The evaluation harness converts every agent's output to a ``(3,)`` float32 array before
    it reaches an env, so refusing arrays here would make this env unevaluatable. Validation
    is :class:`~fly_driver.interface.ControlVector`'s own, which rejects the wrong shape,
    non-finite values and out-of-range components without clipping any of them.
    """
    if isinstance(action, ControlVector):
        return action
    raw = action.to_array() if hasattr(action, "to_array") else action
    return ControlVector.from_array(raw)
