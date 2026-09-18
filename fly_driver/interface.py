"""The shared contract: a camera frame in, a control vector out.

`AGENTS.md` §3 fixes this so every stage can be built and swapped independently — the
practice track (GH-16), the Assetto Corsa bridge (GH-24), and the tethered fly body
(GH-21) all speak it, so a policy trained against one runs against another unchanged.

Two layers. The *types* -- :class:`ControlVector`, :data:`Frame`, :data:`Features` and
their validators -- are what an env needs. The *stages* -- :class:`Eye`, :class:`Policy`,
:class:`Body` and :class:`Driver` -- are the seams between the tracks, as
:class:`typing.Protocol` so a stage is whatever has the right methods, with no base class to
inherit and no torch to import. :mod:`fly_driver.drivers` composes them.

numpy-only, so importing this never drags in mujoco, torch, or flyvis.

Two rules enforced in code rather than documented, because `AGENTS.md` §11 lists both as
failures that look plausible when wrong:

- **No silent clipping.** :class:`ControlVector` rejects out-of-range values. Squashing a
  raw network output is spelled :meth:`ControlVector.clipped`.
- **No silent resizing.** :func:`validate_frame` raises on an unexpected shape instead of
  quietly interpolating.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = [
    "CONTROL_DTYPE",
    "FEATURE_DTYPE",
    "FRAME_DTYPE",
    "FRAME_RATE_HZ",
    "FRAME_SHAPE",
    "Body",
    "ControlVector",
    "Driver",
    "Eye",
    "Features",
    "Frame",
    "Policy",
    "validate_features",
    "validate_frame",
]

#: Frames are ``(height, width, 3)`` uint8 in ``[0, 255]`` -- what MuJoCo's offscreen
#: renderer returns and what an Assetto Corsa screen capture gives us. The dtype and the
#: channel count are the hard part of the contract; the resolution is
#: :data:`FRAME_SHAPE`, which is the project's agreed default and still an argument on
#: whichever env is in the loop.
FRAME_DTYPE = np.uint8

#: Actions are float32 -- the dtype a policy head emits, so no conversion at the seam.
CONTROL_DTYPE = np.float32

#: Features are float32 too -- what the flyvis readout emits and what a policy head
#: consumes -- and one flat ``(feature_dim,)`` vector, whatever the eye's internal geometry.
FEATURE_DTYPE = np.float32

Frame = npt.NDArray[np.uint8]
Features = npt.NDArray[np.float32]

#: The camera frame every stage of the pipeline agrees on, ``(height, width, 3)``.
#:
#: **This is a choice, not a constraint.** It reaches us from an earlier draft that used
#: Gymnasium CarRacing, whose FPS and frame size the eye was first written against; the
#: practice track replaced that design (`AGENTS.md` §6) but the number was worth keeping --
#: it is what the hex resampler's geometry was verified at, and 96x96 renders fast enough
#: to leave the eye as the bottleneck rather than the renderer. Both
#: :class:`~fly_driver.eyes.HexResampler` and :class:`~fly_driver.eyes.FlyvisEye` take it as
#: a constructor argument, so changing it here is legitimate as long as the eye is
#: constructed from :attr:`~fly_driver.envs.practice_track.PracticeTrack.frame_shape` rather
#: than from a number typed out again at the call site.
FRAME_SHAPE: tuple[int, int, int] = (96, 96, 3)

#: Frames per second of simulated time: one frame per environment step.
#:
#: **This one is not negotiable.** flyvis is a dynamical system integrated with Euler steps
#: of ``dt`` seconds and one frame is one step, so the environment's frame rate *is* the
#: optic lobe's integration rate. ``FlyvisEye`` raises below 50 Hz -- ``dt > 1/50`` is
#: outside the integration limit the pretrained network was fitted under -- and warns above
#: it. Anything that steps the car at some other rate and feeds the eye is quietly
#: simulating a different fly.
FRAME_RATE_HZ: float = 50.0

#: ``(name, low, high)`` per control component, in :meth:`ControlVector.to_array` order.
_BOUNDS: tuple[tuple[str, float, float], ...] = (
    ("steer", -1.0, 1.0),
    ("throttle", 0.0, 1.0),
    ("brake", 0.0, 1.0),
)


@dataclass(frozen=True)
class ControlVector:
    """A car control action. These three numbers are the whole shared interface.

    Normalised rather than physical: an env maps ``steer`` onto its own steering limit and
    ``throttle`` onto its own torque. That is what lets the same policy drive the MuJoCo
    practice track and an Assetto Corsa Formula car without rescaling.

    Args:
        steer: -1.0 (full left) to 1.0 (full right).
        throttle: 0.0 to 1.0.
        brake: 0.0 to 1.0.

    Raises:
        TypeError: If a component is not a real number.
        ValueError: If a component is non-finite or outside its range.
    """

    steer: float
    throttle: float
    brake: float

    def __post_init__(self) -> None:
        for name, low, high in _BOUNDS:
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, (int, float, np.floating, np.integer)):
                raise TypeError(f"{name} must be a real number, got {type(raw).__name__}")
            value = float(raw)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            if not low <= value <= high:
                raise ValueError(
                    f"{name}={value!r} is outside [{low}, {high}]. "
                    "Use ControlVector.clipped(...) if squashing is intended."
                )
            # Normalise ints and numpy scalars to plain floats.
            object.__setattr__(self, name, value)

    @classmethod
    def clipped(cls, steer: float, throttle: float, brake: float) -> ControlVector:
        """Build a ``ControlVector``, clipping each component into range.

        The explicit path for raw policy outputs. Non-finite values are still rejected:
        clipping a NaN would silently produce a bound, which is the plausible-looking wrong
        value `AGENTS.md` §11 warns about.
        """
        given = {"steer": steer, "throttle": throttle, "brake": brake}
        values = {}
        for name, low, high in _BOUNDS:
            value = float(given[name])
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
            values[name] = min(max(value, low), high)
        return cls(**values)

    @classmethod
    def neutral(cls) -> ControlVector:
        """Straight ahead, coasting: no throttle, no brake."""
        return cls(steer=0.0, throttle=0.0, brake=0.0)

    @classmethod
    def from_array(cls, array: npt.ArrayLike, *, clip: bool = False) -> ControlVector:
        """Build from a length-3 ``(steer, throttle, brake)`` array."""
        values = np.asarray(array, dtype=np.float64)
        if values.shape != (3,):
            raise ValueError(
                f"expected 3 components (steer, throttle, brake), got shape {values.shape}"
            )
        factory = cls.clipped if clip else cls
        return factory(float(values[0]), float(values[1]), float(values[2]))

    @classmethod
    def from_any(cls, action: object, *, clip: bool = False) -> ControlVector:
        """Accept a ``ControlVector``, a length-3 array-like, or anything with ``to_array()``.

        What an env or a driver sees at its seam: the typed thing itself, the raw ``(3,)``
        array a policy head emits, or the evaluation harness's converted action. Validation
        is the constructor's own -- wrong shape, non-finite and out-of-range values all
        raise -- and nothing is clipped unless ``clip`` says so.
        """
        if isinstance(action, cls):
            return action
        raw = action.to_array() if hasattr(action, "to_array") else action
        return cls.from_array(raw, clip=clip)

    def to_array(self) -> npt.NDArray[np.float32]:
        """As ``(3,)`` float32, ordered ``(steer, throttle, brake)``."""
        return np.array([self.steer, self.throttle, self.brake], dtype=CONTROL_DTYPE)


def validate_frame(frame: object, expected_shape: tuple[int, int, int]) -> Frame:
    """Check a frame against the contract and return it unchanged.

    Never resizes, never casts, never rescales. A mismatch means a bug upstream, and
    silently fixing it here would hide a misconfigured camera or a broken env wrapper
    (`AGENTS.md` §11). It matters more than usual for this project: the hex resampler
    (GH-13) maps pixels onto a fixed 721-column lattice, so a frame that quietly changed
    size would corrupt the retinal geometry rather than merely look wrong.

    Args:
        frame: The object to check.
        expected_shape: The ``(height, width, channels)`` the consumer declared.

    Raises:
        TypeError: If ``frame`` is not a uint8 ndarray.
        ValueError: If its shape is not ``expected_shape``.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be a numpy array, got {type(frame).__name__}")
    if frame.dtype != FRAME_DTYPE:
        raise TypeError(
            f"frame must be {np.dtype(FRAME_DTYPE).name} in [0, 255], got {frame.dtype}. "
            "No implicit cast: check the renderer or capture path."
        )
    if frame.shape != expected_shape:
        raise ValueError(
            f"frame shape {frame.shape} != expected {expected_shape}. "
            "No implicit resize: configure the camera deliberately."
        )
    return frame


def validate_features(features: object, expected_dim: int) -> Features:
    """Check an eye's output against the contract and return it unchanged.

    The same rule as :func:`validate_frame`, one seam downstream: never reshaped, never
    cast, never patched. A policy that quietly accepted a ``(721, 8)`` array where it
    expected ``(5768,)`` would learn from scrambled features and never say so.

    Args:
        features: The object to check.
        expected_dim: The feature count the policy was built for.

    Raises:
        TypeError: If ``features`` is not a float32 ndarray.
        ValueError: If its shape is not ``(expected_dim,)`` or it contains NaN or inf.
    """
    if not isinstance(features, np.ndarray):
        raise TypeError(f"features must be a numpy array, got {type(features).__name__}")
    if features.dtype != FEATURE_DTYPE:
        raise TypeError(
            f"features must be {np.dtype(FEATURE_DTYPE).name}, got {features.dtype}. "
            "No implicit cast: have the eye emit float32."
        )
    expected = (int(expected_dim),)
    if features.shape != expected:
        raise ValueError(
            f"features shape {features.shape} != expected {expected}. "
            "No implicit reshape: the eye and the policy disagree on feature_dim."
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("features must be finite; the eye produced NaN or inf")
    return features


@runtime_checkable
class Eye(Protocol):
    """Stage 1: a camera frame in, a feature vector out.

    ``frame_shape`` is the frame it accepts and ``feature_dim`` the length of what it
    emits; a driver reads both from the eye rather than having them typed again. ``reset``
    starts an episode. For an optic lobe that keeps state between frames
    (:class:`~fly_driver.eyes.FlyvisEye`) it is the grey warm-up, and it is required.
    """

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """``(height, width, 3)`` of the frames :meth:`encode` accepts."""

    @property
    def feature_dim(self) -> int:
        """Length of the vector :meth:`encode` returns."""

    def reset(self) -> None:
        """Start an episode from the resting state."""

    def encode(self, frame: Frame) -> Features:
        """Advance one frame and return ``(feature_dim,)`` float32 features."""


@runtime_checkable
class Policy(Protocol):
    """Stage 2: features in, control out -- the brain, or the readout on top of it."""

    @property
    def feature_dim(self) -> int:
        """Length of the vector :meth:`act` expects."""

    def reset(self, seed: int | None = None) -> None:
        """Start an episode; ``seed`` fixes any sampling the policy does."""

    def act(self, features: Features) -> ControlVector:
        """Choose a control. A raw ``(3,)`` array is tolerated at the seam and range-checked
        there; returning a :class:`ControlVector` makes the check the policy's own."""


@runtime_checkable
class Body(Protocol):
    """Stage 3: the tethered fly. The brain's intended control in, the cockpit's reading out.

    Direct drive skips this stage. The fly body (GH-21) implements it: wings and legs move
    in response to ``intent``, and ``actuate`` returns what the cockpit measures afterwards.
    """

    def reset(self) -> None:
        """Start an episode with the body at rest."""

    def actuate(self, intent: ControlVector) -> ControlVector:
        """Move, and return the control the cockpit actually reads."""


@runtime_checkable
class Driver(Protocol):
    """The whole rig as an env sees it: a frame in, a control out.

    This is the contract `AGENTS.md` §3 fixes for every track. Direct drive and the
    embodied fly are both drivers (:mod:`fly_driver.drivers`), so an env or the evaluation
    harness cannot tell them apart -- which is what makes the comparison between them fair.
    """

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """``(height, width, 3)`` of the frames :meth:`act` accepts."""

    def reset(self, seed: int | None = None) -> None:
        """Start an episode: every stage back to its resting state."""

    def act(self, frame: Frame) -> ControlVector:
        """One frame in, one control out."""
