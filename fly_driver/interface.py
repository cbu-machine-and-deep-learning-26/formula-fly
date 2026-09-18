"""The shared contract: a camera frame in, a control vector out.

`AGENTS.md` §3 fixes this so every stage can be built and swapped independently — the
practice track (GH-16), the Assetto Corsa bridge (GH-24), and the tethered fly body
(GH-21) all speak it, so a policy trained against one runs against another unchanged.

Scoped deliberately to what an env needs: the action type and frame validation. The
``Eye`` and ``Policy`` protocols belong with the tickets that introduce them (GH-13/14/15)
rather than being invented here ahead of a real implementation.

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

import numpy as np
import numpy.typing as npt

__all__ = [
    "CONTROL_DTYPE",
    "FRAME_DTYPE",
    "FRAME_RATE_HZ",
    "FRAME_SHAPE",
    "ControlVector",
    "Frame",
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

Frame = npt.NDArray[np.uint8]

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
