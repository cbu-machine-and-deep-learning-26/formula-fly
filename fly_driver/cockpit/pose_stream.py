"""Stream the cockpit fly's pose to Assetto Corsa over UDP, one datagram per frame (GH-26).

The Custom Shaders Patch app in ``scripts/cockpit/fly_pose/`` receives it and turns the fly
driver's bones. Wrap whatever already drives the car, without editing it:

- :class:`PoseStreamDriver` wraps any :class:`~fly_driver.interface.Driver`. With a
  :class:`~fly_driver.drivers.DirectDriveAgent` that has a brain, the brain is read too.
- :class:`PoseStreamEnv` wraps any env with :class:`~fly_driver.envs.practice_track.PracticeTrack`'s
  ``reset``/``step``, for training loops that call the policy themselves. Pass the brain.

A datagram is ASCII, ``FLY1 <seq> <8 angles in degrees>``, space-separated, in
:class:`~fly_driver.cockpit.live_pose.FlyPose` field order. Text, because Lua parses it with
``gmatch`` and ``tonumber`` and needs no FFI. UDP, because a late pose is worth nothing: the
game takes the newest one and a lost one costs a frame. Stdlib only.
"""

from __future__ import annotations

import contextlib
import socket
from dataclasses import fields
from typing import Any

from fly_driver.cockpit.live_pose import FlyPose, fly_pose, smooth_pose
from fly_driver.interface import ControlVector, Driver, Frame

__all__ = [
    "DEFAULT_PORT",
    "MAGIC",
    "PoseStream",
    "PoseStreamDriver",
    "PoseStreamEnv",
    "decode_pose",
    "encode_pose",
]

#: Not 9996, which Assetto Corsa's own UDP telemetry uses, nor the bridge's 2350.
DEFAULT_PORT = 9871
MAGIC = "FLY1"
_FIELD_COUNT = len(fields(FlyPose))


def encode_pose(seq: int, pose: FlyPose) -> bytes:
    """One datagram. Angles to a thousandth of a degree."""
    return " ".join([MAGIC, str(seq), *(f"{value:.3f}" for value in pose.values())]).encode()


def decode_pose(datagram: bytes) -> tuple[int, FlyPose]:
    """Parse a datagram from :func:`encode_pose`.

    Raises:
        ValueError: If it is not a pose datagram.
    """
    tokens = datagram.decode("ascii").split()
    if len(tokens) != 2 + _FIELD_COUNT or tokens[0] != MAGIC:
        raise ValueError(f"not a {MAGIC} pose datagram: {datagram[:40]!r}")
    return int(tokens[1]), FlyPose(*(float(token) for token in tokens[2:]))


class PoseStream:
    """Maps, smooths and sends one pose per call to :meth:`publish`.

    Args:
        host: Where Assetto Corsa runs. The game box itself, or its address when the fly
            runs elsewhere on the network.
        port: The app's listening port; :data:`DEFAULT_PORT` unless changed in both places.
        alpha: Smoothing, see :func:`~fly_driver.cockpit.live_pose.smooth_pose`.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = DEFAULT_PORT, alpha: float = 0.5):
        self.address = (host, port)
        self.alpha = alpha
        self.seq = 0
        self.pose: FlyPose | None = None
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def reset(self) -> None:
        """Forget the last pose, so a new episode starts unsmoothed."""
        self.pose = None

    def publish(self, control: ControlVector, brain: Any = None) -> FlyPose:
        """Send the pose for ``control`` and ``brain``'s current activity, and return it.

        Args:
            control: What the car was just sent.
            brain: Anything with ``rates()`` and ``populations``, as
                :class:`~fly_driver.brains.central_complex.CentralComplexBrain` has, or ``None``.
        """
        if brain is not None and hasattr(brain, "rates"):
            target = fly_pose(control, brain.rates(), brain.populations)
        else:
            target = fly_pose(control)
        self.pose = smooth_pose(self.pose, target, self.alpha)
        # Nobody listening, or the network blipped: never stop the drive for a pose.
        with contextlib.suppress(OSError):
            self._socket.sendto(encode_pose(self.seq, self.pose), self.address)
        self.seq += 1
        return self.pose

    def close(self) -> None:
        """Close the socket."""
        self._socket.close()


class PoseStreamDriver:
    """A :class:`~fly_driver.interface.Driver` that also streams the fly's pose each frame.

    Args:
        driver: The driver to wrap. Its ``brain`` attribute, if any, is read after each act.
        stream: Where the pose goes.
    """

    def __init__(self, driver: Driver, stream: PoseStream) -> None:
        self.driver = driver
        self.stream = stream

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """The wrapped driver's."""
        return self.driver.frame_shape

    def reset(self, seed: int | None = None) -> None:
        """Reset the driver and the smoothing."""
        self.driver.reset(seed=seed)
        self.stream.reset()

    def act(self, frame: Frame) -> ControlVector:
        """The wrapped driver's control, after streaming the pose it implies."""
        control = self.driver.act(frame)
        self.stream.publish(control, getattr(self.driver, "brain", None))
        return control


class PoseStreamEnv:
    """An env that streams the fly's pose on every :meth:`step`.

    Everything but ``reset`` and ``step`` passes through to the wrapped env.

    Args:
        env: Anything with ``reset`` and ``step(action)``, such as ``PracticeTrack``.
        stream: Where the pose goes.
        brain: Read after each step for head, wings and antennae. ``None`` poses the legs only.
    """

    def __init__(self, env: Any, stream: PoseStream, brain: Any = None) -> None:
        self.env = env
        self.stream = stream
        self.brain = brain

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Reset the env and the smoothing."""
        self.stream.reset()
        return self.env.reset(*args, **kwargs)

    def step(self, action: object) -> Any:
        """Step the env, then stream the pose for the action it accepted."""
        result = self.env.step(action)
        self.stream.publish(ControlVector.from_any(action), self.brain)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
