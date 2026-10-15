"""The fly end of the bridge: serve any :class:`~fly_driver.interface.Driver` over TCP.

The fly side listens and the car side connects, so the slow part (loading the eye and the
policy) is ready before AC starts its real-time loop. Every frame is validated against the
driver's ``frame_shape`` with :func:`~fly_driver.interface.validate_frame`, never resized
(`AGENTS.md` §11), and every control is clipped before it goes back on the wire.

Usage::

    listener = open_listener("0.0.0.0", DEFAULT_PORT)
    serve(driver, listener)  # one car connection, until the car hangs up
"""

from __future__ import annotations

import logging
import socket
import time
from pathlib import Path
from types import SimpleNamespace

from fly_driver.bridge.wire import recv_frame, send_control
from fly_driver.interface import (
    FRAME_RATE_HZ,
    FRAME_SHAPE,
    ControlVector,
    Driver,
    Frame,
    validate_frame,
)

__all__ = ["ConstantDriver", "load_trained_driver", "open_listener", "serve"]

logger = logging.getLogger(__name__)


def open_listener(host: str, port: int) -> socket.socket:
    """Bind and listen. ``port=0`` picks a free port; read it from ``getsockname()``."""
    listener = socket.create_server((host, port))
    listener.listen(1)
    return listener


def serve(driver: Driver, listener: socket.socket) -> int:
    """Accept one car connection and drive it until the car closes it.

    The driver is reset on every frame the car marks as an episode start.

    Args:
        driver: The rig. Its ``frame_shape`` is the only frame shape accepted.
        listener: From :func:`open_listener`.

    Returns:
        The number of frames answered.

    Raises:
        TypeError, ValueError: If a frame breaks the contract. The connection closes and
            the car side stops rather than driving on a misconfigured camera.
    """
    conn, address = listener.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    logger.info("car connected from %s:%d", *address[:2])
    frames = 0
    with conn:
        while (message := recv_frame(conn)) is not None:
            seq, is_reset, frame = message
            started = time.perf_counter()
            if is_reset:
                driver.reset()
            validate_frame(frame, tuple(driver.frame_shape))
            control = ControlVector.from_any(driver.act(frame), clip=True)
            send_control(conn, seq, control, (time.perf_counter() - started) * 1000.0)
            frames += 1
    logger.info("car disconnected after %d frames", frames)
    return frames


def load_trained_driver(checkpoint: Path, device: str = "cpu") -> Driver:
    """Rebuild a frozen-eye :class:`~fly_driver.drivers.DirectDriveAgent` from a training run.

    The eye is rebuilt from the config the trainer stored in the checkpoint, sized to
    :data:`~fly_driver.interface.FRAME_SHAPE` at :data:`~fly_driver.interface.FRAME_RATE_HZ`.
    Needs torch, and flyvis for the flyvis eye.

    Raises:
        ValueError: If the run fine-tuned its eye; the bridge only loads frozen eyes.
    """
    from fly_driver.drivers import DirectDriveAgent
    from fly_driver.policies.mlp_policy import MlpPolicy
    from fly_driver.training.config import TrainConfig
    from fly_driver.training.train import build_eye

    policy = MlpPolicy.load(checkpoint, device=device)
    eye_config = TrainConfig.from_dict(policy.extra["config"]).eye
    if not eye_config.frozen:
        raise ValueError(f"{checkpoint} fine-tuned its eye; the bridge loads frozen eyes only")
    camera = SimpleNamespace(frame_shape=FRAME_SHAPE, frame_rate_hz=FRAME_RATE_HZ)
    return DirectDriveAgent(build_eye(eye_config, camera), policy)


class ConstantDriver:
    """Holds one control regardless of the frame. Proves the bridge without a trained policy.

    Args:
        control: The control to send every frame.
        frame_shape: Frames it accepts; anything else is rejected like any driver would.
    """

    def __init__(self, control: ControlVector, frame_shape: tuple[int, int, int]) -> None:
        self.control = control
        self.frame_shape = frame_shape

    def reset(self, seed: int | None = None) -> None:
        """Nothing to reset."""

    def act(self, frame: Frame) -> ControlVector:
        """Return the fixed control."""
        return self.control
