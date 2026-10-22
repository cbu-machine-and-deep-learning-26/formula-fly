"""The training env's link to AC: a lockstep server next to the game, a client in the trainer.

The bridge (:mod:`~fly_driver.bridge.car_side`) is for driving: AC pushes frames and never
waits. Training needs the opposite: the trainer asks for a reset or a step and blocks until
AC has an answer. :func:`serve_env` owns the car and answers; :class:`RemoteCar` is the car
as the trainer sees it, with the same methods as the car itself, so
:class:`~fly_driver.envs.assetto_corsa.AssettoCorsaTrack` drives either.

Real time. A ``STEP`` applies its control at once and blocks until the next AC tick. AC
does not wait, so a trainer that comes back late (a PPO update, an evaluation) gets the
newest tick and the ticks in between show as a gap in ``Telemetry.tick``; the env counts
them. Nothing queues and nothing is replayed.

Several clients. The trainer opens a second connection for each periodic evaluation while
its training connection sits idle. Requests from all connections are served one at a time
under a lock, and every observation carries the server's reset count. When another client
resets the car, the next answer to the first carries a different count, and
:meth:`RemoteCar.step` reports the episode as ended rather than carrying on in a car that
has moved.

The server side runs in AssettoCorsaGym's Python 3.9 env: 3.9 syntax at runtime, stdlib and
numpy only.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any, Protocol

from fly_driver.bridge.wire import (
    RESET,
    STEP,
    Telemetry,
    recv_hello,
    recv_observation,
    recv_request,
    send_hello,
    send_observation,
    send_request,
)
from fly_driver.interface import ControlVector, Frame

__all__ = ["RESET_ATTEMPTS", "AcCar", "RemoteCar", "checked_reset", "serve_env"]

logger = logging.getLogger(__name__)

#: Resets tried before the server gives up on a car that will not come back to the road.
RESET_ATTEMPTS = 3


class AcCar(Protocol):
    """A car the training env can drive: AssettoCorsaCar, FakeAc, or RemoteCar."""

    def reset(self) -> Frame:
        """Put the car back on the road at rest; return the first frame."""

    def apply(self, control: ControlVector) -> None:
        """Send a control now."""

    def step(self) -> tuple[Frame, bool]:
        """Block until the next tick; ``True`` if the episode ended outside the env."""

    def telemetry(self) -> Telemetry:
        """State at the last tick returned."""

    def close(self) -> None:
        """Release the car."""


def checked_reset(car: AcCar, attempts: int = RESET_ATTEMPTS) -> Frame:
    """Reset until the car is on the road with all four tyres in.

    Upstream's reset sends one message and sleeps a second; nothing confirms the car moved.
    This checks the telemetry it lands with and tries again.

    Raises:
        RuntimeError: If ``attempts`` resets leave the car off the road.
    """
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        frame = car.reset()
        telemetry = car.telemetry()
        logger.info(
            "reset %d took %.2f s: at %.0f m, %d tyres out, %.1f m/s",
            attempt,
            time.perf_counter() - started,
            telemetry.lap_dist_m,
            telemetry.tyres_out,
            telemetry.speed_mps,
        )
        if telemetry.centre_on_track and telemetry.tyres_out == 0:
            return frame
    raise RuntimeError(f"car still off the road after {attempts} resets: {telemetry}")


def serve_env(car: AcCar, listener: socket.socket, hello: dict[str, Any]) -> None:
    """Serve ``car`` to every client that connects, until ``listener`` is closed.

    Args:
        car: The car. Only this function touches it, under one lock.
        listener: Bound and listening, e.g. from
            :func:`~fly_driver.bridge.fly_side.open_listener`.
        hello: Sent to each client on connect: ``track``, ``car``, ``frame_shape``,
            ``frame_rate_hz``.
    """
    lock = threading.Lock()
    state = {"episode": 0}

    def handle(conn: socket.socket, address: Any) -> None:
        requests = 0
        with conn:
            try:
                send_hello(conn, hello)
                while True:
                    request = recv_request(conn)
                    if request is None:
                        break
                    kind, control = request
                    with lock:
                        if kind == RESET:
                            frame, ended = checked_reset(car), False
                            state["episode"] += 1
                        else:
                            car.apply(control)
                            frame, ended = car.step()
                        telemetry = car.telemetry()
                        episode = state["episode"]
                    send_observation(conn, episode, ended, telemetry, frame)
                    requests += 1
            except (ConnectionError, OSError):
                logger.warning("client %s dropped", address)
            except Exception:
                # A failed reset or step closes this client, so its env raises rather
                # than training on a car in an unknown state. The server stays up.
                logger.exception("car failed serving %s", address)
            finally:
                with lock:
                    car.apply(ControlVector.neutral())
        logger.info("client %s left after %d requests", address, requests)

    while True:
        try:
            conn, address = listener.accept()
        except OSError:
            return
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        logger.info("client connected from %s:%d", *address[:2])
        threading.Thread(target=handle, args=(conn, address), daemon=True).start()


class RemoteCar:
    """The car behind :func:`serve_env`, with the car's own methods.

    Args:
        host: The env server's address; ``127.0.0.1`` on the AC box itself.
        port: Its port.
        timeout_s: Longest wait for any answer. A reset takes about a second on AC.

    Attributes:
        hello: What the server announced: track, car, frame shape and rate.
    """

    def __init__(self, host: str, port: int, timeout_s: float = 60.0) -> None:
        self._sock = socket.create_connection((host, port), timeout=timeout_s)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.hello = recv_hello(self._sock)
        self._control = ControlVector.neutral()
        self._episode: int | None = None
        self._telemetry: Telemetry | None = None

    def reset(self) -> Frame:
        """Reset the car on the server."""
        send_request(self._sock, RESET, ControlVector.neutral())
        self._episode, _, self._telemetry, frame = recv_observation(self._sock)
        return frame

    def apply(self, control: ControlVector) -> None:
        """Hold ``control`` for the next :meth:`step`; it is sent with it."""
        self._control = control

    def step(self) -> tuple[Frame, bool]:
        """Send the control and wait for the next tick.

        Returns:
            ``(frame, ended)``. ``ended`` is true when AC ended the episode, or when another
            client reset the car since this client's last reset.
        """
        send_request(self._sock, STEP, self._control)
        episode, ended, self._telemetry, frame = recv_observation(self._sock)
        return frame, ended or episode != self._episode

    def telemetry(self) -> Telemetry:
        """State at the last tick returned."""
        if self._telemetry is None:
            raise RuntimeError("call reset() first")
        return self._telemetry

    def close(self) -> None:
        """Hang up; the server releases the controls."""
        self._sock.close()
