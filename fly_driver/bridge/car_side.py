"""The car end of the bridge: the real-time loop that runs next to Assetto Corsa.

AC runs in real time and does not wait for anyone. Each tick the car side sends the newest
frame, waits at most ``deadline_s`` for the fly's answer, applies whatever control it has,
and steps. A late answer never blocks AC: the tick repeats the last control and counts a
miss. At most one frame is in flight, so a slow fly falls behind by skipped frames rather
than by a growing queue of stale ones; the late answer is applied on the tick it arrives.

Runs in AssettoCorsaGym's Python 3.9 environment, so this module keeps to 3.9 syntax at
runtime and imports nothing beyond numpy and the stdlib.
"""

from __future__ import annotations

import csv
import logging
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from fly_driver.bridge.wire import CONTROL, decode_control, send_frame
from fly_driver.interface import ControlVector, Frame, validate_frame

__all__ = ["CarEnv", "FlyLink", "LinkStats", "run_episode"]

logger = logging.getLogger(__name__)


class CarEnv(Protocol):
    """What the loop needs from a car simulator. :mod:`fly_driver.bridge.assetto` wraps AC."""

    def reset(self) -> Frame:
        """Start an episode and return the first frame."""

    def apply(self, control: ControlVector) -> None:
        """Send a control to the car now, without waiting for the next tick."""

    def step(self) -> tuple[Frame, bool]:
        """Block until the next simulator tick; return ``(frame, done)``."""


@dataclass
class LinkStats:
    """Per-tick timing. ``rtt_ms[i]`` is ``None`` on a tick that missed its deadline."""

    rtt_ms: list[float | None] = field(default_factory=list)
    compute_ms: list[float | None] = field(default_factory=list)

    @property
    def ticks(self) -> int:
        """Ticks run."""
        return len(self.rtt_ms)

    @property
    def misses(self) -> int:
        """Ticks that repeated the last control because no answer arrived in time."""
        return sum(rtt is None for rtt in self.rtt_ms)

    def summary(self) -> str:
        """One line: ticks, misses, and round-trip percentiles over answered ticks."""
        answered = [rtt for rtt in self.rtt_ms if rtt is not None]
        line = f"ticks={self.ticks} misses={self.misses}"
        if answered:
            p50, p95, p99 = np.percentile(answered, [50, 95, 99])
            line += f" rtt_ms p50={p50:.2f} p95={p95:.2f} p99={p99:.2f} max={max(answered):.2f}"
        return line

    def write_csv(self, path: Path) -> None:
        """One row per tick: ``tick, rtt_ms, compute_ms, missed``."""
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["tick", "rtt_ms", "compute_ms", "missed"])
            # No zip(strict=...): it is 3.10+, and this runs on AC's Python 3.9.
            for tick, (rtt, compute) in enumerate(zip(self.rtt_ms, self.compute_ms)):  # noqa: B905
                writer.writerow([tick, rtt, compute, int(rtt is None)])


class FlyLink:
    """The car's TCP connection to a fly side served by :func:`~.fly_side.serve`.

    Args:
        host: Fly side's address; ``127.0.0.1`` when both ends share a machine.
        port: Fly side's port.
        deadline_s: Longest one tick waits for an answer. Must leave room in the tick for
            AC's own step: at 50 Hz the tick is 20 ms.
        connect_timeout_s: How long to wait for the fly side to accept.
    """

    def __init__(
        self, host: str, port: int, deadline_s: float, connect_timeout_s: float = 10.0
    ) -> None:
        self.deadline_s = deadline_s
        self.stats = LinkStats()
        self.last_control = ControlVector.neutral()
        self._sock = socket.create_connection((host, port), timeout=connect_timeout_s)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buffer = bytearray()
        self._next_seq = 0
        self._pending: tuple[int, float] | None = None  # (seq, sent_at)
        self._reset_owed = False

    def exchange(self, frame: Frame, *, is_reset: bool = False) -> ControlVector:
        """Offer this tick's frame and return the control to apply now.

        The frame is sent only if nothing is in flight; otherwise it is dropped and the tick
        keeps waiting on the earlier one. An episode start that lands on a dropped frame is
        carried to the next frame sent, so the fly side always sees it.

        Returns:
            The fresh control if one arrived within the deadline, else the last control.
        """
        tick_start = time.perf_counter()
        self._reset_owed = self._reset_owed or is_reset
        if self._pending is None:
            send_frame(self._sock, self._next_seq, frame, is_reset=self._reset_owed)
            self._pending = (self._next_seq, tick_start)
            self._next_seq += 1
            self._reset_owed = False

        record = self._read_record(tick_start + self.deadline_s)
        if record is None:
            self.stats.rtt_ms.append(None)
            self.stats.compute_ms.append(None)
            return self.last_control

        seq, control, compute_ms = decode_control(record)
        pending_seq, sent_at = self._pending
        if seq != pending_seq:
            raise RuntimeError(f"fly side answered frame {seq}, expected {pending_seq}")
        self._pending = None
        self.last_control = control
        self.stats.rtt_ms.append((time.perf_counter() - sent_at) * 1000.0)
        self.stats.compute_ms.append(compute_ms)
        return control

    def close(self) -> None:
        """Hang up; the fly side's :func:`~.fly_side.serve` returns."""
        self._sock.close()

    def _read_record(self, until: float) -> bytes | None:
        """One control record if it completes before ``until``; partial bytes are kept."""
        while len(self._buffer) < CONTROL.size:
            remaining = until - time.perf_counter()
            if remaining <= 0:
                return None
            self._sock.settimeout(remaining)
            try:
                data = self._sock.recv(4096)
            except socket.timeout:  # noqa: UP041 -- not yet TimeoutError on Python 3.9
                return None
            if not data:
                raise ConnectionError("fly side closed the connection")
            self._buffer += data
        record = bytes(self._buffer[: CONTROL.size])
        del self._buffer[: CONTROL.size]
        return record


def run_episode(
    env: CarEnv, link: FlyLink, frame_shape: tuple[int, int, int], max_ticks: int
) -> LinkStats:
    """Drive one episode: frame to the fly, control to the car, once per tick.

    Args:
        env: The car simulator.
        link: Connected to the fly side.
        frame_shape: Checked on every frame before it leaves, so a misconfigured capture
            fails here rather than on the far machine.
        max_ticks: Stop after this many ticks if the episode has not ended.

    Returns:
        ``link.stats``.
    """
    frame = env.reset()
    is_reset = True
    for _ in range(max_ticks):
        control = link.exchange(validate_frame(frame, frame_shape), is_reset=is_reset)
        is_reset = False
        env.apply(control)
        frame, done = env.step()
        if done:
            break
    logger.info("episode over: %s", link.stats.summary())
    return link.stats
