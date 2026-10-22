"""Human laps for behaviour cloning: the replay parser, controls, the dataset, the eye cache."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from fly_driver.interface import FRAME_RATE_HZ, FRAME_SHAPE
from fly_driver.training.bc_data import (
    HumanLaps,
    cache_eye_features,
    parse_replay,
    replay_controls,
)

INTERVAL_MS = 30.0
WINGS = 2


def _text(value: str) -> bytes:
    return struct.pack("<I", len(value)) + value.encode()


def write_replay(path: Path, rows: list[dict], *, version: int = 16, car: str = "ks_ferrari_sf70h"):
    """A minimal one-car replay, laid out as AC writes version 16."""
    header = struct.pack("<Id", version, INTERVAL_MS) + _text("4_mid_clear")
    header += _text("ks_silverstone") + _text("gp")
    header += struct.pack("<IIII", 1, len(rows), len(rows), 0)  # cars, index, frames, objects
    header += struct.pack("<I", 0) * len(rows)  # sun angles, no track objects
    header += _text(car) + _text("fruitjam4") + _text("") + _text("") + _text("00_default")
    header += struct.pack("<II", len(rows), WINGS) + bytes(20)
    body = b""
    for row in rows:
        frame = bytearray(256)
        struct.pack_into("<3f", frame, 0, *row["position"])
        struct.pack_into("<3e", frame, 164, *row["velocity"])
        struct.pack_into("<e", frame, 212, row["steer_deg"])
        frame[234] = row.get("gear", 3)
        frame[244] = row["gas"]
        frame[245] = row["brake"]
        frame[246] = row["lap"]
        body += bytes(frame) + bytes(20 + 4 * WINGS)
    path.write_bytes(header + body)
    return path


def _drive(count: int = 40, parked: int = 10, lap_change: int = 30) -> list[dict]:
    """``parked`` frames at rest on the brake, then driving with a steering sweep."""
    rows = []
    for index in range(count):
        moving = index >= parked
        rows.append(
            {
                "position": (float(index), 0.5, -2.0 * index),
                "velocity": (30.0 if moving else 0.0, 0.0, 40.0 if moving else 0.0),
                "steer_deg": float(index - parked) * 6.0 if moving else 0.0,
                "gas": 255 if moving else 0,
                "brake": 0 if moving else 255,
                "lap": 0 if index < lap_change else 1,
            }
        )
    return rows


def test_parse_replay_reads_the_player_car(tmp_path: Path) -> None:
    replay = parse_replay(write_replay(tmp_path / "drive.acreplay", _drive()))
    assert replay.car == "ks_ferrari_sf70h"
    assert replay.track == "ks_silverstone-gp"
    assert replay.interval_s == pytest.approx(0.03)
    assert len(replay.time_s) == 40
    assert replay.time_s[-1] == pytest.approx(39 * 0.03)
    np.testing.assert_allclose(replay.position_m[12], (12.0, 0.5, -24.0))
    assert replay.speed_mps[12] == pytest.approx(50.0)
    assert replay.steer_deg[12] == pytest.approx(12.0)
    assert replay.gas[12] == 1.0 and replay.brake[3] == 1.0
    assert replay.lap.tolist() == [0] * 30 + [1] * 10


def test_parse_replay_refuses_other_versions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="version 15"):
        parse_replay(write_replay(tmp_path / "old.acreplay", _drive(), version=15))


def test_controls_use_the_sf70h_lock_and_interpolate(tmp_path: Path) -> None:
    replay = parse_replay(write_replay(tmp_path / "drive.acreplay", _drive()))
    # Frame 25 has 90 deg of wheel: half the SF70H's 180 deg lock.
    controls = replay_controls(replay, [25 * 0.03, 25.5 * 0.03])
    np.testing.assert_allclose(controls[0], (0.5, 1.0, 0.0), atol=1e-3)
    assert controls[1, 0] == pytest.approx((90.0 + 96.0) / 2 / 180.0, abs=1e-3)
    # Frame 39 has 174 deg: past a 100 deg lock, steer clips to full.
    assert replay_controls(replay, [39 * 0.03], steer_lock_deg=100.0)[0, 0] == 1.0
    with pytest.raises(KeyError):
        replay_controls(
            parse_replay(write_replay(tmp_path / "x.acreplay", _drive(), car="unknown")), [0.0]
        )


def test_from_replay_drops_parked_frames_and_splits_laps(tmp_path: Path) -> None:
    rows = _drive(count=200, parked=60, lap_change=120)
    replay = parse_replay(write_replay(tmp_path / "drive.acreplay", rows))
    count = int(replay.duration_s * FRAME_RATE_HZ) + 1
    index = np.arange(count) % 256
    frames = index.astype(np.uint8)[:, None, None, None] * np.ones(FRAME_SHAPE, np.uint8)
    laps = HumanLaps.from_replay(replay, frames)
    # Parked until replay frame 60 (1.8 s); lap 1 from frame 120 (3.6 s). Frame 89, at
    # 1.78 s, is kept: the interpolated throttle is already a third on, the launch.
    assert laps.frames[0, 0, 0, 0] == 89
    assert len(laps) == count - 89
    assert [(stop - start) for start, stop in laps.segment_bounds()] == [91, count - 180]
    assert laps.laps[0] == 0 and laps.laps[-1] == 1
    assert np.all(laps.controls[1:, 1] == 1.0)
    with pytest.raises(ValueError, match="past the replay"):
        HumanLaps.from_replay(replay, np.zeros((count + 5, *FRAME_SHAPE), np.uint8))


def test_human_laps_round_trip_and_validation(tmp_path: Path) -> None:
    frames = np.zeros((6, *FRAME_SHAPE), np.uint8)
    controls = np.tile(np.array([0.1, 0.5, 0.0], np.float32), (6, 1))
    laps = HumanLaps(frames, controls, np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 1, 1, 1]))
    loaded = HumanLaps.load(laps.save(tmp_path / "laps.npz"))
    assert loaded.segment_bounds() == [(0, 3), (3, 6)]
    np.testing.assert_array_equal(loaded.controls, controls)
    with pytest.raises(ValueError, match="outside"):
        HumanLaps(frames, controls * 3, laps.segments, laps.laps)
    with pytest.raises(ValueError, match="uint8"):
        HumanLaps(frames[:, :48], controls, laps.segments, laps.laps)


class CountingEye:
    """Feature = frame mean plus the frames seen since the last reset."""

    feature_dim = 2

    def __init__(self) -> None:
        self.calls = 0
        self.since_reset = 0

    def reset(self) -> None:
        self.since_reset = 0

    def encode(self, frame: np.ndarray) -> np.ndarray:
        self.calls += 1
        self.since_reset += 1
        return np.array([frame.mean(), self.since_reset], np.float32)


def test_eye_cache_runs_segments_in_order_once(tmp_path: Path) -> None:
    frames = np.arange(5, dtype=np.uint8)[:, None, None, None] * np.ones(FRAME_SHAPE, np.uint8)
    laps = HumanLaps(frames, np.zeros((5, 3), np.float32), np.array([0, 0, 0, 1, 1]), np.zeros(5))
    eye = CountingEye()
    features = cache_eye_features(laps, eye, tmp_path / "cache.npz")
    assert features[:, 0].tolist() == [0, 1, 2, 3, 4]
    assert features[:, 1].tolist() == [1, 2, 3, 1, 2]  # reset at each segment's start
    again = cache_eye_features(laps, eye, tmp_path / "cache.npz")
    assert eye.calls == 5
    np.testing.assert_array_equal(again, features)
