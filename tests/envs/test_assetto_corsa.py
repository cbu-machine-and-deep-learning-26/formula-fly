"""AssettoCorsaTrack against FakeAc: resets, the seam, laps, missed ticks, YAML."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from fly_driver.bridge.env_link import serve_env
from fly_driver.bridge.fake_ac import FakeAc
from fly_driver.bridge.fly_side import open_listener
from fly_driver.envs.assetto_corsa import AssettoCorsaTrack
from fly_driver.interface import FRAME_RATE_HZ, FRAME_SHAPE, ControlVector

FULL = ControlVector(0.0, 1.0, 0.0)


def make_env(**kwargs: object) -> tuple[AssettoCorsaTrack, FakeAc]:
    fake_kwargs = {key: kwargs.pop(key) for key in ("start_m", "track_length_m") if key in kwargs}
    car = FakeAc(realtime=False, drift=0.0, **fake_kwargs)
    return AssettoCorsaTrack(car, **kwargs), car


def drive(env: AssettoCorsaTrack, control: ControlVector, steps: int) -> list[tuple]:
    results = []
    for _ in range(steps):
        result = env.step(control)
        results.append(result)
        if result[2] or result[3]:
            break
    return results


def test_off_track_terminates_and_reset_puts_the_car_back_on_the_road() -> None:
    env, car = make_env()
    env.reset()
    *_, (_, reward, terminated, truncated, info) = drive(env, ControlVector(1.0, 0.5, 0.0), 2000)
    assert terminated and not truncated
    assert info["off_track_fraction"] > 0.5 and info["reward_terms"]["off_track"] < 0
    stopped_at = car.lap_dist_m

    _, info = env.reset()
    assert info["on_track"] and info["tyres_out"] == 0 and info["speed_mps"] == 0.0
    assert info["arclength_m"] == pytest.approx(stopped_at)  # where it stopped, not a grid
    assert not env.step(FULL)[2]


def test_stall_terminates_after_stall_s_and_the_next_episode_drives() -> None:
    env, _ = make_env(stall_s=1.0)
    env.reset()
    results = drive(env, ControlVector.neutral(), 500)
    assert len(results) == FRAME_RATE_HZ  # one second of ticks at rest
    assert results[-1][2] and results[-1][4]["stalled"]

    env.reset()
    results = drive(env, FULL, 200)
    assert len(results) == 200 and not results[-1][4]["stalled"]


def test_step_cap_truncates_and_reset_starts_over() -> None:
    env, _ = make_env(max_steps=10)
    for _ in range(2):
        _, info = env.reset()
        assert info["steps"] == 0
        results = drive(env, FULL, 100)
        assert len(results) == 10
        assert results[-1][3] and not results[-1][2]
    with pytest.raises(RuntimeError):
        env.step(FULL)


def test_progress_is_positive_across_the_start_finish_line() -> None:
    env, car = make_env(start_m=590.0)
    _, info = env.reset()
    results = drive(env, FULL, 100)
    progress = [result[4]["progress_m"] for result in results]
    assert min(progress) >= 0.0
    assert car.lap_dist_m < 590.0  # it did wrap
    assert sum(progress) == pytest.approx((car.lap_dist_m - 590.0) % 600.0)


def test_a_full_lap_is_detected_once_and_timed_from_the_first_crossing() -> None:
    # Out lap to the line, then one 600 m lap at a steady speed.
    env, car = make_env(start_m=590.0)
    env.reset()
    crossed_at = None
    laps = []
    for _ in range(5000):
        throttle = 0.0 if car.speed_mps > 25.0 else 1.0
        _, reward, terminated, truncated, info = env.step(ControlVector(0.0, throttle, 0.0))
        assert not (terminated or truncated)
        if crossed_at is None and not info["on_out_lap"]:
            crossed_at = info["sim_time"]
        if info["lap_complete"]:
            laps.append((info["lap_time"], info["sim_time"]))
            assert info["reward_terms"]["lap_bonus"] > 0
            break
    assert len(laps) == 1
    lap_time, now = laps[0]
    assert lap_time == pytest.approx(now - crossed_at, abs=2.0 / FRAME_RATE_HZ)
    assert 600.0 / 26.0 < lap_time < 600.0 / 24.0


def test_a_late_step_counts_its_missed_ticks_and_truncates_past_the_limit() -> None:
    car = FakeAc()  # real time
    env = AssettoCorsaTrack(car, max_missed_ticks=3)
    env.reset()
    assert env.step(FULL)[4]["missed_ticks"] == 0

    time.sleep(0.1)  # five ticks at 50 Hz
    _, _, terminated, truncated, info = env.step(FULL)
    assert info["missed_ticks"] >= 4
    assert env.missed_ticks == info["missed_ticks"]
    assert truncated and not terminated


def test_a_custom_reward_plugs_in_unchanged() -> None:
    seen = []

    def speed_reward(info, dt):
        seen.append(sorted(info))
        return info["speed_mps"] * dt, {"speed": info["speed_mps"] * dt}

    env, _ = make_env(reward=speed_reward)
    env.reset()
    _, reward, *_, info = env.step(FULL)
    assert reward == info["reward_terms"]["speed"] > 0
    for key in ("progress_m", "lateral_m", "on_track", "off_track", "lap_complete", "speed_mps"):
        assert key in seen[0]


def test_env_type_assetto_corsa_builds_from_yaml(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from fly_driver.training.config import TrainConfig
    from fly_driver.training.train import build_env

    listener = open_listener("127.0.0.1", 0)
    port = listener.getsockname()[1]
    hello = {
        "track": "ks_silverstone-gp",
        "car": "dallara_f317",
        "frame_shape": list(FRAME_SHAPE),
        "frame_rate_hz": FRAME_RATE_HZ,
    }
    threading.Thread(
        target=serve_env, args=(FakeAc(realtime=False), listener, hello), daemon=True
    ).start()
    path = tmp_path / "ac.yaml"
    path.write_text(
        "env:\n"
        "  type: assetto_corsa\n"
        "  max_steps: 7\n"
        f"  params: {{host: 127.0.0.1, port: {port}, track: ks_silverstone-gp,"
        " car: dallara_f317, stall_s: 5.0}\n"
    )
    config = TrainConfig.from_yaml(path)
    try:
        env = build_env(config.env)
        assert isinstance(env, AssettoCorsaTrack)
        env.reset()
        assert len(drive(env, FULL, 100)) == 7
        env.close()
        with pytest.raises(ValueError, match="serves"):
            build_env(config.env.__class__(type="assetto_corsa", params={"port": port, "car": "x"}))
    finally:
        listener.close()
