"""Run-time channel overrides (GH-104), on stub drivers and the real track."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from fly_driver.analysis.channel_overrides import (
    ChannelOverride,
    ChannelSpec,
    WatchedEnv,
    governor_pedals,
    scripted_steer,
)
from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.interface import ControlVector

PROFILE_M = np.array([0.0, 6000.0])
PROFILE_MPS = np.array([100.0, 100.0])
OWN = ControlVector(steer=0.25, throttle=0.5, brake=0.75)


class _Driver:
    def __init__(self) -> None:
        self.frames: list[Any] = []
        self.reset_seeds: list[int | None] = []

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)

    def act(self, frame: Any) -> ControlVector:
        self.frames.append(frame)
        return OWN


def _override(label: str, *, speed_mps: float, arclength_m: float = 100.0) -> ChannelOverride:
    board = {"env": None, "info": {"speed_mps": speed_mps, "arclength_m": arclength_m}}
    return ChannelOverride(
        _Driver(),
        board,
        ChannelSpec.parse(label),
        PROFILE_M,
        PROFILE_MPS,
        steer_function=lambda env, info: -0.5,
    )


@pytest.mark.parametrize(
    ("label", "spec"),
    [
        ("own", ChannelSpec()),
        ("gov_0.85", ChannelSpec(governor=0.85)),
        ("gov_1", ChannelSpec(governor=1.0)),
        ("vale_85", ChannelSpec(vale_cap_kmh=85.0)),
        ("script_own", ChannelSpec(script_steer=True)),
        ("script_gov_0.6", ChannelSpec(script_steer=True, governor=0.6)),
    ],
)
def test_labels_parse(label: str, spec: ChannelSpec) -> None:
    assert ChannelSpec.parse(label) == spec


@pytest.mark.parametrize("label", ["", "gov", "gov_x", "script", "script_vale_85", "vale_", "ownn"])
def test_an_unknown_label_is_refused(label: str) -> None:
    with pytest.raises(ValueError, match="unknown channel condition"):
        ChannelSpec.parse(label)


@pytest.mark.parametrize(
    ("speed", "pedals"),
    [
        (99.0, (1.0, 0.0)),  # below: flat out
        (101.0, (0.0, 0.0)),  # just over: coast
        (102.0, (0.0, 0.0)),  # at the margin: still coast
        (103.0, (0.0, 1.0)),  # past the margin: brake
    ],
)
def test_governor_pedals(speed: float, pedals: tuple[float, float]) -> None:
    assert governor_pedals(speed, 100.0) == pedals


def test_own_changes_nothing() -> None:
    assert _override("own", speed_mps=50.0).act(None) == OWN


def test_scripted_steering_replaces_only_the_steering() -> None:
    control = _override("script_own", speed_mps=50.0).act(None)
    assert control == ControlVector(steer=-0.5, throttle=OWN.throttle, brake=OWN.brake)


def test_the_governor_replaces_only_the_pedals_and_scales_the_profile() -> None:
    # Target is 0.5 * 100 = 50 m/s.
    below = _override("gov_0.5", speed_mps=40.0).act(None)
    above = _override("gov_0.5", speed_mps=60.0).act(None)
    assert below == ControlVector(steer=OWN.steer, throttle=1.0, brake=0.0)
    assert above == ControlVector(steer=OWN.steer, throttle=0.0, brake=1.0)


@pytest.mark.parametrize(
    ("arclength", "speed", "capped"),
    [
        (5400.0, 30.0, True),  # in the zone, over 85 km/h
        (5400.0, 20.0, False),  # in the zone, under it
        (5000.0, 30.0, False),  # before the zone
        (5700.0, 30.0, False),  # after it
    ],
)
def test_the_vale_cap_only_applies_in_its_zone(
    arclength: float, speed: float, capped: bool
) -> None:
    control = _override("vale_85", speed_mps=speed, arclength_m=arclength).act(None)
    expected = ControlVector(steer=OWN.steer, throttle=0.0, brake=1.0) if capped else OWN
    assert control == expected


def test_the_driver_runs_every_frame_even_when_overridden() -> None:
    override = _override("script_gov_0.5", speed_mps=50.0)
    override.act("first")
    override.act("second")
    assert override.driver.frames == ["first", "second"]


def test_reset_is_forwarded_to_the_driver() -> None:
    override = _override("own", speed_mps=0.0)
    override.reset(seed=4)
    assert override.driver.reset_seeds == [4]


class _StubEnv:
    frame_rate_hz = 50.0

    def reset(self, *, seed: int | None = None) -> tuple[str, dict[str, Any]]:
        del seed
        return "frame", {"arclength_m": 1.0}

    def step(self, action: Any) -> tuple[str, float, bool, bool, dict[str, Any]]:
        del action
        return "frame", 0.0, False, False, {"arclength_m": 2.0}


def test_watched_env_posts_the_env_and_its_latest_info() -> None:
    stub, board = _StubEnv(), {}
    watched = WatchedEnv(stub, board)
    watched.reset(seed=0)
    assert board["env"] is stub
    assert board["info"]["arclength_m"] == 1.0
    watched.step(None)
    assert board["info"]["arclength_m"] == 2.0
    assert watched.frame_rate_hz == 50.0


def test_scripted_steering_drives_the_real_car_along_the_track() -> None:
    env = PracticeTrack()
    try:
        try:
            _, info = env.reset(seed=0)
        except Exception as exc:  # pragma: no cover - depends on the machine
            pytest.skip(f"no offscreen GL context: {exc}")
        ended = False
        for _ in range(300):
            steer = scripted_steer(env, info)
            assert -1.0 <= steer <= 1.0
            speed = float(info.get("speed_mps", 0.0))
            throttle, brake = governor_pedals(speed, 15.0)
            control = ControlVector(steer=steer, throttle=throttle, brake=brake)
            _, _, terminated, truncated, info = env.step(control)
            ended = terminated or truncated
            if ended:
                break
        assert not ended
        assert info["on_track"]
    finally:
        env.close()
