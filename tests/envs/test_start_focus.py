"""Focused starts (GH-103): the zone and share they promise, and nothing changed without one."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.envs.start_focus import StartFocus, draw_start

LAP_M = 5886.8
ZONE = StartFocus(share=0.5, from_m=4000.0, to_m=5200.0)


def test_without_a_focus_the_draw_is_the_uniform_draw_it_always_was() -> None:
    """A run that does not ask for a focus must be unchanged to the byte."""
    for seed in range(50):
        start_m, is_focused = draw_start(np.random.default_rng(seed), LAP_M, None)
        assert start_m == np.random.default_rng(seed).uniform(0.0, LAP_M)
        assert not is_focused


def test_a_focused_start_is_in_the_zone_and_the_share_is_what_was_asked_for() -> None:
    rng = np.random.default_rng(0)
    draws = [draw_start(rng, LAP_M, ZONE) for _ in range(20_000)]
    focused = [start_m for start_m, is_focused in draws if is_focused]
    assert len(focused) / len(draws) == pytest.approx(ZONE.share, abs=0.02)
    assert min(focused) >= ZONE.from_m
    assert max(focused) <= ZONE.to_m


def test_the_uniform_share_is_still_uniform_round_the_lap() -> None:
    """The lab supervisor compares this share with the runs before it, so it must not move."""
    rng = np.random.default_rng(1)
    uniform = [
        s
        for s, is_focused in (draw_start(rng, LAP_M, ZONE) for _ in range(40_000))
        if not is_focused
    ]
    counts, _ = np.histogram(uniform, bins=10, range=(0.0, LAP_M))
    assert counts.min() > 0.9 * counts.mean()
    assert counts.max() < 1.1 * counts.mean()


@pytest.mark.parametrize(
    "share, from_m, to_m",
    [
        pytest.param(0.0, 4000.0, 5200.0, id="share-zero-is-off-not-a-focus"),
        pytest.param(1.0, 4000.0, 5200.0, id="share-one-leaves-nothing-to-score"),
        pytest.param(-0.1, 4000.0, 5200.0, id="share-negative"),
        pytest.param(0.5, 5200.0, 4000.0, id="zone-backwards"),
        pytest.param(0.5, 4000.0, 4000.0, id="zone-empty"),
        pytest.param(0.5, -1.0, 4000.0, id="zone-before-the-line"),
    ],
)
def test_a_bad_focus_is_refused(share: float, from_m: float, to_m: float) -> None:
    with pytest.raises(ValueError, match="start_focus"):
        StartFocus(share, from_m, to_m)


def test_a_focus_without_random_start_is_refused() -> None:
    with pytest.raises(ValueError, match="needs random_start"):
        PracticeTrack(start_focus=ZONE)


@pytest.mark.render
class TestTheEnv:
    """Through :class:`PracticeTrack`, where the car actually gets put."""

    def test_a_zone_past_the_end_of_the_lap_is_refused(self) -> None:
        with pytest.raises(ValueError, match="past the end"):
            PracticeTrack(random_start=True, start_focus=StartFocus(0.5, 5000.0, 6000.0))

    def test_a_mapping_from_config_builds_the_focus(self) -> None:
        env = PracticeTrack(
            random_start=True, start_focus={"share": 0.5, "from_m": 4000.0, "to_m": 5200.0}
        )
        try:
            assert env.start_focus == ZONE
        finally:
            env.close()

    def test_the_car_is_put_where_the_draw_says_and_info_says_which_kind(self) -> None:
        env = PracticeTrack(random_start=True, start_focus=ZONE)
        try:
            kinds = set()
            for seed in range(12):
                _, info = env.reset(seed=seed)
                expected_m, is_focused = draw_start(
                    np.random.default_rng(seed), env.centerline.length, ZONE
                )
                assert info["arclength_m"] == pytest.approx(expected_m, abs=0.05)
                assert info["start_focused"] == is_focused
                assert info["on_track"]
                assert info["speed_mps"] == 0.0
                kinds.add(is_focused)
            assert kinds == {True, False}, "twelve seeds should show both kinds"
        finally:
            env.close()

    def test_the_kind_stays_in_info_for_the_whole_episode(self) -> None:
        env = PracticeTrack(random_start=True, start_focus=ZONE)
        try:
            seed = next(
                seed
                for seed in range(50)
                if draw_start(np.random.default_rng(seed), env.centerline.length, ZONE)[1]
            )
            _, info = env.reset(seed=seed)
            assert info["start_focused"]
            for _ in range(5):
                info = env.step((0.0, 1.0, 0.0))[4]
                assert info["start_focused"]
        finally:
            env.close()

    def test_without_a_focus_starts_are_the_uniform_ones_and_never_marked(self) -> None:
        env = PracticeTrack(random_start=True)
        try:
            for seed in (0, 1, 2):
                _, info = env.reset(seed=seed)
                expected_m = np.random.default_rng(seed).uniform(0.0, env.centerline.length)
                assert info["arclength_m"] == pytest.approx(expected_m, abs=0.05)
                assert not info["start_focused"]
        finally:
            env.close()

    def test_off_by_default_the_grid_start_is_never_marked_focused(self) -> None:
        env = PracticeTrack()
        try:
            _, info = env.reset(seed=0)
            assert np.array_equal(env.data.qpos, env.model.qpos0)
            assert not info["start_focused"]
        finally:
            env.close()
