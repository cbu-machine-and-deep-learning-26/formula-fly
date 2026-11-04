"""Focused starts (GH-103): the zones and share they promise, and nothing changed without one."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.practice_track import PracticeTrack
from fly_driver.envs.start_focus import StartFocus, draw_start

LAP_M = 5886.8
ZONES = ((470.0, 620.0), (1290.0, 1700.0), (4800.0, 5200.0))
FOCUS = StartFocus(share=0.5, zones=ZONES)


def test_without_a_focus_the_draw_is_the_uniform_draw_it_always_was() -> None:
    """A run that does not ask for a focus must be unchanged to the byte."""
    for seed in range(50):
        start_m, is_focused = draw_start(np.random.default_rng(seed), LAP_M, None)
        assert start_m == np.random.default_rng(seed).uniform(0.0, LAP_M)
        assert not is_focused


def test_a_focused_start_is_in_a_zone_and_the_share_is_what_was_asked_for() -> None:
    rng = np.random.default_rng(0)
    draws = [draw_start(rng, LAP_M, FOCUS) for _ in range(20_000)]
    focused = [start_m for start_m, is_focused in draws if is_focused]
    assert len(focused) / len(draws) == pytest.approx(FOCUS.share, abs=0.02)
    assert all(any(low <= start_m <= high for low, high in ZONES) for start_m in focused)


def test_every_zone_gets_the_same_practice_however_long_it_is() -> None:
    """T3's zone is 150 m and T10's 400 m. Sharing by length would starve the short one."""
    rng = np.random.default_rng(2)
    draws = [draw_start(rng, LAP_M, FOCUS) for _ in range(30_000)]
    focused = [start_m for start_m, is_focused in draws if is_focused]
    for low, high in ZONES:
        in_zone = sum(low <= start_m <= high for start_m in focused)
        assert in_zone / len(focused) == pytest.approx(1 / 3, abs=0.02)


def test_starts_fill_each_zone_from_end_to_end() -> None:
    rng = np.random.default_rng(3)
    draws = [draw_start(rng, LAP_M, FOCUS) for _ in range(30_000)]
    focused = [start_m for start_m, is_focused in draws if is_focused]
    for low, high in ZONES:
        in_zone = [start_m for start_m in focused if low <= start_m <= high]
        assert min(in_zone) < low + 0.02 * (high - low)
        assert max(in_zone) > high - 0.02 * (high - low)


def test_the_uniform_share_is_still_uniform_round_the_lap() -> None:
    """The lab supervisor compares this share with the runs before it, so it must not move."""
    rng = np.random.default_rng(1)
    uniform = [
        s
        for s, is_focused in (draw_start(rng, LAP_M, FOCUS) for _ in range(40_000))
        if not is_focused
    ]
    counts, _ = np.histogram(uniform, bins=10, range=(0.0, LAP_M))
    assert counts.min() > 0.9 * counts.mean()
    assert counts.max() < 1.1 * counts.mean()


@pytest.mark.parametrize(
    "share, zones",
    [
        pytest.param(0.0, ZONES, id="share-zero-is-off-not-a-focus"),
        pytest.param(1.0, ZONES, id="share-one-leaves-nothing-to-score"),
        pytest.param(-0.1, ZONES, id="share-negative"),
        pytest.param(0.5, (), id="no-zones"),
        pytest.param(0.5, ((5200.0, 4000.0),), id="zone-backwards"),
        pytest.param(0.5, ((4000.0, 4000.0),), id="zone-empty"),
        pytest.param(0.5, ((-1.0, 4000.0),), id="zone-before-the-line"),
        pytest.param(0.5, ((1.0, 2.0, 3.0),), id="zone-not-a-pair"),
        pytest.param(0.5, ((470.0, 620.0), (5200.0, 4000.0)), id="a-later-zone-is-bad"),
    ],
)
def test_a_bad_focus_is_refused(share: float, zones: tuple) -> None:
    with pytest.raises(ValueError, match="start_focus"):
        StartFocus(share, zones)


def test_a_focus_without_random_start_is_refused() -> None:
    with pytest.raises(ValueError, match="needs random_start"):
        PracticeTrack(start_focus=FOCUS)


@pytest.mark.render
class TestTheEnv:
    """Through :class:`PracticeTrack`, where the car actually gets put."""

    def test_a_zone_past_the_end_of_the_lap_is_refused(self) -> None:
        with pytest.raises(ValueError, match="past the end"):
            PracticeTrack(random_start=True, start_focus=StartFocus(0.5, ((5000.0, 6000.0),)))

    def test_a_mapping_from_config_builds_the_focus(self) -> None:
        zones = [[470.0, 620.0], [1290.0, 1700.0], [4800.0, 5200.0]]  # as YAML gives them
        env = PracticeTrack(random_start=True, start_focus={"share": 0.5, "zones": zones})
        try:
            assert env.start_focus == FOCUS
        finally:
            env.close()

    def test_the_car_is_put_where_the_draw_says_and_info_says_which_kind(self) -> None:
        env = PracticeTrack(random_start=True, start_focus=FOCUS)
        try:
            kinds = set()
            for seed in range(12):
                _, info = env.reset(seed=seed)
                expected_m, is_focused = draw_start(
                    np.random.default_rng(seed), env.centerline.length, FOCUS
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
        env = PracticeTrack(random_start=True, start_focus=FOCUS)
        try:
            seed = next(
                seed
                for seed in range(50)
                if draw_start(np.random.default_rng(seed), env.centerline.length, FOCUS)[1]
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
