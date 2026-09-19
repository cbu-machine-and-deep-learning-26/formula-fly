"""Segmentation and segment timing (GH-16).

Two kinds of failure to guard against. The geometry can quietly produce nonsense -- sixty
segments, or three, or a set that does not cover the lap -- and every one of those still
"works" in the sense that nothing raises. And the timer can hand out a time for a stretch
that was not driven, which looks like a very fast segment rather than like a bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.centerline import Centerline
from fly_driver.envs.segments import (
    Segment,
    SegmentSettings,
    SegmentTimer,
    find_segments,
    signed_curvature,
)


@pytest.fixture(scope="module")
def silverstone() -> Centerline:
    return Centerline.load()


@pytest.fixture(scope="module")
def segments(silverstone) -> list[Segment]:
    return find_segments(silverstone)


def _circle(radius: float, count: int = 400) -> np.ndarray:
    angle = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack([radius * np.cos(angle), radius * np.sin(angle)], axis=1)


class TestCurvature:
    def test_a_circle_has_the_curvature_its_radius_implies(self):
        """1/r, everywhere. If this is wrong every threshold downstream means something
        other than the radius it claims to be."""
        for radius in (50.0, 200.0, 1000.0):
            measured = np.abs(signed_curvature(_circle(radius)))
            assert measured.mean() == pytest.approx(1.0 / radius, rel=0.02)

    def test_the_sign_says_which_way_the_corner_goes(self):
        """Positive is left, matching Centerline.normal. A flipped sign would mislabel
        every corner's direction and nothing else would notice."""
        anticlockwise = signed_curvature(_circle(100.0))
        assert anticlockwise.mean() > 0
        assert signed_curvature(_circle(100.0)[::-1]).mean() < 0

    def test_a_straight_line_has_none(self):
        line = np.stack([np.linspace(0, 1000, 200), np.zeros(200)], axis=1)
        # Endpoints wrap on a closed polyline, so judge the interior.
        assert np.abs(signed_curvature(line))[5:-5].max() < 1e-6


class TestTheSegmentationOfSilverstone:
    """The circuit is known, so the output can be judged rather than merely accepted."""

    def test_it_finds_a_plausible_number_of_segments(self, segments):
        """Silverstone has 18 numbered corners, several of which merge into complexes.
        Wide bounds on purpose: this is here to catch a threshold change that silently
        produces sixty segments or three, not to pin an exact count."""
        corners = [segment for segment in segments if segment.is_corner]
        assert 12 <= len(segments) <= 30, f"{len(segments)} segments"
        assert 7 <= len(corners) <= 20, f"{len(corners)} corners against Silverstone's 18"

    def test_the_segments_tile_the_lap_exactly(self, segments, silverstone):
        """No gaps and no overlaps, including across the start line. A gap loses a
        segment's time silently on every single lap."""
        assert segments[0].start_m == pytest.approx(0.0)
        assert segments[-1].end_m == pytest.approx(silverstone.length)
        for before, after in zip(segments, segments[1:], strict=False):
            assert before.end_m == pytest.approx(after.start_m)

    def test_corners_and_straights_alternate(self, segments):
        """Two corners in a row means the merge left a boundary with nothing on it."""
        kinds = [segment.is_corner for segment in segments]
        assert all(a != b for a, b in zip(kinds, kinds[1:], strict=False))

    def test_nothing_is_too_short_to_drive(self, segments):
        """A 15 m 'corner' is a kink in the survey. Timing one tells a driver nothing."""
        shortest = min(segment.length_m for segment in segments)
        assert shortest >= 50.0, f"{shortest:.0f} m segment"

    def test_it_finds_a_long_straight_and_a_slow_corner(self, segments):
        """Silverstone has Hangar Straight and it has Village. A segmentation that found
        neither would be uniform mush and would still pass the counting tests."""
        assert max(s.length_m for s in segments if not s.is_corner) > 600.0
        assert min(s.min_radius_m for s in segments if s.is_corner) < 80.0

    def test_names_are_unique_and_ordered(self, segments):
        names = [segment.name for segment in segments]
        assert len(set(names)) == len(names)
        assert [s.index for s in segments] == list(range(len(segments)))

    def test_every_corner_says_which_way_it_goes(self, segments):
        for segment in segments:
            assert (segment.turns_left is None) is (not segment.is_corner)

    def test_the_lap_turns_through_exactly_one_full_circle(self, silverstone):
        """The strongest check on curvature there is, and it needs no synthetic shape: for
        any simple closed curve the integral of kappa ds is exactly 2*pi. This runs on the
        real survey, so it catches a scale error that a hand-made circle might flatter --
        the earlier factor-of-two bug reads -0.499 here. Negative because Silverstone is
        driven clockwise."""
        points = silverstone.resample(2.0).points
        step = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
        turning = float(np.sum(signed_curvature(points) * step))
        assert turning / (2.0 * np.pi) == pytest.approx(-1.0, abs=0.01)

    def test_it_is_the_same_every_time(self, silverstone):
        """The lap document holds records per segment name, so a segmentation that moved
        between runs would silently compare times from different stretches of road."""
        first = find_segments(silverstone)
        second = find_segments(silverstone)
        assert [(s.name, s.start_m, s.end_m) for s in first] == [
            (s.name, s.start_m, s.end_m) for s in second
        ]


def _cornering_metres(segments: list[Segment]) -> float:
    return sum(segment.length_m for segment in segments if segment.is_corner)


class TestSettings:
    def test_a_looser_threshold_calls_more_of_the_lap_a_corner(self, silverstone):
        """Metres, not segment count. Raising the radius does not reliably produce *more*
        corners -- past a point each corner run grows until it swallows the short straight
        to the next one and the two merge into one complex, so the count can fall. What
        must rise monotonically is how much road is classified as cornering."""
        strict = find_segments(silverstone, SegmentSettings(enter_radius_m=120, exit_radius_m=200))
        loose = find_segments(silverstone, SegmentSettings(enter_radius_m=400, exit_radius_m=600))
        assert _cornering_metres(loose) > _cornering_metres(strict)

    def test_hysteresis_must_point_the_right_way(self):
        with pytest.raises(ValueError, match="harder to enter"):
            SegmentSettings(enter_radius_m=300.0, exit_radius_m=200.0)

    @pytest.mark.parametrize(
        "kwargs", [{"spacing_m": 0.0}, {"min_corner_m": -5.0}, {"smoothing_m": 0.0}]
    )
    def test_lengths_must_be_positive(self, kwargs):
        with pytest.raises(ValueError, match="must be positive"):
            SegmentSettings(**kwargs)

    def test_a_circuit_too_short_to_cut_is_an_error(self, silverstone):
        with pytest.raises(ValueError, match="too few"):
            find_segments(silverstone, SegmentSettings(spacing_m=2000.0))


def _straight_line_segments() -> list[Segment]:
    """Four equal segments round a 400 m lap, for timing tests with no geometry in them."""
    return [
        Segment(
            index=index,
            name=f"X{index}",
            is_corner=index % 2 == 0,
            turns_left=True if index % 2 == 0 else None,
            start_m=index * 100.0,
            end_m=(index + 1) * 100.0,
            min_radius_m=50.0 if index % 2 == 0 else float("inf"),
        )
        for index in range(4)
    ]


class TestTiming:
    def test_it_times_a_segment_from_entry_to_entry(self):
        timer = SegmentTimer(_straight_line_segments())
        assert timer.update(10.0, 0.0) is None  # anchors in X0
        assert timer.update(90.0, 3.0) is None
        done = timer.update(110.0, 4.0)
        assert done is not None
        assert done.segment.name == "X0"
        assert done.seconds == pytest.approx(4.0)

    def test_the_segment_being_driven_is_reported(self):
        timer = SegmentTimer(_straight_line_segments())
        timer.update(150.0, 0.0)
        assert timer.current is not None and timer.current.name == "X1"
        assert timer.elapsed(2.5) == pytest.approx(2.5)

    def test_rolling_backwards_over_a_line_awards_nothing(self):
        """Otherwise reversing across a boundary hands out a segment time for a few
        metres -- the same trap LapTimer avoids for whole laps."""
        timer = SegmentTimer(_straight_line_segments())
        timer.update(150.0, 0.0)
        assert timer.update(90.0, 1.0) is None
        assert timer.current is not None and timer.current.name == "X0"

    def test_being_put_back_on_the_grid_awards_nothing(self):
        timer = SegmentTimer(_straight_line_segments())
        timer.update(150.0, 0.0)
        assert timer.update(350.0, 1.0) is None, "a jump of two segments is not a lap"

    def test_it_wraps_at_the_start_line(self):
        timer = SegmentTimer(_straight_line_segments())
        timer.update(350.0, 0.0)
        done = timer.update(10.0, 2.0)
        assert done is not None and done.segment.name == "X3"

    def test_an_excursion_marks_the_whole_segment_dirty(self):
        """Sticky on purpose: a corner cut at its entry does not become clean by the exit."""
        timer = SegmentTimer(_straight_line_segments())
        timer.update(10.0, 0.0)
        timer.update(40.0, 1.0, off_track=True)
        timer.update(90.0, 2.0, off_track=False)
        assert timer.went_off_track
        done = timer.update(110.0, 3.0)
        assert done is not None and done.went_off_track and not done.is_clean

    def test_the_next_segment_starts_clean(self):
        timer = SegmentTimer(_straight_line_segments())
        timer.update(10.0, 0.0)
        timer.update(40.0, 1.0, off_track=True)
        timer.update(110.0, 2.0)
        assert not timer.went_off_track, "dirt carried into the next segment"

    def test_a_clean_run_is_eligible_for_a_record(self):
        timer = SegmentTimer(_straight_line_segments())
        timer.update(10.0, 0.0)
        done = timer.update(110.0, 5.0)
        assert done is not None and done.is_clean

    def test_it_needs_at_least_one_segment(self):
        with pytest.raises(ValueError, match="at least one"):
            SegmentTimer([])

    def test_lookup_covers_the_whole_lap(self, segments, silverstone):
        """Including the last metre, which is where an off-by-one hides."""
        timer = SegmentTimer(segments)
        for arclength in np.linspace(0.0, silverstone.length - 0.01, 500):
            found = timer.segment_at(float(arclength))
            assert found.start_m <= arclength < found.end_m + 1e-6
