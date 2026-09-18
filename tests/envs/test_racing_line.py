"""Racing-line tests (GH-16).

The line is a guide to look at, not a trajectory anything is scored against, so what
matters is that it is on the circuit, closed, straighter than the centreline, and the same
every time. A line that wandered onto the grass would be actively misleading.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.centerline import Centerline
from fly_driver.envs.racing_line import racing_line, total_curvature


@pytest.fixture(scope="module")
def track() -> Centerline:
    return Centerline.load()


@pytest.fixture(scope="module")
def line(track) -> np.ndarray:
    return racing_line(track)


def _length(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))


class TestShape:
    def test_it_is_a_closed_loop(self, line):
        """First and last points are neighbours, not a gap the width of the circuit."""
        gap = float(np.linalg.norm(line[0] - line[-1]))
        assert gap < 20.0, f"{gap:.0f} m between the ends"

    def test_it_has_enough_points_to_look_smooth(self, line):
        assert len(line) > 500

    def test_no_duplicated_points(self, line):
        steps = np.linalg.norm(np.diff(line, axis=0), axis=1)
        assert steps.min() > 0.01

    def test_it_never_doubles_back(self, line):
        """Offsetting along the normal keeps it parameterised by the centreline, so every
        step must go forwards. A line that folded would render as a mess."""
        steps = np.roll(line, -1, axis=0) - line
        heading = steps / np.linalg.norm(steps, axis=1, keepdims=True)
        dots = np.einsum("ij,ij->i", heading, np.roll(heading, -1, axis=0))
        assert dots.min() > 0.0, "the line reverses on itself"


class TestItStaysOnTheCircuit:
    def test_every_point_is_between_the_edges(self, track, line):
        """The whole point. A racing line off the track is worse than none."""
        worst = max(track.project(float(x), float(y)).edge_overshoot for x, y in line)
        assert worst == 0.0, f"{worst:.2f} m past a track edge"

    def test_it_keeps_clear_of_the_edges_by_the_margin(self, track):
        """The line marks where a car's centre goes, so it has to leave room for the car."""
        margin = 1.2
        tight = racing_line(track, margin_m=margin)
        for x, y in tight:
            projection = track.project(float(x), float(y))
            edge = (
                projection.half_width_left
                if projection.lateral >= 0
                else projection.half_width_right
            )
            # resampling moves the projection station slightly, so allow a little slack
            assert edge - abs(projection.lateral) > margin - 0.5


class TestItIsARacingLine:
    def test_it_is_straighter_than_the_centreline(self, track, line):
        middle = track.resample(4.0).points
        assert total_curvature(line) < total_curvature(middle)

    def test_it_is_shorter_than_the_centreline(self, track, line):
        """Straightening the corners takes distance out of the lap."""
        assert _length(line) < _length(track.resample(4.0).points)

    def test_it_actually_uses_the_width_of_the_track(self, track, line):
        """A line that hugged the centre would not be a racing line at all."""
        offsets = [track.project(float(x), float(y)).lateral for x, y in line]
        assert max(offsets) > 3.0
        assert min(offsets) < -3.0

    def test_more_iterations_do_not_make_it_worse(self, track):
        few = racing_line(track, iterations=200)
        many = racing_line(track, iterations=3000)
        assert total_curvature(many) <= total_curvature(few)


class TestReproducible:
    def test_the_same_track_gives_the_same_line(self, track):
        assert np.array_equal(racing_line(track), racing_line(track))


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"spacing_m": 0.0},
            {"margin_m": -1.0},
            {"rate": 0.0},
            {"rate": 1.0},
        ],
    )
    def test_rejects_bad_arguments(self, track, kwargs):
        with pytest.raises(ValueError):
            racing_line(track, **kwargs)

    def test_rejects_a_margin_wider_than_the_track(self, track):
        with pytest.raises(ValueError):
            racing_line(track, margin_m=50.0)
