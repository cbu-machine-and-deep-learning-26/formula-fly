"""Centerline geometry tests (GH-16).

`AGENTS.md` §11 names geometry orientation and reward shaping as failures that look
plausible when wrong, so the sign convention and the lap-seam behaviour are pinned with
hand-computed cases on a square track whose answers can be checked by eye, not just
asserted against the real circuit.

The square is deliberately trivial: 100 m sides, counter-clockwise, 5 m half-widths. Every
expected value below can be worked out on paper.
"""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.envs.centerline import DEFAULT_CENTERLINE_PATH, Centerline

#: Silverstone GP circuit official length, metres. Our polyline samples it at ~5 m, so it
#: reads slightly short; 0.5% is generous headroom for that while still catching a unit
#: error or a dropped segment.
SILVERSTONE_OFFICIAL_M = 5891.0


@pytest.fixture
def square() -> Centerline:
    """A 100 m counter-clockwise square with 5 m half-widths. Lap length 400 m."""
    points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    widths = [5.0] * 4
    return Centerline(points=points, half_width_right=widths, half_width_left=widths)


class TestConstruction:
    def test_closes_the_loop(self, square):
        assert square.num_segments == 4
        np.testing.assert_allclose(square.points[0], square.points[-1])

    def test_lap_length(self, square):
        assert square.length == pytest.approx(400.0)

    def test_arclength_is_monotonic(self, square):
        assert np.all(np.diff(square.arclength) > 0)

    def test_rejects_too_few_points(self):
        with pytest.raises(ValueError, match="at least 3 points"):
            Centerline([(0, 0), (1, 1)], [1, 1], [1, 1])

    def test_rejects_mismatched_widths(self):
        with pytest.raises(ValueError, match="must match points"):
            Centerline([(0, 0), (1, 0), (1, 1)], [1, 1], [1, 1, 1])

    def test_rejects_non_positive_widths(self):
        with pytest.raises(ValueError, match="positive"):
            Centerline([(0, 0), (1, 0), (1, 1)], [1, 0, 1], [1, 1, 1])

    def test_rejects_duplicate_consecutive_points(self):
        with pytest.raises(ValueError, match="zero length"):
            Centerline([(0, 0), (0, 0), (1, 1)], [1, 1, 1], [1, 1, 1])


class TestProjectionSignConvention:
    """Positive lateral is LEFT of travel. Asserted, never inferred."""

    def test_left_of_travel_is_positive(self, square):
        # On the bottom edge heading +x, "left" is +y.
        proj = square.project(50.0, 3.0)
        assert proj.lateral == pytest.approx(3.0)

    def test_right_of_travel_is_negative(self, square):
        proj = square.project(50.0, -3.0)
        assert proj.lateral == pytest.approx(-3.0)

    def test_convention_holds_on_a_perpendicular_segment(self, square):
        # On the right edge heading +y, "left" is -x. A point at x=97 is 3 m left.
        proj = square.project(97.0, 50.0)
        assert proj.lateral == pytest.approx(3.0)

    def test_on_centerline_is_zero(self, square):
        assert square.project(50.0, 0.0).lateral == pytest.approx(0.0, abs=1e-9)


class TestProjectionArclength:
    def test_start_of_lap(self, square):
        assert square.project(0.0, 0.0).arclength == pytest.approx(0.0)

    def test_quarter_of_the_way(self, square):
        assert square.project(50.0, 0.0).arclength == pytest.approx(50.0)

    def test_after_the_first_corner(self, square):
        # 100 m along the bottom, then 25 m up the right side.
        assert square.project(100.0, 25.0).arclength == pytest.approx(125.0)

    def test_point_far_off_track_still_projects(self, square):
        """Early RL drives into fields. The projection must stay sane out there."""
        proj = square.project(50.0, -500.0)
        assert proj.arclength == pytest.approx(50.0)
        assert proj.lateral == pytest.approx(-500.0)
        assert not proj.is_on_track

    def test_projects_onto_corner_when_outside_it(self, square):
        """A point diagonally outside a corner clamps to the corner, not past it."""
        proj = square.project(110.0, -10.0)
        assert proj.arclength == pytest.approx(100.0, abs=1e-6)


class TestOnTrack:
    def test_inside_both_edges(self, square):
        assert square.project(50.0, 4.9).is_on_track
        assert square.project(50.0, -4.9).is_on_track

    def test_outside_the_left_edge(self, square):
        proj = square.project(50.0, 5.5)
        assert not proj.is_on_track
        assert proj.edge_overshoot == pytest.approx(0.5)

    def test_outside_the_right_edge(self, square):
        proj = square.project(50.0, -5.5)
        assert not proj.is_on_track
        assert proj.edge_overshoot == pytest.approx(0.5)

    def test_on_track_has_no_overshoot(self, square):
        assert square.project(50.0, 2.0).edge_overshoot == 0.0

    def test_asymmetric_widths_are_respected(self):
        """Guards against left and right widths being swapped."""
        pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        track = Centerline(points=pts, half_width_right=[2.0] * 4, half_width_left=[8.0] * 4)
        assert track.project(50.0, 7.0).is_on_track  # 7 m left, limit 8
        assert not track.project(50.0, -7.0).is_on_track  # 7 m right, limit 2


class TestProgressDeltaAcrossTheSeam:
    """The reward integrates this. Getting it wrong punishes completing a lap."""

    def test_normal_forward_progress(self, square):
        assert square.progress_delta(10.0, 15.0) == pytest.approx(5.0)

    def test_normal_backward_progress(self, square):
        assert square.progress_delta(15.0, 10.0) == pytest.approx(-5.0)

    def test_crossing_the_line_forwards_is_positive(self, square):
        # 398 m -> 2 m is 4 m forward, not 396 m backward.
        assert square.progress_delta(398.0, 2.0) == pytest.approx(4.0)

    def test_crossing_the_line_backwards_is_negative(self, square):
        assert square.progress_delta(2.0, 398.0) == pytest.approx(-4.0)

    def test_a_full_lap_of_small_steps_sums_to_one_lap(self, square):
        """The property that actually matters: progress integrates to the lap length."""
        steps = np.linspace(0.0, square.length, 401)
        total = sum(
            square.progress_delta(float(a % square.length), float(b % square.length))
            for a, b in zip(steps[:-1], steps[1:], strict=True)
        )
        assert total == pytest.approx(square.length, rel=1e-9)


class TestPoseAt:
    def test_start_pose_faces_along_the_first_segment(self, square):
        position, yaw = square.pose_at(0.0)
        np.testing.assert_allclose(position, [0.0, 0.0], atol=1e-9)
        assert yaw == pytest.approx(0.0)  # heading +x

    def test_pose_after_the_first_corner_turns_left(self, square):
        position, yaw = square.pose_at(150.0)
        np.testing.assert_allclose(position, [100.0, 50.0], atol=1e-9)
        assert yaw == pytest.approx(np.pi / 2)  # heading +y

    def test_arclength_wraps_rather_than_clamping(self, square):
        wrapped = square.pose_at(square.length + 25.0)
        direct = square.pose_at(25.0)
        np.testing.assert_allclose(wrapped[0], direct[0], atol=1e-9)
        assert wrapped[1] == pytest.approx(direct[1])

    def test_pose_round_trips_through_project(self, square):
        for s in (0.0, 37.5, 100.0, 250.0, 399.9):
            position, _ = square.pose_at(s)
            assert square.project(*position).arclength == pytest.approx(s, abs=1e-6)


class TestEdges:
    def test_edges_are_offset_by_the_half_widths(self, square):
        left, right = square.edges()
        # First point sits at the origin on a segment heading +x, so left is +y.
        np.testing.assert_allclose(left[0], [0.0, 5.0], atol=1e-9)
        np.testing.assert_allclose(right[0], [0.0, -5.0], atol=1e-9)

    def test_edges_close(self, square):
        left, right = square.edges()
        np.testing.assert_allclose(left[0], left[-1], atol=1e-9)
        np.testing.assert_allclose(right[0], right[-1], atol=1e-9)

    def test_edge_separation_matches_total_width(self, square):
        left, right = square.edges()
        np.testing.assert_allclose(np.linalg.norm(left - right, axis=1), 10.0, atol=1e-9)


class TestResample:
    def test_preserves_lap_length(self, square):
        coarse = square.resample(20.0)
        assert coarse.length == pytest.approx(square.length, rel=0.02)

    def test_changes_point_count(self, square):
        assert square.resample(20.0).num_segments == 20

    def test_rejects_non_positive_spacing(self, square):
        with pytest.raises(ValueError, match="positive"):
            square.resample(0.0)

    def test_rejects_spacing_longer_than_the_lap(self, square):
        with pytest.raises(ValueError, match="lap length"):
            square.resample(1000.0)


@pytest.fixture(scope="module")
def track() -> Centerline:
    """The real vendored Silverstone data. Module-scoped: loading is I/O and the object
    is immutable, so every test in the file can share one."""
    return Centerline.load()


class TestVendoredSilverstone:
    """Checks the real data file, not a synthetic square."""

    def test_data_file_ships_with_the_package(self):
        assert DEFAULT_CENTERLINE_PATH.exists(), (
            f"{DEFAULT_CENTERLINE_PATH} missing; check package-data in pyproject.toml"
        )

    def test_length_matches_the_real_circuit(self, track):
        """Catches a unit error or dropped segments. Polyline sampling reads slightly
        short, so we allow 0.5% under but nothing over."""
        assert track.length == pytest.approx(SILVERSTONE_OFFICIAL_M, rel=0.005)

    def test_is_closed(self, track):
        np.testing.assert_allclose(track.points[0], track.points[-1])

    def test_point_spacing_is_roughly_uniform(self, track):
        spacing = np.linalg.norm(np.diff(track.points, axis=0), axis=1)
        # The closing segment is the data's own gap and is legitimately longer.
        assert spacing[:-1].min() > 4.0
        assert spacing[:-1].max() < 6.0

    def test_track_widths_are_plausible_for_an_f1_circuit(self, track):
        total = track.half_width_left + track.half_width_right
        assert total.min() > 10.0
        assert total.max() < 20.0

    def test_projection_is_self_consistent_around_the_whole_lap(self, track):
        """Walk the lap and confirm every point projects back to where it came from.
        A chirality or ordering error shows up here as a lateral offset that is not zero."""
        for s in np.linspace(0.0, track.length, 200, endpoint=False):
            position, _ = track.pose_at(float(s))
            proj = track.project(float(position[0]), float(position[1]))
            assert proj.lateral == pytest.approx(0.0, abs=1e-6)
            assert proj.arclength == pytest.approx(s, abs=1e-3)

    def test_progress_integrates_to_one_lap(self, track):
        steps = np.linspace(0.0, track.length, 2000)
        total = sum(
            track.progress_delta(float(a % track.length), float(b % track.length))
            for a, b in zip(steps[:-1], steps[1:], strict=True)
        )
        assert total == pytest.approx(track.length, rel=1e-6)
