"""Lap timing and lap-log tests (GH-16).

Lap detection is the sort of thing AGENTS.md section 11 is about: a timer that hands out a
lap for reversing over the start line still produces plausible-looking numbers, and the
same signal feeds GH-17's reward. So the awkward cases are driven explicitly rather than
assumed -- backwards over the line, a teleport, a shortcut, a half lap.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from fly_driver.envs.centerline import Centerline
from fly_driver.envs.lap import (
    LapLog,
    LapTimer,
    format_lap_time,
    parse_lap_time,
)


@pytest.fixture
def square() -> Centerline:
    """A 400 m square lap: 100 m a side, so the arithmetic is checkable by hand."""
    points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    return Centerline(points=points, half_width_right=[5.0] * 4, half_width_left=[5.0] * 4)


class TestFormatLapTime:
    def test_under_a_minute(self):
        assert format_lap_time(45.25) == "0:45.250"

    def test_over_a_minute(self):
        assert format_lap_time(87.097) == "1:27.097"

    def test_exactly_a_minute(self):
        assert format_lap_time(60.0) == "1:00.000"

    def test_pads_to_three_decimals(self):
        assert format_lap_time(62.5) == "1:02.500"

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            format_lap_time(-1.0)

    def test_round_trips_through_the_parser(self):
        for seconds in (0.5, 45.25, 87.097, 123.456, 600.0):
            assert parse_lap_time(format_lap_time(seconds)) == pytest.approx(seconds, abs=5e-4)


class TestParseLapTime:
    def test_minutes_and_seconds(self):
        assert parse_lap_time("1:27.097") == pytest.approx(87.097)

    def test_bare_seconds(self):
        assert parse_lap_time("87.097") == pytest.approx(87.097)

    def test_tolerates_surrounding_space(self):
        assert parse_lap_time("  1:27.097  ") == pytest.approx(87.097)

    @pytest.mark.parametrize("text", ["", "Time", "---", "abc", "1:2:3", "-5.0", "0", "1:"])
    def test_unreadable_text_is_none_not_an_exception(self, text):
        """The file is hand-edited; one bad row must not break the record book."""
        assert parse_lap_time(text) is None


class TestLapTimer:
    """Driven by feeding arclengths directly, which is what the sim loop does."""

    def _drive(self, timer, centerline, *, laps=1.0, steps=400, start_time=0.0, dt=0.1):
        """Drive smoothly forward, returning every lap time that came back."""
        times = []
        total = centerline.length * laps
        for i in range(1, steps + 1):
            s = (total * i / steps) % centerline.length
            result = timer.update(s, start_time + i * dt)
            if result is not None:
                times.append(result)
        return times

    def test_one_lap_round_is_one_lap(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(0.0, 0.0)
        times = self._drive(timer, square, laps=1.0, steps=400, dt=0.1)
        assert len(times) == 1
        assert times[0] == pytest.approx(40.0, abs=0.2)

    def test_three_laps_are_three_laps(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(0.0, 0.0)
        times = self._drive(timer, square, laps=3.0, steps=1200, dt=0.1)
        assert len(times) == 3
        assert timer.completed == 3

    def test_half_a_lap_is_no_lap(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(0.0, 0.0)
        assert self._drive(timer, square, laps=0.5, steps=200, dt=0.1) == []

    def test_reversing_over_the_line_does_not_award_a_lap(self, square):
        """Creep up to the line, back over it, and forward again. That is metres of
        driving, not a lap, and a naive crossing detector would count one or even two."""
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(0.0, 0.0)
        t = 0.0
        laps = []
        for s in (390.0, 395.0, 399.0, 2.0, 399.0, 2.0, 5.0, 399.0, 1.0):
            t += 0.1
            # start just behind the line so the first sample is a forward creep
            result = timer.update(s, t)
            if result is not None:
                laps.append(result)
        assert laps == []

    def test_going_backwards_must_be_made_up_again(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(0.0, 0.0)
        t = 0.0
        for s in np.linspace(0, 300, 60)[1:]:  # three quarters round
            t += 0.1
            timer.update(float(s), t)
        for s in np.linspace(300, 100, 40)[1:]:  # back to a quarter
            t += 0.1
            assert timer.update(float(s), t) is None
        got = None
        for s in np.linspace(100, 400, 60)[1:]:  # forward again, crossing the line
            t += 0.1
            got = timer.update(float(s) % square.length, t) or got
        assert got is not None, "a lap was driven in total and never registered"

    def test_a_teleport_does_not_count_as_progress(self, square):
        """A reset, or a car dropped back on track, must not gift most of a lap."""
        timer = LapTimer(square, min_lap_seconds=0.0, max_step_m=50.0)
        timer.reset(0.0, 0.0)
        timer.update(10.0, 0.1)
        assert timer.update(190.0, 0.2) is None  # 180 m in one step
        assert timer.lap_fraction == 0.0

    def test_implausibly_fast_laps_are_dropped(self, square):
        timer = LapTimer(square, min_lap_seconds=30.0)
        timer.reset(0.0, 0.0)
        times = self._drive(timer, square, laps=1.0, steps=400, dt=0.01)  # 4 s lap
        assert times == []

    def test_lap_fraction_tracks_progress(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(0.0, 0.0)
        for s in np.linspace(0, 200, 40)[1:]:
            timer.update(float(s), 1.0)
        assert timer.lap_fraction == pytest.approx(0.5, abs=0.02)

    def test_current_lap_time_counts_up_and_resets(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(0.0, 100.0)
        assert timer.current_lap_time(105.0) == pytest.approx(5.0)
        self._drive(timer, square, laps=1.0, steps=400, start_time=100.0, dt=0.1)
        assert timer.current_lap_time(140.5) == pytest.approx(0.5, abs=0.2)

    def test_the_first_sample_only_anchors(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        assert timer.update(123.0, 0.0) is None
        assert timer.lap_fraction == 0.0

    @pytest.mark.parametrize("kwargs", [{"min_lap_seconds": -1.0}, {"max_step_m": 0.0}])
    def test_rejects_bad_values(self, square, kwargs):
        with pytest.raises(ValueError):
            LapTimer(square, **kwargs)


class TestLapLog:
    def test_a_missing_file_has_no_best(self, tmp_path):
        assert LapLog(tmp_path / "nope.md").best() is None

    def test_recording_creates_the_file_with_a_header(self, tmp_path):
        log = LapLog(tmp_path / "lap_times.md")
        log.record(95.5, driver="gamepad")
        text = log.path.read_text(encoding="utf-8")
        assert text.startswith("# Lap times")
        assert "| Date | Time | Driver | Note |" in text
        assert "1:35.500" in text

    def test_the_first_lap_is_a_best(self, tmp_path):
        assert LapLog(tmp_path / "l.md").record(95.5) is True

    def test_a_slower_lap_is_not_a_best(self, tmp_path):
        log = LapLog(tmp_path / "l.md")
        log.record(95.5)
        assert log.record(101.2) is False

    def test_a_faster_lap_is_a_best(self, tmp_path):
        log = LapLog(tmp_path / "l.md")
        log.record(95.5)
        assert log.record(90.1) is True

    def test_best_is_the_fastest_row_not_the_last(self, tmp_path):
        log = LapLog(tmp_path / "l.md")
        for seconds in (100.0, 92.25, 97.0):
            log.record(seconds)
        assert log.best() == pytest.approx(92.25)

    def test_every_lap_is_appended(self, tmp_path):
        log = LapLog(tmp_path / "l.md")
        for seconds in (100.0, 92.25, 97.0):
            log.record(seconds)
        assert len(log.entries()) == 3

    def test_the_date_and_driver_are_written(self, tmp_path):
        log = LapLog(tmp_path / "l.md")
        log.record(95.5, driver="fly", when=datetime(2026, 9, 17, 20, 13, 45))
        entry = log.entries()[0]
        assert entry.when == "2026-09-17 20:13:45"
        assert entry.driver == "fly"

    def test_a_new_best_is_noted(self, tmp_path):
        log = LapLog(tmp_path / "l.md")
        log.record(100.0)
        log.record(90.0)
        assert [e.note for e in log.entries()] == ["new best", "new best"]
        log.record(95.0)
        assert log.entries()[-1].note == ""

    def test_deleting_a_row_by_hand_changes_the_best(self, tmp_path):
        """Payton's whole reason for keeping this in a document: "if i put down the best
        lap i need to go in and delete it so its only the flys best lap." """
        log = LapLog(tmp_path / "l.md")
        log.record(100.0, driver="fly")
        log.record(88.0, driver="payton")
        assert log.best() == pytest.approx(88.0)

        kept = [
            line
            for line in log.path.read_text(encoding="utf-8").splitlines()
            if "payton" not in line
        ]
        newline = chr(10)
        log.path.write_text(newline.join(kept) + newline, encoding="utf-8")

        assert log.best() == pytest.approx(100.0), "the deleted lap is still the record"

    def test_an_edit_is_picked_up_without_reconstructing_the_log(self, tmp_path):
        """The same LapLog object must see the change, because the sim holds one open."""
        log = LapLog(tmp_path / "l.md")
        log.record(100.0)
        assert log.best() == pytest.approx(100.0)
        with log.path.open("a", encoding="utf-8") as handle:
            handle.write("| 2026-09-17 21:00:00 | 1:20.000 | fly | |" + chr(10))
        assert log.best() == pytest.approx(80.0)

    def test_mangled_rows_are_skipped_not_fatal(self, tmp_path):
        path = tmp_path / "l.md"
        rows = [
            "| Date | Time | Driver | Note |",
            "| --- | --- | --- | --- |",
            "| 2026-09-17 | 1:35.500 | fly | |",
            "| oops this row is broken",
            "| 2026-09-17 | not a time | fly | |",
            "| 2026-09-17 | 1:30.000 | fly | |",
        ]
        path.write_text(chr(10).join(rows) + chr(10), encoding="utf-8")
        log = LapLog(path)
        assert len(log.entries()) == 2
        assert log.best() == pytest.approx(90.0)

    def test_an_emptied_table_has_no_best(self, tmp_path):
        log = LapLog(tmp_path / "l.md")
        log.record(95.0)
        kept = [
            line
            for line in log.path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("| 20")
        ]
        log.path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        assert log.best() is None

    def test_rejects_a_nonsense_lap(self, tmp_path):
        with pytest.raises(ValueError):
            LapLog(tmp_path / "l.md").record(0.0)

    def test_the_repo_default_path_is_the_document_at_the_root(self):
        from fly_driver.envs.lap import DEFAULT_LAP_LOG_PATH

        assert DEFAULT_LAP_LOG_PATH.name == "lap_times.md"
        assert (DEFAULT_LAP_LOG_PATH.parent / "pyproject.toml").exists()


@pytest.mark.slow
class TestOnTheRealCircuit:
    """The unit tests above feed synthetic arclengths. This one walks the car round
    Silverstone itself and goes through the same calls the drive script makes, because
    the failure that matters is a lap that never registers on the real geometry.
    """

    SPEED_MPS = 65.0

    @pytest.fixture(scope="class")
    def driven(self, tmp_path_factory):
        from fly_driver.envs.lap import LapLog

        centerline = Centerline.load()
        log = LapLog(tmp_path_factory.mktemp("laps") / "lap_times.md")
        timer = LapTimer(centerline)
        laps = []
        distance, now, dt = 0.0, 0.0, 1.0 / 50.0
        while now < 200.0:
            position, _ = centerline.pose_at(distance % centerline.length)
            projection = centerline.project(float(position[0]), float(position[1]))
            completed = timer.update(projection.arclength, now)
            if completed is not None:
                laps.append((completed, log.record(completed, driver="rails")))
            distance += self.SPEED_MPS * dt
            now += dt
        return centerline, log, laps

    def test_laps_are_detected(self, driven):
        _, _, laps = driven
        assert len(laps) == 2

    def test_the_time_is_the_distance_over_the_speed(self, driven):
        centerline, _, laps = driven
        expected = centerline.length / self.SPEED_MPS
        assert laps[0][0] == pytest.approx(expected, abs=0.05)

    def test_every_lap_reached_the_document(self, driven):
        _, log, laps = driven
        assert len(log.entries()) == len(laps)

    def test_deleting_the_record_by_hand_promotes_the_next_one(self, driven):
        """Payton's workflow, end to end on the real circuit."""
        _, log, _ = driven
        entries = sorted(e.seconds for e in log.entries())
        assert log.best() == pytest.approx(entries[0])

        text = log.path.read_text(encoding="utf-8")
        rows = [line for line in text.splitlines() if line.startswith("| 20")]
        fastest = min(rows, key=lambda row: parse_lap_time(row.split("|")[2]) or 1e9)
        kept = [row for row in rows if row != fastest]
        newline = chr(10)
        log.path.write_text(text.split("| 20")[0] + newline.join(kept) + newline, encoding="utf-8")

        assert log.best() == pytest.approx(entries[1])
