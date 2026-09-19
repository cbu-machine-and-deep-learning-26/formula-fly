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
    SEGMENT_TABLE_END,
    SEGMENT_TABLE_START,
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
    """Driven by feeding arclengths directly, which is what the sim loop does.

    These start timing from ``reset`` so they test accumulate-and-complete on its own;
    the out-lap behaviour that the drive script uses is in :class:`TestOutLap`.
    """

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
        timer = LapTimer(square, min_lap_seconds=0.0, start_on_crossing=False)
        timer.reset(0.0, 0.0)
        times = self._drive(timer, square, laps=1.0, steps=400, dt=0.1)
        assert len(times) == 1
        assert times[0] == pytest.approx(40.0, abs=0.2)

    def test_three_laps_are_three_laps(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0, start_on_crossing=False)
        timer.reset(0.0, 0.0)
        times = self._drive(timer, square, laps=3.0, steps=1200, dt=0.1)
        assert len(times) == 3
        assert timer.completed == 3

    def test_half_a_lap_is_no_lap(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0, start_on_crossing=False)
        timer.reset(0.0, 0.0)
        assert self._drive(timer, square, laps=0.5, steps=200, dt=0.1) == []

    def test_reversing_over_the_line_does_not_award_a_lap(self, square):
        """Creep up to the line, back over it, and forward again. That is metres of
        driving, not a lap, and a naive crossing detector would count one or even two."""
        timer = LapTimer(square, min_lap_seconds=0.0, start_on_crossing=False)
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
        timer = LapTimer(square, min_lap_seconds=0.0, start_on_crossing=False)
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
        timer = LapTimer(square, min_lap_seconds=0.0, max_step_m=50.0, start_on_crossing=False)
        timer.reset(0.0, 0.0)
        timer.update(10.0, 0.1)
        assert timer.update(190.0, 0.2) is None  # 180 m in one step
        assert timer.lap_fraction == 0.0

    def test_implausibly_fast_laps_are_dropped(self, square):
        timer = LapTimer(square, min_lap_seconds=30.0, start_on_crossing=False)
        timer.reset(0.0, 0.0)
        times = self._drive(timer, square, laps=1.0, steps=400, dt=0.01)  # 4 s lap
        assert times == []

    def test_lap_fraction_tracks_progress(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0, start_on_crossing=False)
        timer.reset(0.0, 0.0)
        for s in np.linspace(0, 200, 40)[1:]:
            timer.update(float(s), 1.0)
        assert timer.lap_fraction == pytest.approx(0.5, abs=0.02)

    def test_current_lap_time_counts_up_and_resets(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0, start_on_crossing=False)
        timer.reset(0.0, 100.0)
        assert timer.current_lap_time(105.0) == pytest.approx(5.0)
        self._drive(timer, square, laps=1.0, steps=400, start_time=100.0, dt=0.1)
        assert timer.current_lap_time(140.5) == pytest.approx(0.5, abs=0.2)

    def test_the_first_sample_only_anchors(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0, start_on_crossing=False)
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
        assert "| Date | Lap | Penalty | Raw | Driver | Note |" in text
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
        # Behind the line, exactly as the car is placed, so this exercises the out lap.
        distance = centerline.length - 150.0
        now, dt = 0.0, 1.0 / 50.0
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


class TestOutLap:
    """Payton: "start the car before the starting line, when it passes start the first lap
    timer, rather than it starting when booting in."

    So the clock is not running when the simulator opens. It starts the moment the car
    first crosses the line, and the run-up does not count against the first lap.
    """

    def _creep(self, timer, square, path, *, dt=0.1, start=0.0):
        """Feed a sequence of arclengths, returning any completed laps."""
        laps, t = [], start
        for s in path:
            t += dt
            result = timer.update(float(s) % square.length, t)
            if result is not None:
                laps.append(result)
        return laps

    def test_the_clock_is_not_running_at_boot(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(square.length - 40.0, 0.0)
        assert not timer.timing
        assert timer.current_lap_time(10.0) == 0.0

    def test_crossing_the_line_starts_it(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(square.length - 40.0, 0.0)
        self._creep(timer, square, np.arange(square.length - 38.0, square.length + 10.0, 2.0))
        assert timer.timing

    def test_the_run_up_is_not_charged_to_the_first_lap(self, square):
        """The whole point. Drive 200 m up to the line, then a lap: the lap time is the
        lap, not the lap plus however long the car sat there at boot."""
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(square.length - 200.0, 0.0)
        # 200 m of out lap at 2 m per 0.1 s step = 10 s that must not be counted
        out = np.arange(square.length - 198.0, square.length + 1.0, 2.0)
        self._creep(timer, square, out)
        assert timer.completed == 0

        lap = np.arange(2.0, square.length + 3.0, 2.0)
        laps = self._creep(timer, square, lap, start=len(out) * 0.1)
        assert len(laps) == 1
        assert laps[0] == pytest.approx(square.length / 20.0, abs=0.15)

    def test_an_out_lap_that_never_reaches_the_line_times_nothing(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(square.length - 100.0, 0.0)
        self._creep(timer, square, np.arange(square.length - 98.0, square.length - 10.0, 2.0))
        assert not timer.timing
        assert timer.completed == 0

    def test_lap_fraction_stays_at_zero_on_the_out_lap(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(square.length - 100.0, 0.0)
        self._creep(timer, square, np.arange(square.length - 98.0, square.length - 20.0, 2.0))
        assert timer.lap_fraction == 0.0

    def test_reversing_before_the_line_still_does_not_start_it(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0)
        timer.reset(square.length - 30.0, 0.0)
        self._creep(
            timer, square, [square.length - 20.0, square.length - 25.0, square.length - 15.0]
        )
        assert not timer.timing

    def test_timing_from_reset_is_still_available(self, square):
        """A training env that starts on the line wants the clock running immediately."""
        timer = LapTimer(square, min_lap_seconds=0.0, start_on_crossing=False)
        timer.reset(0.0, 0.0)
        assert timer.timing
        assert timer.current_lap_time(5.0) == pytest.approx(5.0)


class TestSubStepAccuracy:
    """A control step is 20 ms at 50 Hz. Charging a whole one to every lap would put a
    fifth of a tenth on each time in the record book, always in the same direction."""

    def test_the_lap_time_is_not_quantised_to_the_step(self, square):
        """Drive a lap whose true time falls between two samples and check the answer is
        the true time, not the sample boundary."""
        timer = LapTimer(square, min_lap_seconds=0.0, start_on_crossing=False)
        timer.reset(0.0, 0.0)
        speed, dt = 7.0, 0.1  # 400 m / 7 = 57.142857 s, not a multiple of 0.1
        distance, t = 0.0, 0.0
        got = None
        while t < 120.0:
            distance += speed * dt
            t += dt
            result = timer.update(distance % square.length, t)
            if result is not None:
                got = result
                break
        assert got == pytest.approx(square.length / speed, abs=1e-3)

    def test_consecutive_laps_do_not_drift(self, square):
        timer = LapTimer(square, min_lap_seconds=0.0, start_on_crossing=False)
        timer.reset(0.0, 0.0)
        speed, dt = 7.0, 0.1
        distance, t, laps = 0.0, 0.0, []
        while t < 300.0:
            distance += speed * dt
            t += dt
            result = timer.update(distance % square.length, t)
            if result is not None:
                laps.append(result)
        assert len(laps) >= 4
        for lap in laps:
            assert lap == pytest.approx(square.length / speed, abs=1e-3)


class TestPenalties:
    """The lap that counts is raw plus penalty, and the two are kept visible separately.

    The failure this guards is quiet: a cut lap filed as though it were clean holds the
    record, and the document gives no sign that anything happened.
    """

    def test_the_penalty_is_added_to_the_recorded_time(self, tmp_path):
        log = LapLog(tmp_path / "l.md")
        log.record(90.0, penalty_seconds=5.0)
        assert log.best() == pytest.approx(95.0)

    def test_a_clean_lap_beats_a_faster_dirty_one(self, tmp_path):
        """The reason the penalty is added before comparing rather than after."""
        log = LapLog(tmp_path / "l.md")
        assert log.record(90.0, penalty_seconds=7.0) is True
        assert log.record(95.0) is True, "a clean 95 beats a 90 that cost 7 seconds"
        assert log.best() == pytest.approx(95.0)

    def test_raw_and_penalty_are_both_readable_afterwards(self, tmp_path):
        log = LapLog(tmp_path / "l.md")
        log.record(90.0, penalty_seconds=5.25, driver="gamepad")
        entry = log.entries()[0]
        assert entry.seconds == pytest.approx(95.25)
        assert entry.penalty_seconds == pytest.approx(5.25)
        assert entry.raw_seconds == pytest.approx(90.0)
        assert entry.driver == "gamepad"
        assert not entry.is_clean

    def test_a_lap_with_no_penalty_is_clean(self, tmp_path):
        log = LapLog(tmp_path / "l.md")
        log.record(90.0, driver="fly", note="hello")
        entry = log.entries()[0]
        assert entry.is_clean
        assert entry.raw_seconds == pytest.approx(entry.seconds)
        assert (entry.driver, entry.note) == ("fly", "hello")

    def test_rows_written_before_penalties_existed_still_read(self, tmp_path):
        """Payton's 1:52.229 is on file in the four-column layout. Losing it to a schema
        change would be the worst kind of silent break: the record book still parses, and
        the human lap the fly is measured against has simply gone."""
        path = tmp_path / "l.md"
        path.write_text(
            "| Date | Time | Driver | Note |\n"
            "| --- | --- | --- | --- |\n"
            "| 2026-09-18 22:14:03 | 1:52.229 | gamepad (PS5 Controller) | new best |\n",
            encoding="utf-8",
        )
        entry = LapLog(path).entries()[0]
        assert entry.seconds == pytest.approx(112.229)
        assert entry.penalty_seconds == 0.0
        assert entry.driver == "gamepad (PS5 Controller)"
        assert entry.note == "new best"

    def test_old_and_new_rows_compare_against_each_other(self, tmp_path):
        path = tmp_path / "l.md"
        path.write_text(
            "| Date | Time | Driver | Note |\n"
            "| --- | --- | --- | --- |\n"
            "| 2026-09-18 22:14:03 | 1:52.229 | gamepad | new best |\n",
            encoding="utf-8",
        )
        log = LapLog(path)
        assert log.record(100.0, penalty_seconds=20.0) is False, "120s does not beat 112.229"
        assert log.best() == pytest.approx(112.229)

    def test_an_unreadable_penalty_cell_costs_only_that_row_its_penalty(self, tmp_path):
        path = tmp_path / "l.md"
        path.write_text(
            "| 2026-09-19 10:00:00 | 1:35.500 | -- | 1:35.500 | fly |  |\n", encoding="utf-8"
        )
        entry = LapLog(path).entries()[0]
        assert entry.seconds == pytest.approx(95.5)
        assert entry.penalty_seconds == 0.0

    def test_a_negative_penalty_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="non-negative"):
            LapLog(tmp_path / "l.md").record(90.0, penalty_seconds=-1.0)


class TestTheSegmentFence:
    """Segment rows live in the same document and look exactly like lap rows.

    Nothing raises if the fence stops working -- the record book simply reports a
    four-second lap, which is why this is tested rather than trusted.
    """

    def _with_segments(self, path, *rows):
        path.write_text(
            "| Date | Lap | Penalty | Raw | Driver | Note |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| 2026-09-19 10:00:00 | 1:35.500 | 0.000 | 1:35.500 | fly |  |\n"
            "\n" + SEGMENT_TABLE_START + "\n" + "".join(rows) + SEGMENT_TABLE_END + "\n",
            encoding="utf-8",
        )
        return LapLog(path)

    def test_a_segment_best_does_not_become_the_fastest_lap(self, tmp_path):
        log = self._with_segments(
            tmp_path / "l.md",
            "| Segment | Best | Driver | Set |\n",
            "| --- | --- | --- | --- |\n",
            "| T1 | 0:04.812 | gamepad | 2026-09-19 |\n",
        )
        assert log.best() == pytest.approx(95.5)
        assert len(log.entries()) == 1

    def test_laps_after_the_fence_are_still_read(self, tmp_path):
        """Lap rows append at the end of the file, so the fence has to close as well as
        open -- otherwise every lap after the first is invisible."""
        path = tmp_path / "l.md"
        self._with_segments(path, "| T1 | 0:04.812 | gamepad | 2026-09-19 |\n")
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write("| 2026-09-19 10:05:00 | 1:30.000 | 0.000 | 1:30.000 | fly |  |\n")
        log = LapLog(path)
        assert len(log.entries()) == 2
        assert log.best() == pytest.approx(90.0)

    def test_a_fresh_document_carries_the_fence(self, tmp_path):
        """So the segment table has somewhere to go without rewriting the prose."""
        log = LapLog(tmp_path / "l.md")
        log.record(95.5)
        text = log.path.read_text(encoding="utf-8")
        assert SEGMENT_TABLE_START in text
        assert text.index(SEGMENT_TABLE_START) < text.index(SEGMENT_TABLE_END)
