"""The segment-best table (GH-16).

Two tables share one document, and the ways that goes wrong are all quiet. A segment row
read as a lap gives a four-second lap record. A rewrite that does not preserve what is
outside the fence eats the lap table. A best that is cached rather than re-read keeps
showing a time the file no longer contains. None of those raise, so each is tested.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from fly_driver.envs.lap import SEGMENT_TABLE_END, SEGMENT_TABLE_START, LapLog
from fly_driver.envs.segment_log import SegmentLog, SegmentOutcome, format_delta


@pytest.fixture
def log(tmp_path) -> SegmentLog:
    return SegmentLog(tmp_path / "lap_times.md")


class TestRecording:
    def test_the_first_run_through_sets_the_record(self, log):
        outcome = log.record("T1", 4.812, driver="gamepad")
        assert outcome.is_best
        assert outcome.previous is None
        assert outcome.delta is None
        assert log.best("T1") == pytest.approx(4.812)

    def test_a_faster_run_replaces_it(self, log):
        log.record("T1", 4.812)
        outcome = log.record("T1", 4.628)
        assert outcome.is_best
        assert outcome.delta == pytest.approx(-0.184)
        assert log.best("T1") == pytest.approx(4.628)

    def test_a_slower_run_leaves_it_alone(self, log):
        log.record("T1", 4.628)
        outcome = log.record("T1", 5.001)
        assert not outcome.is_best
        assert outcome.delta == pytest.approx(0.373)
        assert log.best("T1") == pytest.approx(4.628)

    def test_only_one_row_per_segment_however_many_laps(self, log):
        for seconds in (5.0, 4.8, 4.9, 4.7, 5.2):
            log.record("T1", seconds)
        assert list(log.records()) == ["T1"]
        assert log.best("T1") == pytest.approx(4.7)

    def test_segments_keep_the_order_they_were_first_driven(self, log):
        """Which is track order, since that is the order a car meets them. A sorted table
        would put T10 between T1 and T2 and read as nonsense against the circuit."""
        for name in ("S1", "T1", "S2", "T2", "S10", "T10"):
            log.record(name, 5.0)
        assert list(log.records()) == ["S1", "T1", "S2", "T2", "S10", "T10"]

    def test_the_driver_and_date_are_kept(self, log):
        log.record("T1", 4.812, driver="gamepad (PS5 Controller)", when=datetime(2026, 9, 19))
        found = log.records()["T1"]
        assert found.driver == "gamepad (PS5 Controller)"
        assert found.when == "2026-09-19"


class TestDirtyRuns:
    """A time set by cutting the corner is not a time through the corner."""

    def test_a_dirty_run_never_sets_a_record(self, log):
        log.record("T1", 5.000)
        outcome = log.record("T1", 3.000, is_clean=False)
        assert not outcome.is_best
        assert log.best("T1") == pytest.approx(5.000)

    def test_a_dirty_run_on_a_virgin_segment_sets_nothing(self, log):
        assert not log.record("T1", 3.0, is_clean=False).is_best
        assert log.best("T1") is None
        assert log.records() == {}

    def test_a_dirty_run_still_reports_its_delta(self, log):
        """The display should show what was driven even though it does not count --
        hiding it would look like the timer had failed."""
        log.record("T1", 5.000)
        outcome = log.record("T1", 3.000, is_clean=False)
        assert outcome.delta == pytest.approx(-2.0)
        assert outcome.is_improvement, "quicker, just not eligible"
        assert not outcome.is_clean


class TestTheDocument:
    def test_the_lap_table_survives_a_segment_write(self, tmp_path):
        """The rewrite replaces the fenced block only. Losing the lap table to it would
        take Payton's own lap with it."""
        path = tmp_path / "lap_times.md"
        laps = LapLog(path)
        laps.record(112.229, driver="gamepad")
        SegmentLog(path).record("T1", 4.812)
        assert laps.best() == pytest.approx(112.229)
        assert "# Lap times" in path.read_text(encoding="utf-8")

    def test_a_lap_appended_after_the_table_is_still_read(self, tmp_path):
        """Lap rows land at the end of the file, below the segment table. Both tables have
        to keep working in that layout, because that is the layout driving produces."""
        path = tmp_path / "lap_times.md"
        laps, segments = LapLog(path), SegmentLog(path)
        laps.record(112.229, driver="gamepad")
        segments.record("T1", 4.812)
        laps.record(95.5, driver="fly")
        assert len(laps.entries()) == 2
        assert laps.best() == pytest.approx(95.5)
        assert segments.best("T1") == pytest.approx(4.812)

    def test_a_segment_row_is_never_read_as_a_lap(self, tmp_path):
        path = tmp_path / "lap_times.md"
        laps = LapLog(path)
        laps.record(112.229)
        SegmentLog(path).record("T1", 4.812)
        assert laps.best() == pytest.approx(112.229), "a 4.8 second lap would be the give-away"

    def test_it_writes_into_a_document_that_has_no_fence(self, tmp_path):
        """Record books written before segments existed have no fence, and Payton has one
        of those with a real lap in it."""
        path = tmp_path / "lap_times.md"
        path.write_text(
            "# Lap times\n\n"
            "| Date | Time | Driver | Note |\n"
            "| --- | --- | --- | --- |\n"
            "| 2026-09-18 22:14:03 | 1:52.229 | gamepad | new best |\n",
            encoding="utf-8",
        )
        SegmentLog(path).record("T1", 4.812)
        text = path.read_text(encoding="utf-8")
        assert SEGMENT_TABLE_START in text and SEGMENT_TABLE_END in text
        assert LapLog(path).best() == pytest.approx(112.229)

    def test_the_fence_goes_above_the_lap_table(self, tmp_path):
        """Below it, every lap driven would push the segment table further down the file."""
        path = tmp_path / "lap_times.md"
        LapLog(path).record(112.229)
        SegmentLog(path).record("T1", 4.812)
        text = path.read_text(encoding="utf-8")
        # Anchor on the lap table's header rather than on a date: a segment row carries a
        # date of its own, so "| 2026-" matches inside the fence and proves nothing.
        assert text.index(SEGMENT_TABLE_END) < text.index("| Date |")

    def test_it_starts_a_document_from_nothing(self, log):
        log.record("T1", 4.812)
        assert log.path.exists()
        assert log.best("T1") == pytest.approx(4.812)

    def test_the_table_says_what_it_is(self, log):
        """The file is meant to be read by a person, so the rules live next to the rows."""
        log.record("T1", 4.812)
        text = log.path.read_text(encoding="utf-8")
        assert "## Best segments" in text
        assert "| Segment | Best | Driver | Set |" in text


class TestTheDocumentIsTheRecord:
    """Payton's rule for laps, applied to segments: delete a row and it is gone."""

    def test_deleting_a_row_lets_the_next_run_set_it_again(self, log):
        log.record("T1", 4.628)
        text = log.path.read_text(encoding="utf-8")
        kept = [line for line in text.split("\n") if not line.startswith("| T1 |")]
        log.path.write_text("\n".join(kept), encoding="utf-8")
        outcome = log.record("T1", 5.500)
        assert outcome.is_best and outcome.previous is None
        assert log.best("T1") == pytest.approx(5.500)

    def test_a_hand_edited_time_is_believed(self, log):
        log.record("T1", 4.628)
        text = log.path.read_text(encoding="utf-8").replace("0:04.628", "0:03.000")
        log.path.write_text(text, encoding="utf-8")
        assert log.best("T1") == pytest.approx(3.0)
        assert not log.record("T1", 4.000).is_best

    def test_a_mangled_row_does_not_stop_the_rest_being_read(self, log):
        log.record("T1", 4.628)
        log.record("T2", 6.100)
        text = log.path.read_text(encoding="utf-8").replace("| T1 | 0:04.628", "| T1 | banana")
        log.path.write_text(text, encoding="utf-8")
        assert log.best("T1") is None
        assert log.best("T2") == pytest.approx(6.100)

    def test_a_missing_document_has_no_records(self, tmp_path):
        log = SegmentLog(tmp_path / "nothing.md")
        assert log.records() == {}
        assert log.best("T1") is None


class TestValidation:
    @pytest.mark.parametrize("seconds", [0.0, -1.0])
    def test_the_time_must_be_positive(self, log, seconds):
        with pytest.raises(ValueError, match="must be positive"):
            log.record("T1", seconds)

    @pytest.mark.parametrize("name", ["", "   ", "T1 | T2"])
    def test_the_name_must_fit_in_one_cell(self, log, name):
        """A pipe in the name would split the row into different cells than it was written
        with, so the record would come back under a different name or not at all."""
        with pytest.raises(ValueError, match=r"free of"):
            log.record(name, 4.0)


class TestDisplay:
    def test_a_gain_carries_its_sign(self):
        assert format_delta(-0.184) == "-0.184"
        assert format_delta(0.373) == "+0.373"

    def test_a_new_best_says_so(self):
        outcome = SegmentOutcome(name="T4", seconds=6.221, previous=6.405, is_best=True)
        line = outcome.describe()
        assert "T4" in line and "0:06.221" in line and "-0.184" in line and "NEW BEST" in line

    def test_a_first_run_has_no_delta_to_show(self):
        outcome = SegmentOutcome(name="T4", seconds=6.221, previous=None, is_best=True)
        assert "+" not in outcome.describe() and "-" not in outcome.describe()

    def test_a_dirty_run_is_labelled(self):
        outcome = SegmentOutcome(
            name="T4", seconds=3.0, previous=6.405, is_best=False, is_clean=False
        )
        assert "off track" in outcome.describe()
        assert "NEW BEST" not in outcome.describe()
