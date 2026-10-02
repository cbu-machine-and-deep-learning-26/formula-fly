"""The segment-best table inside the lap-time document (GH-16).

Forza shows a delta as you cross each sector marker, and that is what this is for: driving
through a corner again should say straight away whether it went better than last time. The
best time through each segment lives in ``lap_times.md`` next to the lap table, under the
same rule Payton set for laps -- *the document is the record*. Delete a row and the next
car through sets it again.

**Why a separate module from :mod:`fly_driver.envs.lap`.** The two tables live in one file
but behave differently. Laps are append-only, so a new lap is one line at the end. Segment
bests are one row per segment, replaced in place, so writing one means rewriting the block.
Keeping that in :class:`~fly_driver.envs.lap.LapLog` would have pushed it past the ~400
line guardrail in AGENTS.md §14.3 and mixed two file formats in one class.

The two coexist because the segment table is fenced by
:data:`~fly_driver.envs.lap.SEGMENT_TABLE_START` and
:data:`~fly_driver.envs.lap.SEGMENT_TABLE_END`, and everything outside the fence is
preserved byte for byte on a rewrite. The fence is not decoration: a segment row is
indistinguishable from a lap row to the lap parser, so without it a four-second corner
would quietly become the fastest lap in the book.

**A dirty run cannot set a best.** A time through a corner that was set by cutting it is
not a time through that corner. :meth:`SegmentLog.record` still returns the delta for a
dirty run, so the display can show what was driven, but it does not write it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fly_driver.envs.lap import (
    DEFAULT_LAP_LOG_PATH,
    SEGMENT_TABLE_END,
    SEGMENT_TABLE_START,
    format_lap_time,
    parse_lap_time,
)

__all__ = ["SegmentLog", "SegmentOutcome", "SegmentRecord", "format_delta"]

_TABLE_HEADING = """## Best segments

The fastest clean time through each segment of the circuit. A run with an off-track
excursion in it never sets one of these, however quick it was -- so a segment appears here
the first time it is driven cleanly, which is why the order is not always track order.

Same rule as the lap table: delete a row and the next car through sets it again.

| Segment | Best | Driver | Set |
| --- | --- | --- | --- |"""


def format_delta(seconds: float) -> str:
    """A signed time difference, the way a timing screen shows it.

    Always carries its sign, including for a gain, because ``0.184`` alone reads as a
    time rather than as an improvement.
    """
    return f"{seconds:+.3f}"


@dataclass(frozen=True)
class SegmentRecord:
    """One row of the segment table.

    Args:
        name: The segment's name, from :class:`~fly_driver.envs.segments.Segment`.
        seconds: The best clean time through it.
        driver: Who or what set it.
        when: Date text exactly as written in the file.
    """

    name: str
    seconds: float
    driver: str = ""
    when: str = ""


@dataclass(frozen=True)
class SegmentOutcome:
    """What happened on one run through a segment, ready to be displayed.

    Args:
        name: The segment's name.
        seconds: The time just driven.
        previous: The record before this run, or ``None`` if there was none.
        is_best: Whether this run set a new record. Always ``False`` for a dirty run.
        is_clean: Whether the run was driven without leaving the circuit.
    """

    name: str
    seconds: float
    previous: float | None
    is_best: bool
    is_clean: bool = True

    @property
    def delta(self) -> float | None:
        """Seconds gained or lost against the previous best. ``None`` when there was none.

        Negative is faster, matching every timing screen ever built.
        """
        if self.previous is None:
            return None
        return self.seconds - self.previous

    @property
    def is_improvement(self) -> bool:
        """Whether this run beat the previous best, whether or not it was allowed to count.

        Distinct from :attr:`is_best`: a dirty run can be quicker without setting anything,
        and showing it in green while refusing to record it is the honest display.
        """
        return self.previous is not None and self.seconds < self.previous

    def describe(self) -> str:
        """A one-line summary for a terminal, e.g. ``T4  0:06.221  -0.184  NEW BEST``."""
        parts = [f"{self.name:<12}", format_lap_time(self.seconds)]
        delta = self.delta
        if delta is not None:
            parts.append(f"{format_delta(delta):>8}")
        if self.is_best:
            parts.append("NEW BEST")
        elif not self.is_clean:
            parts.append("off track")
        return "  ".join(parts)


class SegmentLog:
    """The segment-best table: read the records, write one when it is beaten.

    Args:
        path: The markdown document. Shared with
            :class:`~fly_driver.envs.lap.LapLog`, which owns the lap table in the same
            file.
    """

    def __init__(self, path: Path | str = DEFAULT_LAP_LOG_PATH) -> None:
        self.path = Path(path)
        self._cache: dict[str, SegmentRecord] = {}
        self._stamp: tuple[int, int] | None = None
        self._loaded = False

    def _file_stamp(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def records(self) -> dict[str, SegmentRecord]:
        """Every readable row, keyed by segment name, in the order the file lists them.

        Re-read only when the file has changed on disk, which is what lets a hand-edit
        take effect mid-session without re-reading the document fifty times a second.
        """
        stamp = self._file_stamp()
        if self._loaded and stamp == self._stamp:
            return self._cache
        self._stamp = stamp
        self._loaded = True
        self._cache = self._read()
        return self._cache

    def best(self, name: str) -> float | None:
        """The record for one segment, or ``None`` when nothing has set it."""
        found = self.records().get(name)
        return None if found is None else found.seconds

    def _read(self) -> dict[str, SegmentRecord]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return {}
        records: dict[str, SegmentRecord] = {}
        for line in _fenced_lines(text.split("\n")):
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            seconds = parse_lap_time(cells[1])
            if seconds is None or not cells[0]:
                continue  # the heading row and any mangled row land here
            records[cells[0]] = SegmentRecord(
                name=cells[0],
                seconds=seconds,
                driver=cells[2] if len(cells) > 2 else "",
                when=cells[3] if len(cells) > 3 else "",
            )
        return records

    def record(
        self,
        name: str,
        seconds: float,
        *,
        is_clean: bool = True,
        driver: str = "",
        when: datetime | None = None,
    ) -> SegmentOutcome:
        """Offer a run through a segment. Writes only when it is clean and a record.

        The comparison reads the document first, so a row deleted by hand stops counting
        from that moment on.

        Args:
            name: The segment's name.
            seconds: How long the run took.
            is_clean: Whether the car stayed on the circuit for the whole segment. A dirty
                run is scored and returned but never written.
            driver: Who or what drove it.
            when: Date; defaults to today.

        Raises:
            ValueError: If the time is not positive, or the name is blank or contains a
                ``|``, which would split the row into different cells than it was written
                with and silently rename or corrupt the record.
        """
        if seconds <= 0:
            raise ValueError(f"segment time must be positive, got {seconds}")
        if not name.strip() or "|" in name:
            raise ValueError(f"segment name must be non-empty and free of '|', got {name!r}")

        previous = self.best(name)
        is_best = is_clean and (previous is None or seconds < previous)
        outcome = SegmentOutcome(
            name=name,
            seconds=seconds,
            previous=previous,
            is_best=is_best,
            is_clean=is_clean,
        )
        if is_best:
            self._write(
                SegmentRecord(
                    name=name,
                    seconds=seconds,
                    driver=driver,
                    when=(when or datetime.now()).strftime("%Y-%m-%d"),
                )
            )
        return outcome

    def _write(self, record: SegmentRecord) -> None:
        """Replace the fenced block with the records including this one.

        Everything outside the fence is carried over untouched, so lap rows appended after
        the table -- which is where they land, the lap table being append-only -- survive.
        """
        records = dict(self.records())
        # New names land at the end. That is track order for a clean lap and not for a
        # messy one, since a segment only appears once it has been driven without going
        # off -- which the table's own text now says rather than claiming track order.
        records[record.name] = record

        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        lines = text.split("\n") if text else []
        start, end = _find_fence(lines)
        if start is None or end is None:
            start, end = _insert_fence(lines)

        body = [_TABLE_HEADING]
        body += [
            f"| {item.name} | {format_lap_time(item.seconds)} | {item.driver} | {item.when} |"
            for item in records.values()
        ]
        lines[start + 1 : end] = body

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("\n".join(lines))
        self._loaded = False  # force a re-read; our own write changed the file


def _fenced_lines(lines: list[str]) -> list[str]:
    """Just the lines between the fences, or nothing when there is no fence."""
    start, end = _find_fence(lines)
    if start is None or end is None:
        return []
    return lines[start + 1 : end]


def _find_fence(lines: list[str]) -> tuple[int | None, int | None]:
    """Indices of the opening and closing fence lines, if both are present and in order."""
    start = end = None
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == SEGMENT_TABLE_START and start is None:
            start = index
        elif stripped == SEGMENT_TABLE_END and start is not None:
            end = index
            break
    return (start, end) if end is not None else (None, None)


def _insert_fence(lines: list[str]) -> tuple[int, int]:
    """Add an empty fence to a document that has none, and return where it went.

    It goes above the lap table, because lap rows are appended to the end of the file and
    anything below them would be pushed further away with every lap driven. Mutates
    ``lines`` in place.
    """
    where = len(lines)
    for index, line in enumerate(lines):
        if line.strip().startswith("| Date"):
            where = index
            break
    lines[where:where] = [SEGMENT_TABLE_START, SEGMENT_TABLE_END, ""]
    return where, where + 1
