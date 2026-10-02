"""Lap timing and the lap-time record book (GH-16).

Two pieces. :class:`LapTimer` turns a stream of positions into completed lap times, and
:class:`LapLog` keeps them in a markdown document on disk.

The document is deliberately the only source of truth for the best lap. Payton's reason:
*"if i put down the best lap i need to go in and delete it so its only the flys best lap."*
So nothing is cached across a lap -- :meth:`LapLog.best` re-reads the file whenever it has
changed on disk, which means deleting a row updates the best time immediately, even while
the simulator is running. Anything that remembered the best in memory would keep showing a
time that no longer exists in the record.

Parsing is deliberately forgiving for the same reason. The file is meant to be edited by
hand, so a row that cannot be understood is skipped rather than raising, and the fastest
row that remains is the record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fly_driver.envs.centerline import Centerline

__all__ = [
    "DEFAULT_LAP_LOG_PATH",
    "SEGMENT_TABLE_END",
    "SEGMENT_TABLE_START",
    "LapEntry",
    "LapLog",
    "LapTimer",
    "format_lap_time",
    "parse_lap_time",
]

#: Where the record book lives. Repo root, so it is easy to find and edit by hand.
DEFAULT_LAP_LOG_PATH = Path(__file__).resolve().parents[2] / "lap_times.md"

#: Fences around the segment-best table, which :mod:`fly_driver.envs.segment_log` owns and
#: rewrites in place. They are load-bearing rather than decorative: a segment row looks
#: exactly like a lap row to :meth:`LapLog._read`, so without somewhere explicit to stop,
#: a four-second corner would quietly become the fastest lap in the book.
SEGMENT_TABLE_START = "<!-- segments -->"
SEGMENT_TABLE_END = "<!-- /segments -->"

_HEADER = f"""# Lap times

The record book for the MuJoCo practice track (Silverstone). Every completed lap is
appended here by `scripts/drive.py`.

**The best lap is simply the fastest row in the table below.** Nothing is cached: the file
is re-read whenever it changes, so if you delete the rows you drove yourself, the best time
updates straight away -- even while the simulator is running. That is the whole point of
keeping this in a document rather than in the code.

A lap only counts if it was driven the whole way round; cutting back across the start line
does not register one.

`Lap` is the time that counts. It is `Raw` plus `Penalty`, where the penalty is the seconds
earned by going off the circuit -- see `fly_driver/envs/penalty.py` for what an excursion
costs. A lap driven entirely on the track has a penalty of zero and a `Lap` equal to `Raw`.

Times are `M:SS.mmm`. Rows may be deleted or reordered freely, but keep the table header.

{SEGMENT_TABLE_START}
{SEGMENT_TABLE_END}

| Date | Lap | Penalty | Raw | Driver | Note |
| --- | --- | --- | --- | --- | --- |
"""

_TIME_PATTERN = re.compile(r"^\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*$")


def format_lap_time(seconds: float) -> str:
    """Seconds to ``M:SS.mmm``, the way a timing screen shows it.

    Raises:
        ValueError: If ``seconds`` is negative.
    """
    if seconds < 0:
        raise ValueError(f"lap time cannot be negative, got {seconds}")
    minutes, remainder = divmod(round(seconds * 1000), 60_000)
    return f"{minutes:d}:{remainder / 1000:06.3f}"


def parse_lap_time(text: str) -> float | None:
    """``M:SS.mmm`` or bare seconds to a float. ``None`` when it cannot be read.

    Returning ``None`` rather than raising is the point: the file is hand-edited, and one
    mangled row must not stop the rest of the record book being readable.
    """
    match = _TIME_PATTERN.match(text)
    if match is None:
        return None
    minutes, seconds = match.groups()
    value = float(seconds) + (60.0 * int(minutes) if minutes else 0.0)
    return value if value > 0 else None


def _parse_seconds(text: str) -> float:
    """A bare number of seconds, or 0.0 when the cell cannot be read.

    Forgiving for the same reason :func:`parse_lap_time` is: a hand-edited penalty cell
    reading ``--`` should cost that row its penalty, not the whole document.
    """
    try:
        return max(0.0, float(text.strip().lstrip("+")))
    except ValueError:
        return 0.0


def _entry_from(cells: list[str], seconds: float) -> LapEntry:
    """Build a row from its cells, tolerating both layouts of the table.

    Rows written before penalties existed have four columns and their time is the whole
    story. Newer rows carry the penalty and the raw time between the lap time and the
    driver. Column 1 is the lap time in both, which is what keeps an old record book
    comparable with a new one rather than quietly unreadable.
    """
    has_penalty = len(cells) >= 5
    if has_penalty:
        return LapEntry(
            seconds=seconds,
            penalty_seconds=_parse_seconds(cells[2]),
            when=cells[0],
            driver=cells[4],
            note=cells[5] if len(cells) > 5 else "",
        )
    return LapEntry(
        seconds=seconds,
        when=cells[0],
        driver=cells[2] if len(cells) > 2 else "",
        note=cells[3] if len(cells) > 3 else "",
    )


def _fraction_of(part: float, whole: float) -> float:
    """``part / whole`` clamped to [0, 1], for interpolating back within one step."""
    if whole <= 0:
        return 0.0
    return min(1.0, max(0.0, part / whole))


@dataclass(frozen=True)
class LapEntry:
    """One row of the record book.

    Args:
        seconds: The lap time that counts -- raw plus penalty. This is what is compared,
            so a lap with a five-second cut in it does not hold the record over a clean
            lap that was four seconds slower on the stopwatch.
        penalty_seconds: Of that, how much was earned by leaving the circuit. Zero for a
            clean lap, and zero for rows written before penalties existed.
        when: Timestamp text exactly as written in the file.
        driver: Who or what drove it.
        note: Free text; ``new best`` is written when the lap beat the record at the time.
    """

    seconds: float
    penalty_seconds: float = 0.0
    when: str = ""
    driver: str = ""
    note: str = ""

    @property
    def raw_seconds(self) -> float:
        """The stopwatch time, before the penalty was added."""
        return self.seconds - self.penalty_seconds

    @property
    def is_clean(self) -> bool:
        """Whether the lap was driven without ever leaving the circuit."""
        return self.penalty_seconds == 0.0


class LapTimer:
    """Turns arclength along the centerline into completed lap times.

    A lap is counted when a full track length of *forward* progress has been accumulated,
    **and** the car is on the circuit at the moment it completes. Both halves matter.

    Progress rather than a line crossing: reversing over the line and crossing it again
    would otherwise hand out a lap for a few metres of driving, and the reward plumbing in
    GH-17 leans on the same signal.

    On the circuit rather than anywhere: a car in the grass projects onto the centerline
    unreliably, because the circuit loops back on itself and a point in the infield can be
    nearest to a completely different part of the lap. Measured at Silverstone, a car 40 m
    off the road at 5500 m projects 56 m further round, and 80 m off projects 82 m further
    -- both past ``max_step_m``. That used to reset the clock mid-lap, which looks exactly
    like the lap having ended early, and is what Payton hit running wide near the end.

    Args:
        centerline: The track. Supplies the lap length and the seam-safe progress delta.
        min_lap_seconds: A lap faster than this is treated as a glitch and dropped. The
            practice track's outright record is about 87 s, so anything under 20 s did not
            happen.
        max_step_m: Progress larger than this in a single update is not driving -- a
            reset, a car dropped back onto the track, or the projection of a car that is
            off the circuit. The distance is skipped. The clock is deliberately **not**
            touched: the driver is still on the lap they started, and restarting it under
            them is worse than the jump it was guarding against. At 350 km/h a 50 Hz step
            covers about 2 m.
        start_on_crossing: Wait for the car to cross the start line before timing anything.
            This is what makes the out lap work: the car is parked behind the line, drives
            up to it, and lap one is timed from the crossing rather than from wherever the
            simulator happened to boot. Set ``False`` to time from :meth:`reset` instead,
            which is what a training environment starting on the line wants.
    """

    def __init__(
        self,
        centerline: Centerline,
        *,
        min_lap_seconds: float = 20.0,
        max_step_m: float = 50.0,
        start_on_crossing: bool = True,
    ) -> None:
        if min_lap_seconds < 0:
            raise ValueError(f"min_lap_seconds must be non-negative, got {min_lap_seconds}")
        if max_step_m <= 0:
            raise ValueError(f"max_step_m must be positive, got {max_step_m}")
        self.centerline = centerline
        self.min_lap_seconds = float(min_lap_seconds)
        self.max_step_m = float(max_step_m)
        self.start_on_crossing = bool(start_on_crossing)
        self._previous_s: float | None = None
        self._previous_time = 0.0
        self._progress = 0.0
        self._lap_started_at = 0.0
        self._timing = not self.start_on_crossing
        self.last_lap: float | None = None
        self.completed = 0

    def reset(self, arclength: float = 0.0, now: float = 0.0) -> None:
        """Start a fresh out lap from here. Call at an episode boundary."""
        self._previous_s = float(arclength)
        self._previous_time = float(now)
        self._progress = 0.0
        self._lap_started_at = float(now)
        self._timing = not self.start_on_crossing

    @property
    def timing(self) -> bool:
        """False while the car is still on its way to the line for the first time."""
        return self._timing

    def current_lap_time(self, now: float) -> float:
        """Seconds since the current lap began, or 0 while still on the out lap."""
        if not self._timing:
            return 0.0
        return max(0.0, float(now) - self._lap_started_at)

    @property
    def lap_fraction(self) -> float:
        """How far round the current lap the car is, 0 to 1."""
        if not self._timing:
            return 0.0
        return min(1.0, max(0.0, self._progress / self.centerline.length))

    def update(self, arclength: float, now: float, on_track: bool = True) -> float | None:
        """Feed one sample. Returns the lap time when this sample completed a lap.

        Args:
            arclength: Distance along the centerline, from :meth:`Centerline.project`.
            now: Simulation time in seconds.
            on_track: Whether the car is within the track limits right now. A lap will not
                complete while this is false; the completion is held until the car is back
                on the circuit. Pass the same signal the penalty is charged on, so the two
                rules agree about where the circuit ends.
        """
        arclength, now = float(arclength), float(now)
        if self._previous_s is None:
            self.reset(arclength, now)
            return None

        previous_s, previous_time = self._previous_s, self._previous_time
        delta = self.centerline.progress_delta(previous_s, arclength)
        self._previous_s, self._previous_time = arclength, now
        step_seconds = now - previous_time

        if abs(delta) > self.max_step_m:
            # A jump this big is not driving: a reset, a car dropped back on, or -- far
            # more often -- the projection of a car that is off the circuit. The distance
            # is skipped, so a shortcut across the infield cannot hand out a free lap.
            # The clock is left alone. Zeroing it here restarted the lap under the driver
            # every time they ran wide near the end, which reads as the lap ending early.
            # A genuine episode boundary calls reset() and does not rely on this.
            return None

        if not self._timing:
            # On the out lap, watching for the start line. Crossing it forwards is the
            # one case where arclength goes *down* while progress goes up.
            if delta > 0 and arclength < previous_s:
                self._lap_started_at = now - _fraction_of(arclength, delta) * step_seconds
                self._progress = arclength
                self._timing = True
            return None

        self._progress += delta
        if self._progress < self.centerline.length:
            return None

        if not on_track:
            # A lap ends at the line, on the circuit -- not by drifting past it through
            # the grass. Hold the completion exactly at a lap's worth of progress rather
            # than banking what is driven while waiting, or a long excursion past the line
            # would start the next lap already part-driven.
            self._progress = self.centerline.length
            return None

        # Interpolate back to the instant the line was actually crossed. A whole control
        # step is 20 ms at 50 Hz, which is a tenth of the gap between a good lap and a
        # great one, and it would be charged to every lap in the record book.
        overshoot = self._progress - self.centerline.length
        crossed_at = now - _fraction_of(overshoot, delta) * step_seconds
        lap_time = crossed_at - self._lap_started_at
        self._progress = overshoot
        self._lap_started_at = crossed_at
        if lap_time < self.min_lap_seconds:
            return None
        self.last_lap = lap_time
        self.completed += 1
        return lap_time


class LapLog:
    """The lap-time document: append completed laps, read the record back out.

    Args:
        path: The markdown file. Created with a header on first write.
    """

    def __init__(self, path: Path | str = DEFAULT_LAP_LOG_PATH) -> None:
        self.path = Path(path)
        self._cache: list[LapEntry] = []
        self._stamp: tuple[int, int] | None = None
        self._loaded = False

    def _file_stamp(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except OSError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def entries(self) -> list[LapEntry]:
        """Every readable row, re-reading the file only when it has changed on disk.

        The mtime-and-size check is what lets a hand-edit take effect mid-session without
        re-reading the file fifty times a second.
        """
        stamp = self._file_stamp()
        if self._loaded and stamp == self._stamp:
            return self._cache
        self._stamp = stamp
        self._loaded = True
        self._cache = self._read()
        return self._cache

    def _read(self) -> list[LapEntry]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return []
        entries: list[LapEntry] = []
        in_segments = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            # The segment table is rows of times in this same document, and every one of
            # them would read as a very fast lap. Skip the fenced block outright.
            if line == SEGMENT_TABLE_START:
                in_segments = True
                continue
            if line == SEGMENT_TABLE_END:
                in_segments = False
                continue
            if in_segments or not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            seconds = parse_lap_time(cells[1])
            if seconds is None:
                continue  # the header row and any mangled row land here
            entries.append(_entry_from(cells, seconds))
        return entries

    def best(self) -> float | None:
        """The fastest lap in the document, or ``None`` when there is none."""
        entries = self.entries()
        return min((entry.seconds for entry in entries), default=None)

    def record(
        self,
        seconds: float,
        *,
        penalty_seconds: float = 0.0,
        driver: str = "",
        when: datetime | None = None,
        note: str = "",
    ) -> bool:
        """Append a completed lap. Returns ``True`` if it beat everything on file.

        The comparison reads the document first, so a record deleted by hand stops
        counting from that moment on.

        Args:
            seconds: The stopwatch time, before any penalty.
            penalty_seconds: Seconds earned by leaving the circuit, from
                :meth:`~fly_driver.envs.penalty.OffTrackPenalty.finish_lap`. Added to
                ``seconds`` to give the time that is recorded and compared. Taking the two
                separately rather than pre-added is what stops a caller adding the penalty
                twice, or forgetting it and filing a cut lap as though it were clean.
            driver: Who or what drove it.
            when: Timestamp; defaults to now.
            note: Free text. ``new best`` is written when the lap beat the record.

        Raises:
            ValueError: If the raw time is not positive, or the penalty is negative.
        """
        if seconds <= 0:
            raise ValueError(f"lap time must be positive, got {seconds}")
        if penalty_seconds < 0:
            raise ValueError(f"penalty must be non-negative, got {penalty_seconds}")
        lap_seconds = seconds + penalty_seconds
        previous = self.best()
        is_best = previous is None or lap_seconds < previous
        if not note:
            note = "new best" if is_best else ""

        stamp = (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
        row = (
            f"| {stamp} | {format_lap_time(lap_seconds)} | {penalty_seconds:.3f} "
            f"| {format_lap_time(seconds)} | {driver} | {note} |\n"
        )
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(_HEADER, encoding="utf-8")
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(row)
        self._loaded = False  # force a re-read; our own write changed the file
        return is_best
