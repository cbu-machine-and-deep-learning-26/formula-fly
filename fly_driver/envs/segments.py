"""Splitting the circuit into corners and straights, and timing them (GH-16).

A lap time is one number for six kilometres of driving. It says a lap was slow; it never
says *where*. Forza Motorsport solves that with segment scores over "a string of corners",
and this is the same idea reduced to what is actually measurable here: the track is cut at
curvature transitions, each piece is timed, and the best time through each is kept. A
per-segment time is also a far denser signal than a lap time, which is why GH-17's reward
shaping and GH-32's human comparison can both use it and a 1-10 score could not.

**The segments come from the geometry, not from a table someone typed.** Curvature along
the centerline, smoothed, tells you where a corner starts and where it straightens out.
That matters because the centerline is data (`fly_driver/envs/data/`) and a hand-written
list of corner positions would silently stop matching it the day the data changes.

Three details separate a working segmentation from a useless one, and all three were found
by running it on Silverstone:

* **Hysteresis.** A single threshold chops one corner into three whenever the curvature
  wobbles across it. Entering a corner needs a tighter radius than leaving one does.
* **A minimum length.** The raw classification finds runs 10-35 m long at a 240 m radius.
  Those are kinks in the surveyed line, not corners, and each would become a segment
  nobody could drive to.
* **Merging complexes.** Corners separated by a short straight are one thing to a driver --
  Maggotts-Becketts-Chapel is a single decision, not three -- and Forza's "string of
  corners" means the same.

A **sign change in curvature always starts a new segment**: left-then-right is two corners
however smoothly one runs into the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from fly_driver.envs.centerline import Centerline

__all__ = [
    "DEFAULT_SEGMENT_SETTINGS",
    "Segment",
    "SegmentSettings",
    "SegmentTimer",
    "SegmentTiming",
    "find_segments",
]


@dataclass(frozen=True)
class SegmentSettings:
    """How finely to cut the circuit up.

    The defaults were chosen against Silverstone: they produce corner runs in the region of
    its 18 numbered corners, and segments long enough that a driver can tell one from the
    next. A different circuit may want different numbers, which is why they are arguments.

    Args:
        spacing_m: Resampling interval before curvature is measured.
        smoothing_m: Window for smoothing curvature. The surveyed centerline is noisy at
            the metre scale and raw curvature from it is unusable.
        enter_radius_m: A bend tighter than this radius starts a corner.
        exit_radius_m: A corner continues until the radius opens past this. Larger than
            ``enter_radius_m`` on purpose -- that gap is the hysteresis.
        min_corner_m: Corner runs shorter than this are kinks, and become straight.
        min_straight_m: A straight shorter than this is not a rest, and gets absorbed.
        link_m: Corners separated by less than this much straight become one complex.

    Raises:
        ValueError: If a length is not positive, or if ``exit_radius_m`` is not greater
            than ``enter_radius_m`` (which would be hysteresis pointing the wrong way).
    """

    spacing_m: float = 5.0
    smoothing_m: float = 40.0
    enter_radius_m: float = 250.0
    exit_radius_m: float = 400.0
    min_corner_m: float = 55.0
    min_straight_m: float = 70.0
    link_m: float = 90.0

    def __post_init__(self) -> None:
        for name in (
            "spacing_m",
            "smoothing_m",
            "enter_radius_m",
            "exit_radius_m",
            "min_corner_m",
            "min_straight_m",
            "link_m",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.exit_radius_m <= self.enter_radius_m:
            raise ValueError(
                f"exit_radius_m={self.exit_radius_m} must exceed "
                f"enter_radius_m={self.enter_radius_m}; a corner has to be harder to enter "
                "than to leave or the hysteresis does nothing"
            )


#: What the circuit gets cut with unless a caller says otherwise.
DEFAULT_SEGMENT_SETTINGS = SegmentSettings()


@dataclass(frozen=True)
class Segment:
    """One stretch of circuit, timed as a unit.

    Args:
        index: Position in the lap, counting from the start line.
        name: Short label for the panel and the lap document -- ``T3``, ``S4``. Numbered
            from the geometry rather than named after real corners: our start/finish line
            need not sit where the real circuit's does, and a confidently wrong corner name
            is worse than a number.
        is_corner: Whether this is a corner (or a complex of them) rather than a straight.
        turns_left: For a corner, which way. ``None`` on a straight.
        start_m: Arclength where it begins.
        end_m: Arclength where it ends. Always greater than ``start_m``: the start line is
            forced to be a boundary, so no segment wraps.
        min_radius_m: Tightest radius inside it. ``inf`` on a straight.
    """

    index: int
    name: str
    is_corner: bool
    turns_left: bool | None
    start_m: float
    end_m: float
    min_radius_m: float

    @property
    def length_m(self) -> float:
        """How long this segment is."""
        return self.end_m - self.start_m

    def __str__(self) -> str:
        kind = "corner" if self.is_corner else "straight"
        return f"{self.name} ({kind}, {self.length_m:.0f} m from {self.start_m:.0f} m)"


def signed_curvature(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Signed curvature at each point of a closed polyline, in 1/metres.

    Positive turns left, matching :meth:`~fly_driver.envs.centerline.Centerline.normal`'s
    convention so the sign means the same thing everywhere in this package. Computed from
    central differences rather than consecutive segments, so the value at a corner bisects
    it instead of jumping between the two sides.
    """
    first = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    second = np.roll(points, -1, axis=0) - 2.0 * points + np.roll(points, 1, axis=0)
    speed = np.linalg.norm(first, axis=1) / 2.0
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    return cross / np.maximum(speed**3, 1e-9) / 4.0


def _smooth_cyclic(values: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Moving average around a closed loop, so the start line is not a discontinuity."""
    window = max(1, window | 1)  # odd, so the window is centred
    padded = np.concatenate([values[-window:], values, values[:window]])
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="same")[window:-window]


def _classify(
    curvature: npt.NDArray[np.float64], settings: SegmentSettings
) -> npt.NDArray[np.bool_]:
    """Corner or straight at each point, with hysteresis.

    A single threshold chops one corner into several whenever the curvature wobbles across
    it. Entering needs ``enter_radius_m``; staying in only needs ``exit_radius_m``.
    """
    enter = 1.0 / settings.enter_radius_m
    exit_ = 1.0 / settings.exit_radius_m
    magnitude = np.abs(curvature)
    corner = np.zeros(len(curvature), dtype=bool)
    # Two passes around the loop: the first decides the state at index 0 without assuming
    # it, the second is the one that is kept.
    inside = bool(magnitude[0] > enter)
    for _ in range(2):
        for index, value in enumerate(magnitude):
            inside = value > enter if not inside else value > exit_
            corner[index] = inside
    return corner


def _runs(corner: npt.NDArray[np.bool_], side: npt.NDArray[np.int8]) -> list[list[int]]:
    """Index runs, broken where the kind changes or a corner reverses direction.

    Walking from index 0 rather than around the loop is deliberate: it forces a boundary at
    the start line, which is where a lap is timed from anyway and which saves every
    consumer from wrap-around arithmetic.
    """
    groups: list[list[int]] = [[0]]
    for index in range(1, len(corner)):
        reversed_direction = corner[index] and side[index] != side[index - 1]
        if corner[index] != corner[index - 1] or reversed_direction:
            groups.append([])
        groups[-1].append(index)
    return groups


def find_segments(centerline: Centerline, settings: SegmentSettings | None = None) -> list[Segment]:
    """Cut a circuit into corners and straights.

    Args:
        centerline: The circuit.
        settings: Thresholds. Defaults to :data:`DEFAULT_SEGMENT_SETTINGS`.

    Returns:
        Segments in lap order, tiling ``[0, length)`` with no gaps and no overlaps.

    Raises:
        ValueError: If the settings are inconsistent, or the circuit is too short to cut.
    """
    settings = settings or DEFAULT_SEGMENT_SETTINGS
    resampled = centerline.resample(settings.spacing_m)
    points = np.asarray(resampled.points, dtype=float)
    if len(points) < 8:
        raise ValueError(
            f"{len(points)} points at {settings.spacing_m} m spacing is too few to find "
            "segments; use a finer spacing or a longer circuit"
        )

    window = max(1, int(round(settings.smoothing_m / settings.spacing_m)))
    curvature = _smooth_cyclic(signed_curvature(points), window)
    corner = _classify(curvature, settings)
    side = np.where(curvature > 0, 1, -1).astype(np.int8)

    groups = _runs(corner, side)
    kinds = [bool(corner[group[0]]) for group in groups]
    groups, kinds = _drop_short_corners(groups, kinds, settings)
    groups, kinds = _merge_complexes(groups, kinds, settings)
    groups, kinds = _absorb_short_straights(groups, kinds, settings)
    return _build(groups, kinds, side, curvature, centerline.length, settings)


def _rebuild(groups: list[list[int]], kinds: list[bool]) -> tuple[list[list[int]], list[bool]]:
    """Merge neighbouring groups that share a kind, after a reclassification."""
    merged: list[list[int]] = []
    merged_kinds: list[bool] = []
    for group, kind in zip(groups, kinds, strict=True):
        if merged and merged_kinds[-1] == kind:
            merged[-1] = merged[-1] + group
        else:
            merged.append(list(group))
            merged_kinds.append(kind)
    return merged, merged_kinds


def _drop_short_corners(
    groups: list[list[int]], kinds: list[bool], settings: SegmentSettings
) -> tuple[list[list[int]], list[bool]]:
    """A 15 m bend at a 240 m radius is a kink in the survey, not a corner."""
    minimum = max(1, int(round(settings.min_corner_m / settings.spacing_m)))
    demoted = [kind and len(group) >= minimum for group, kind in zip(groups, kinds, strict=True)]
    return _rebuild(groups, demoted)


def _merge_complexes(
    groups: list[list[int]], kinds: list[bool], settings: SegmentSettings
) -> tuple[list[list[int]], list[bool]]:
    """Corners split by a short straight are one complex, as a driver experiences them."""
    link = max(1, int(round(settings.link_m / settings.spacing_m)))
    promoted = list(kinds)
    for index in range(1, len(groups) - 1):
        bridges_two_corners = kinds[index - 1] and kinds[index + 1]
        if not kinds[index] and len(groups[index]) <= link and bridges_two_corners:
            promoted[index] = True
    return _rebuild(groups, promoted)


def _absorb_short_straights(
    groups: list[list[int]], kinds: list[bool], settings: SegmentSettings
) -> tuple[list[list[int]], list[bool]]:
    """A 30 m straight is not a rest; fold it into the corner beside it."""
    minimum = max(1, int(round(settings.min_straight_m / settings.spacing_m)))
    promoted = list(kinds)
    for index, group in enumerate(groups):
        if kinds[index] or len(group) >= minimum or len(groups) < 3:
            continue
        beside_a_corner = (index > 0 and kinds[index - 1]) or (
            index + 1 < len(groups) and kinds[index + 1]
        )
        if beside_a_corner:
            promoted[index] = True
    return _rebuild(groups, promoted)


def _build(
    groups: list[list[int]],
    kinds: list[bool],
    side: npt.NDArray[np.int8],
    curvature: npt.NDArray[np.float64],
    length_m: float,
    settings: SegmentSettings,
) -> list[Segment]:
    """Turn index groups into named segments covering the whole lap."""
    segments: list[Segment] = []
    corners_seen = 0
    straights_seen = 0
    for index, (group, is_corner) in enumerate(zip(groups, kinds, strict=True)):
        start_m = group[0] * settings.spacing_m
        end_m = (group[-1] + 1) * settings.spacing_m
        if index == len(groups) - 1:
            end_m = length_m  # the last group runs to the line, whatever rounding says
        if is_corner:
            corners_seen += 1
            name = f"T{corners_seen}"
            turns_left: bool | None = bool(side[group][0] > 0)
            radius = 1.0 / max(float(np.abs(curvature[group]).max()), 1e-9)
        else:
            straights_seen += 1
            name = f"S{straights_seen}"
            turns_left = None
            radius = float("inf")
        segments.append(
            Segment(
                index=index,
                name=name,
                is_corner=is_corner,
                turns_left=turns_left,
                start_m=start_m,
                end_m=min(end_m, length_m),
                min_radius_m=radius,
            )
        )
    return segments


@dataclass(frozen=True)
class SegmentTiming:
    """One completed run through one segment.

    Args:
        segment: Which segment was driven.
        seconds: How long it took.
        went_off_track: Whether the car left the circuit anywhere inside it. A time set
            by cutting the corner is not a time, so a dirty run is reported and never
            becomes a record -- the same call Forza makes when a penalty drops a segment
            score to its floor.
    """

    segment: Segment
    seconds: float
    went_off_track: bool

    @property
    def is_clean(self) -> bool:
        """Whether this run is eligible to set a record."""
        return not self.went_off_track


class SegmentTimer:
    """Times each segment as the car passes through it.

    Driven from arclength, which :class:`~fly_driver.envs.practice_track.PracticeTrack`
    already computes every step for the lap timer and the track-limits rule, so this adds
    no new geometry.

    Only *forward* entry into the following segment completes one. Rolling backwards over
    a boundary, or being put back on the grid, re-anchors instead -- otherwise reversing
    across a line would hand out a segment time for a few metres of driving, which is the
    same trap :class:`~fly_driver.envs.lap.LapTimer` avoids for whole laps.

    Args:
        segments: The circuit, from :func:`find_segments`.

    Raises:
        ValueError: If given no segments.
    """

    def __init__(self, segments: list[Segment]) -> None:
        if not segments:
            raise ValueError("a timer needs at least one segment")
        self.segments = segments
        self._starts = np.array([segment.start_m for segment in segments], dtype=float)
        self._current: int | None = None
        self._entered_at = 0.0
        self._dirty = False

    def segment_at(self, arclength: float) -> Segment:
        """Which segment contains this point on the lap."""
        index = int(np.searchsorted(self._starts, float(arclength), side="right") - 1)
        return self.segments[max(0, min(index, len(self.segments) - 1))]

    @property
    def current(self) -> Segment | None:
        """The segment being driven, or ``None`` before the first update."""
        return None if self._current is None else self.segments[self._current]

    @property
    def went_off_track(self) -> bool:
        """Whether the car has left the circuit inside the segment being driven."""
        return self._dirty

    def elapsed(self, now: float) -> float:
        """Seconds since the current segment was entered."""
        return 0.0 if self._current is None else max(0.0, now - self._entered_at)

    def reset(self, arclength: float = 0.0, now: float = 0.0) -> None:
        """Re-anchor at a point on the lap without completing anything."""
        self._current = self.segment_at(arclength).index
        self._entered_at = float(now)
        self._dirty = False

    def update(self, arclength: float, now: float, off_track: bool = False) -> SegmentTiming | None:
        """Feed one sample. Returns a timing when this sample completed a segment.

        Args:
            arclength: Distance along the centerline, from
                :meth:`~fly_driver.envs.centerline.Centerline.project`.
            now: Simulation time in seconds.
            off_track: Whether the car is outside the track limits right now. Sticky for
                the rest of the segment once seen.
        """
        here = self.segment_at(arclength).index
        if self._current is None:
            self.reset(arclength, now)
            self._dirty = off_track
            return None

        self._dirty = self._dirty or off_track
        if here == self._current:
            return None

        expected = (self._current + 1) % len(self.segments)
        if here != expected:
            # A jump: reversed over a line, or put back on the grid. Start again here
            # rather than award a time for a stretch that was not driven.
            self.reset(arclength, now)
            self._dirty = off_track
            return None

        completed = SegmentTiming(
            segment=self.segments[self._current],
            seconds=max(0.0, float(now) - self._entered_at),
            went_off_track=self._dirty,
        )
        self._current = here
        self._entered_at = float(now)
        self._dirty = off_track
        return completed
