"""Track centerline geometry for the practice track (GH-16).

Pure numpy, no MuJoCo. Two reasons that matters: the geometry tests run on a headless CI
box with no GL context, and the reward function needs this maths every step without
touching the physics engine.

The centerline is the track's coordinate system. Everything downstream is expressed in
terms of it:

- **progress** along the lap is arclength, which is what the reward integrates
- **off-track** is the lateral offset compared against the half-width at that point
- the **track surface mesh** is the centerline swept left and right by those half-widths
- the **car's start pose** is a point on it, facing along the tangent

:meth:`Centerline.project` is the one function here that can be wrong in a way that still
looks plausible — a flipped sign convention makes left and right swap, and a mishandled
lap seam makes progress jump by a full lap length at the start/finish line. `AGENTS.md`
§11 calls out exactly this class of failure, so both are pinned by hand-computed tests in
``tests/test_centerline.py`` rather than eyeballed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["Centerline", "DEFAULT_CENTERLINE_PATH"]

#: The vendored Silverstone centerline. See ``data/README.md`` for provenance and licence.
DEFAULT_CENTERLINE_PATH = Path(__file__).parent / "data" / "silverstone_centerline.csv"

#: Geometry is float64 throughout. Arclength accumulates over ~5.9 km at millimetre scale,
#: and float32 would lose resolution at the far end of a lap.
_GEOM_DTYPE = np.float64


@dataclass(frozen=True)
class Projection:
    """Where a world point sits relative to the track.

    Args:
        arclength: Distance along the centerline from the start/finish line, in metres,
            wrapped into ``[0, length)``.
        lateral: Signed perpendicular offset in metres. **Positive is left** of the
            direction of travel, negative is right. The sign convention is asserted in
            tests, never inferred.
        half_width_left: Track half-width to the left at this point, in metres.
        half_width_right: Track half-width to the right at this point, in metres.
        segment: Index of the centerline segment the point projected onto.
    """

    arclength: float
    lateral: float
    half_width_left: float
    half_width_right: float
    segment: int

    @property
    def is_on_track(self) -> bool:
        """True when the point lies between the two track edges."""
        if self.lateral >= 0.0:
            return self.lateral <= self.half_width_left
        return -self.lateral <= self.half_width_right

    @property
    def edge_overshoot(self) -> float:
        """Metres beyond the nearer track edge; ``0.0`` when on track.

        This is what the off-track reward penalty scales with, so that clipping a kerb
        costs less than driving into a field.
        """
        if self.lateral >= 0.0:
            return max(0.0, self.lateral - self.half_width_left)
        return max(0.0, -self.lateral - self.half_width_right)


class Centerline:
    """A closed track centerline with per-point half-widths.

    The polyline is stored closed: the final point is the first point repeated, so segment
    ``i`` always runs from ``points[i]`` to ``points[i + 1]`` with no special case at the
    lap seam.

    Args:
        points: ``(N, 2)`` centerline positions in metres.
        half_width_right: ``(N,)`` half-width to the right, in metres.
        half_width_left: ``(N,)`` half-width to the left, in metres.
        close: When True (the default), append the first point to the end to close the
            loop. Pass False only if ``points`` is already explicitly closed.
    """

    def __init__(
        self,
        points: npt.ArrayLike,
        half_width_right: npt.ArrayLike,
        half_width_left: npt.ArrayLike,
        *,
        close: bool = True,
    ) -> None:
        pts = np.asarray(points, dtype=_GEOM_DTYPE)
        wr = np.asarray(half_width_right, dtype=_GEOM_DTYPE).reshape(-1)
        wl = np.asarray(half_width_left, dtype=_GEOM_DTYPE).reshape(-1)

        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"points must be (N, 2), got shape {pts.shape}")
        if pts.shape[0] < 3:
            raise ValueError(f"a track needs at least 3 points, got {pts.shape[0]}")
        if wr.shape[0] != pts.shape[0] or wl.shape[0] != pts.shape[0]:
            raise ValueError(
                f"width arrays must match points: got {pts.shape[0]} points, "
                f"{wr.shape[0]} right widths, {wl.shape[0]} left widths"
            )
        if not np.all(np.isfinite(pts)):
            raise ValueError("points contain non-finite values")
        if np.any(wr <= 0) or np.any(wl <= 0):
            raise ValueError("track half-widths must all be positive")

        if close:
            pts = np.vstack([pts, pts[:1]])
            wr = np.concatenate([wr, wr[:1]])
            wl = np.concatenate([wl, wl[:1]])

        segments = np.diff(pts, axis=0)
        seg_len = np.linalg.norm(segments, axis=1)
        if np.any(seg_len <= 0):
            bad = int(np.argmin(seg_len))
            raise ValueError(
                f"duplicate consecutive points at index {bad}; segment has zero length"
            )

        self._points = pts
        self._half_width_right = wr
        self._half_width_left = wl
        self._segments = segments
        self._segment_lengths = seg_len
        # arclength[i] is the distance from the start to points[i]; last entry is lap length.
        self._arclength = np.concatenate([[0.0], np.cumsum(seg_len)])

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CENTERLINE_PATH) -> Centerline:
        """Load from a TUMFTM-format CSV: ``x_m, y_m, w_tr_right_m, w_tr_left_m``.

        The file's header line is a ``#`` comment, so ``np.loadtxt`` skips it.

        Raises:
            FileNotFoundError: If ``path`` does not exist. The message names the packaging
                cause, because a missing CSV in a non-editable install is the likely way
                this fails on someone else's machine.
            ValueError: If the file does not have four columns.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"centerline data not found at {path}. If this is an installed (non-editable) "
                f"copy, the CSV may be missing from package-data in pyproject.toml."
            )
        raw = np.loadtxt(path, delimiter=",", comments="#", dtype=_GEOM_DTYPE)
        if raw.ndim != 2 or raw.shape[1] != 4:
            raise ValueError(
                f"expected 4 columns (x_m, y_m, w_tr_right_m, w_tr_left_m), "
                f"got shape {raw.shape} from {path}"
            )
        return cls(points=raw[:, :2], half_width_right=raw[:, 2], half_width_left=raw[:, 3])

    @property
    def points(self) -> npt.NDArray[np.float64]:
        """``(N + 1, 2)`` closed polyline; the last point repeats the first."""
        return self._points

    @property
    def half_width_left(self) -> npt.NDArray[np.float64]:
        return self._half_width_left

    @property
    def half_width_right(self) -> npt.NDArray[np.float64]:
        return self._half_width_right

    @property
    def arclength(self) -> npt.NDArray[np.float64]:
        """``(N + 1,)`` cumulative distance to each point; the last entry is :attr:`length`."""
        return self._arclength

    @property
    def length(self) -> float:
        """Total lap length in metres."""
        return float(self._arclength[-1])

    @property
    def num_segments(self) -> int:
        return int(self._segments.shape[0])

    def tangent(self, segment: int) -> npt.NDArray[np.float64]:
        """Unit direction of travel along ``segment``."""
        return self._segments[segment] / self._segment_lengths[segment]

    def normal(self, segment: int) -> npt.NDArray[np.float64]:
        """Unit left-pointing normal of ``segment``.

        Left is the 90-degree counter-clockwise rotation of the tangent, ``(-ty, tx)``.
        This single definition is what fixes the sign convention in :class:`Projection`.
        """
        tx, ty = self.tangent(segment)
        return np.array([-ty, tx], dtype=_GEOM_DTYPE)

    def pose_at(self, arclength: float) -> tuple[npt.NDArray[np.float64], float]:
        """Position and heading at a distance along the lap.

        Args:
            arclength: Metres from the start/finish line. Wrapped into ``[0, length)``, so
                a value past the end of the lap comes back round rather than clamping.

        Returns:
            ``(position_xy, yaw_radians)`` where yaw is measured from the +x axis.
        """
        s = float(arclength) % self.length
        # searchsorted gives the first index with arclength > s, so subtract one for the
        # segment containing s. side="right" keeps s exactly on a node in the segment that
        # starts there rather than the one that ends there.
        segment = int(np.searchsorted(self._arclength, s, side="right") - 1)
        segment = min(max(segment, 0), self.num_segments - 1)
        along = s - self._arclength[segment]
        tangent = self.tangent(segment)
        position = self._points[segment] + tangent * along
        yaw = float(np.arctan2(tangent[1], tangent[0]))
        return position, yaw

    def project(self, x: float, y: float) -> Projection:
        """Find where a world point sits relative to the track.

        Checks every segment rather than tracking a nearest-segment hint. The car can
        spin, reverse, or leave the track entirely during early RL training, and a
        stateful search would silently latch onto the wrong part of the lap when that
        happens — reporting a plausible arclength that is a kilometre wrong.

        Args:
            x: World x in metres.
            y: World y in metres.

        Returns:
            A :class:`Projection`. ``lateral`` is positive to the left of the direction of
            travel.
        """
        point = np.array([x, y], dtype=_GEOM_DTYPE)
        if not np.all(np.isfinite(point)):
            raise ValueError(f"point must be finite, got ({x!r}, {y!r})")

        starts = self._points[:-1]
        to_point = point - starts
        # Fraction along each segment of the closest point, clamped to the segment's ends
        # so a point off the side of a corner projects onto the corner itself.
        t = np.einsum("ij,ij->i", to_point, self._segments) / (self._segment_lengths**2)
        t = np.clip(t, 0.0, 1.0)
        closest = starts + self._segments * t[:, None]
        distances = np.linalg.norm(point - closest, axis=1)
        segment = int(np.argmin(distances))

        along = float(t[segment] * self._segment_lengths[segment])
        arclength = float(self._arclength[segment] + along) % self.length
        lateral = float(np.dot(point - closest[segment], self.normal(segment)))

        # Interpolate width across the segment; widths are per-point, not per-segment.
        frac = t[segment]
        wl = float(
            self._half_width_left[segment] * (1 - frac) + self._half_width_left[segment + 1] * frac
        )
        wr = float(
            self._half_width_right[segment] * (1 - frac)
            + self._half_width_right[segment + 1] * frac
        )
        return Projection(
            arclength=arclength,
            lateral=lateral,
            half_width_left=wl,
            half_width_right=wr,
            segment=segment,
        )

    def progress_delta(self, previous_s: float, current_s: float) -> float:
        """Signed lap progress between two arclengths, handling the start/finish seam.

        Crossing the line forwards takes arclength from ``length - epsilon`` to ``0``, a
        raw difference of nearly a negative lap. Left unhandled that reads as a huge
        backwards move and the reward would punish completing a lap — exactly the silent
        reward bug `AGENTS.md` §11 warns about.

        Any jump of more than half a lap is interpreted as a seam crossing rather than a
        genuine teleport, which is safe because no physics step moves the car 3 km.

        Returns:
            Metres of forward progress; negative when going backwards.
        """
        delta = float(current_s) - float(previous_s)
        half = self.length / 2.0
        if delta > half:
            delta -= self.length
        elif delta < -half:
            delta += self.length
        return delta

    def resample(self, spacing_m: float) -> Centerline:
        """Return a copy resampled to roughly uniform ``spacing_m`` spacing.

        Used to control track mesh density: the raw data is ~5 m apart, which is finer
        than the visual mesh needs on straights.

        Args:
            spacing_m: Target distance between consecutive points, in metres.
        """
        if spacing_m <= 0:
            raise ValueError(f"spacing_m must be positive, got {spacing_m}")
        if spacing_m >= self.length:
            raise ValueError(
                f"spacing_m={spacing_m} is at least the lap length {self.length:.1f}; "
                f"that would leave fewer than 2 points"
            )

        count = max(3, int(round(self.length / spacing_m)))
        targets = np.linspace(0.0, self.length, count, endpoint=False)
        points = np.empty((count, 2), dtype=_GEOM_DTYPE)
        wl = np.empty(count, dtype=_GEOM_DTYPE)
        wr = np.empty(count, dtype=_GEOM_DTYPE)
        for i, s in enumerate(targets):
            points[i] = self.pose_at(float(s))[0]
            # Widths come from the projection so they interpolate the same way everywhere.
            proj = self.project(float(points[i][0]), float(points[i][1]))
            wl[i] = proj.half_width_left
            wr[i] = proj.half_width_right
        return Centerline(points=points, half_width_right=wr, half_width_left=wl)

    def edges(self) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Left and right track edge polylines, each ``(N + 1, 2)``.

        The track surface mesh is the ribbon between these two. Each point is offset along
        the normal of the segment that starts there; at the closing point the first
        segment's normal is reused so the ribbon closes cleanly.
        """
        normals = np.empty_like(self._points)
        for i in range(self.num_segments):
            normals[i] = self.normal(i)
        normals[-1] = normals[0]
        left = self._points + normals * self._half_width_left[:, None]
        right = self._points - normals * self._half_width_right[:, None]
        return left, right

    def __repr__(self) -> str:
        return f"Centerline({self.num_segments} segments, length={self.length:.1f} m, closed)"
