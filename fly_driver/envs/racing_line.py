"""A static racing line for the practice track (GH-16).

Not the centreline. A centreline is the middle of the road; a racing line straightens the
corners out by running wide on entry, clipping the apex and drifting wide again on exit,
because what limits a car in a corner is curvature and curvature is what this minimises.

The line is computed once from the track's own geometry rather than recorded from a lap,
so it does not depend on anyone having driven well, and it is the same every time.

Method: the line is described by one number per point -- how far it sits to the left or
right of the centreline -- and those numbers are pushed downhill on total squared
curvature by gradient descent, clamped back inside the track edges after each step.

The gradient matters and is easy to get wrong. Curvature at a point is the second
difference of the path, so the gradient of the *sum of its squares* is the second
difference of that again -- a fourth difference. Descending on the second difference
instead gives mean-curvature flow, which shortens a closed loop rather than straightening
it, and drives the line into the track edges and stays there: on Silverstone that pinned
37% of the points against a boundary and left the total curvature slightly *worse* than
the centreline it started from. The fourth difference is the real thing and converges
monotonically.

A textbook minimum-curvature solve would set this up as one least-squares problem over all
points at once. That is a dense 2358x1179 system for Silverstone, seconds per scene build,
against milliseconds here. It is a guide to look at, not a trajectory anything is scored
against, so the cheap method wins as long as it is the correct cheap method.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fly_driver.envs.centerline import Centerline

__all__ = ["racing_line", "total_curvature"]

#: Computed lines, keyed by the track and the settings that produced them. Twenty thousand
#: gradient steps is about a second, which is fine once and not fine on every scene build --
#: and the tests build a great many scenes.
_CACHE: dict[tuple, npt.NDArray[np.float64]] = {}


def _point_normals(points: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Unit normals at each point, left of the direction of travel.

    From a central difference of the neighbours rather than a single segment, so the
    normal at a corner bisects it instead of jumping between the two sides.
    """
    tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(lengths, 1e-9)
    # Left of travel: rotate the tangent a quarter turn anticlockwise.
    return np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)


def total_curvature(points: npt.NDArray[np.float64]) -> float:
    """Sum of squared discrete curvature around a closed path, for comparing lines."""
    second = np.roll(points, 1, axis=0) - 2.0 * points + np.roll(points, -1, axis=0)
    return float(np.sum(second**2))


def racing_line(
    centerline: Centerline,
    *,
    margin_m: float = 1.2,
    spacing_m: float = 4.0,
    iterations: int = 20_000,
    rate: float = 0.05,
) -> npt.NDArray[np.float64]:
    """An ``(N, 2)`` closed racing line through the track, in world metres.

    Args:
        centerline: The circuit.
        margin_m: How far inside the track edge the line is kept. This is half a car plus
            a little, so the line marks where a car's *centre* can be rather than where
            its outside wheel would end up.
        spacing_m: Point spacing along the line.
        iterations: Gradient steps. Convergence is slow and the count decides how much of
            the road the line is willing to use: on Silverstone 1,000 steps wander 3.8 m
            either side of the centreline, 4,000 wander 7.0 m and 20,000 wander 11 m,
            which is most of the 11.4 m the margin leaves. A racing line that stayed near
            the middle would not be one, so this is set high and the result is cached.
        rate: Step size per iteration. The fourth-difference operator has eigenvalues up to
            16, so anything much above 1/16 overshoots and the line oscillates.

    Raises:
        ValueError: If the arguments cannot produce a line -- a non-positive spacing, or a
            margin wide enough to leave no room between the edges.
    """
    if spacing_m <= 0:
        raise ValueError(f"spacing_m must be positive, got {spacing_m}")
    if margin_m < 0:
        raise ValueError(f"margin_m must be non-negative, got {margin_m}")
    if not 0.0 < rate <= 0.0625:
        raise ValueError(f"rate must be in (0, 1/16], got {rate}")

    line = centerline.resample(spacing_m)
    points = np.asarray(line.points, dtype=float)
    key = (points.tobytes(), margin_m, spacing_m, int(iterations), rate)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached.copy()

    normals = _point_normals(points)

    # Room to move either way, after keeping clear of the edges.
    left = np.maximum(np.asarray(line.half_width_left, dtype=float) - margin_m, 0.0)
    right = np.maximum(np.asarray(line.half_width_right, dtype=float) - margin_m, 0.0)
    if float(np.max(left + right)) <= 0.0:
        raise ValueError(f"margin_m={margin_m} leaves no room between the track edges")

    offset = np.zeros(len(points))
    for _ in range(int(iterations)):
        current = points + offset[:, None] * normals
        curvature = np.roll(current, 1, axis=0) - 2.0 * current + np.roll(current, -1, axis=0)
        # Gradient of the summed squared curvature: the second difference of the curvature
        # itself. See the module docstring for why this is not the curvature.
        gradient = np.roll(curvature, 1, axis=0) - 2.0 * curvature + np.roll(curvature, -1, axis=0)
        # Move only along the normal, so the line stays parameterised by the centreline
        # and cannot bunch up or double back on itself.
        offset = offset - rate * np.einsum("ij,ij->i", gradient, normals)
        offset = np.clip(offset, -right, left)

    result = points + offset[:, None] * normals
    _CACHE[key] = result
    return result.copy()
