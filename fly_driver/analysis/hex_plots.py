"""Plot flyvis hexagonal column maps in camera orientation.

Readout features from :class:`fly_driver.eyes.FlyvisEye` are one 721-value map
per cell type, in flyvis hexagonal column order. These helpers split a feature
vector into those maps and draw one map so the picture is oriented like the
camera frame (row down, column right): :func:`hex_scatter` draws one marker per
column for static figures, and :class:`HexRaster` paints the same lattice as a
small image (one lookup per pixel) for live views where drawing 5,000+ markers
per frame is too slow. Callers pass in the axes; matplotlib is only imported
lazily for the raster's colormap.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from fly_driver.eyes.hex_resampler import (
    HEX_COLUMN_COUNT,
    HEX_EXTENT,
    HEX_KERNEL_SIZE,
    hex_receptor_centers,
)

__all__ = [
    "HEX_BACKGROUND_COLOR",
    "HexRaster",
    "hex_raster_index",
    "hex_scatter",
    "split_readout_maps",
]

HEX_BACKGROUND_COLOR = "#808080"
HEX_MARKER_SIZE = 9
DEFAULT_RASTER_RESOLUTION = 128
# Receptor centers are 13 px apart along u and ~14.5 px along v, so the Voronoi
# cell around each center reaches about 0.6 of the larger spacing.
RASTER_CELL_RADIUS_FACTOR = 0.62


def split_readout_maps(
    features: npt.NDArray[np.floating[Any]], readout_names: Sequence[str]
) -> dict[str, npt.NDArray[np.floating[Any]]]:
    """Split concatenated readout features into one 721-column map per readout.

    Args:
        features: ``(..., len(readout_names) * 721)`` features; leading axes
            (for example time) are preserved.
        readout_names: Cell types in the order they were concatenated.

    Returns:
        ``{readout_name: (..., 721) map}``.

    Raises:
        ValueError: If the last axis does not hold one map per readout.
    """
    expected = len(readout_names) * HEX_COLUMN_COUNT
    if features.shape[-1] != expected:
        raise ValueError(
            f"features last axis is {features.shape[-1]}, expected "
            f"{len(readout_names)} readouts x {HEX_COLUMN_COUNT} columns = {expected}"
        )
    return {
        name: features[..., index * HEX_COLUMN_COUNT : (index + 1) * HEX_COLUMN_COUNT]
        for index, name in enumerate(readout_names)
    }


def hex_raster_index(
    resolution: int = DEFAULT_RASTER_RESOLUTION,
    extent: int = HEX_EXTENT,
    kernel_size: int = HEX_KERNEL_SIZE,
) -> npt.NDArray[np.int64]:
    """Map every pixel of a square raster to its nearest hexagonal column.

    The raster covers the lattice's bounding square in camera orientation.
    Pixels farther from every receptor center than one cell radius (outside
    the hexagonal field) get ``-1``.

    Args:
        resolution: Raster side length in pixels.
        extent: Hexagonal lattice radius in receptors.
        kernel_size: Receptor spacing in flyvis pixels.

    Returns:
        An ``(resolution, resolution)`` int64 array of column indices or ``-1``.
    """
    if resolution < 1:
        raise ValueError("resolution must be positive")
    centers = hex_receptor_centers(extent, kernel_size).numpy().astype(np.float32)
    half_span = (extent + 0.5) * kernel_size
    pixel_positions = (np.arange(resolution, dtype=np.float32) + 0.5) / resolution
    offsets = pixel_positions * 2 * half_span - half_span
    rows, columns = np.meshgrid(offsets, offsets, indexing="ij")
    pixels = np.stack([rows.ravel(), columns.ravel()], axis=1)

    nearest = np.empty(pixels.shape[0], dtype=np.int64)
    nearest_distance = np.empty(pixels.shape[0], dtype=np.float32)
    chunk = 4096
    for start in range(0, pixels.shape[0], chunk):
        block = pixels[start : start + chunk]
        distances = np.linalg.norm(block[:, None, :] - centers[None, :, :], axis=2)
        nearest[start : start + chunk] = distances.argmin(axis=1)
        nearest_distance[start : start + chunk] = distances.min(axis=1)

    largest_spacing = float(np.hypot(kernel_size / 2, kernel_size))
    outside = nearest_distance > RASTER_CELL_RADIUS_FACTOR * largest_spacing
    nearest[outside] = -1
    return nearest.reshape(resolution, resolution)


class HexRaster:
    """Paint 721-column maps as small images for fast live drawing.

    Args:
        resolution: Raster side length in pixels.
    """

    def __init__(self, resolution: int = DEFAULT_RASTER_RESOLUTION) -> None:
        self.index = hex_raster_index(resolution)
        self.is_inside = self.index >= 0
        self._lookup = np.where(self.is_inside, self.index, 0)

    def render(
        self, values: npt.NDArray[np.floating[Any]]
    ) -> np.ma.MaskedArray[Any, np.dtype[np.float32]]:
        """Return the map as a masked float32 image (background masked out).

        Args:
            values: ``(721,)`` activity values in flyvis column order.

        Returns:
            A ``(resolution, resolution)`` masked array for ``imshow``.
        """
        values = np.asarray(values, dtype=np.float32)
        if values.shape != (HEX_COLUMN_COUNT,):
            raise ValueError(f"values must have shape ({HEX_COLUMN_COUNT},)")
        return np.ma.MaskedArray(values[self._lookup], mask=~self.is_inside)

    def imshow(
        self, axes: Any, values: npt.NDArray[np.floating[Any]], **kwargs: Any
    ) -> Any:
        """Draw one map on ``axes``; update it later with ``image.set_data``.

        Args:
            axes: matplotlib axes to draw on.
            values: ``(721,)`` activity values in flyvis column order.
            **kwargs: Forwarded to ``axes.imshow`` (``cmap``, ``vmin``, ``vmax``, ...).

        Returns:
            The image artist.
        """
        import matplotlib

        colormap = matplotlib.colormaps[kwargs.pop("cmap", "viridis")].copy()
        colormap.set_bad(HEX_BACKGROUND_COLOR)
        axes.set_facecolor(HEX_BACKGROUND_COLOR)
        image = axes.imshow(self.render(values), cmap=colormap, **kwargs)
        axes.set_xticks([])
        axes.set_yticks([])
        return image


def hex_scatter(axes: Any, values: npt.NDArray[np.floating[Any]], **kwargs: Any) -> Any:
    """Draw one 721-column map on ``axes`` in camera orientation.

    Args:
        axes: matplotlib axes to draw on.
        values: ``(721,)`` activity values in flyvis column order.
        **kwargs: Forwarded to ``axes.scatter`` (``cmap``, ``vmin``, ``vmax``, ...).

    Returns:
        The scatter artist, whose ``set_array`` updates the colours in place.
    """
    centers = hex_receptor_centers().numpy()
    axes.set_facecolor(HEX_BACKGROUND_COLOR)
    scatter = axes.scatter(
        centers[:, 1],
        -centers[:, 0],
        c=values,
        s=HEX_MARKER_SIZE,
        marker="h",
        linewidths=0,
        **kwargs,
    )
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    return scatter
