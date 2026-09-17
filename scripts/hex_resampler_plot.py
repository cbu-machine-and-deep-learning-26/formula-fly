#!/usr/bin/env python3
"""Plot a known camera pattern next to its 721-column hex resampling.

The figure is the visual companion of the orientation and chirality test in
``tests/eyes/test_hex_resampler.py``: a bright top-left quadrant must land in
the top-left of the flyvis retina. Both panels sit on a mid-grey background so
that 0.0 (black) and 1.0 (white) luminance are both distinguishable from the
figure canvas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon

from fly_driver.eyes.hex_resampler import (
    DEFAULT_FRAME_SHAPE,
    HexResampler,
    frame_to_gray,
    hex_coordinates,
)

AXES_FACECOLOR = "0.5"
SOURCE_MARGIN_PIXELS = 3
HEX_CIRCUMRADIUS = 1 / np.sqrt(3)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        help="PNG path to write.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Output resolution (default: 180).",
    )
    return parser.parse_args()


def create_top_left_frame() -> np.ndarray:
    """Return the uint8 RGB frame with a bright top-left quadrant."""
    frame = np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.uint8)
    frame[: DEFAULT_FRAME_SHAPE[0] // 2, : DEFAULT_FRAME_SHAPE[1] // 2] = 255
    return frame


def hex_display_positions() -> tuple[np.ndarray, np.ndarray]:
    """Return display ``(x, y)`` per column with image-up as positive ``y``.

    The resampler places column ``v`` along image columns and ``u + v / 2``
    along image rows. Columns of constant ``v`` are spaced by ``sqrt(3) / 2``
    receptor units so flat-topped unit hexagons tile without gaps.
    """
    u_coordinates, v_coordinates = hex_coordinates()
    x_positions = (np.sqrt(3) / 2) * v_coordinates
    y_positions = -(u_coordinates + v_coordinates / 2)
    return x_positions, y_positions


def _plot_source(axes: plt.Axes, gray: np.ndarray) -> None:
    height, width = gray.shape
    axes.set_facecolor(AXES_FACECOLOR)
    axes.imshow(gray, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes.set_xlim(-0.5 - SOURCE_MARGIN_PIXELS, width - 0.5 + SOURCE_MARGIN_PIXELS)
    axes.set_ylim(height - 0.5 + SOURCE_MARGIN_PIXELS, -0.5 - SOURCE_MARGIN_PIXELS)
    axes.set_title("Source: bright top-left quadrant")
    axes.set_xlabel("image column \u2192 right")
    axes.set_ylabel("image row \u2192 down")


def _plot_hex(axes: plt.Axes, values: np.ndarray) -> PatchCollection:
    x_positions, y_positions = hex_display_positions()
    patches = [
        RegularPolygon(
            (x_position, y_position),
            numVertices=6,
            radius=HEX_CIRCUMRADIUS,
            orientation=np.pi / 6,
        )
        for x_position, y_position in zip(x_positions, y_positions)
    ]
    # Face-colored edges hide anti-aliasing seams between adjacent hexagons.
    collection = PatchCollection(patches, cmap="gray", edgecolor="face", linewidth=0.3)
    collection.set_array(values)
    collection.set_clim(0, 1)
    axes.add_collection(collection)
    axes.set_facecolor(AXES_FACECOLOR)
    axes.set_aspect("equal")
    axes.set_xlim(x_positions.min() - 1, x_positions.max() + 1)
    axes.set_ylim(y_positions.min() - 1, y_positions.max() + 1)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(f"Resampled: {values.size} flyvis columns")
    axes.set_xlabel("hex display x \u2192 right")
    axes.set_ylabel("hex display y \u2192 up")
    return collection


def save_figure(output: Path, dpi: int) -> Path:
    """Render the source frame and its hex resampling to ``output``.

    Args:
        output: PNG path to write; parent directories are created.
        dpi: Output resolution.

    Returns:
        The written path.
    """
    frame = create_top_left_frame()
    gray = frame_to_gray(frame)
    hex_values = HexResampler().frame(frame)[0, 0, 0].numpy()

    figure, (source_axes, hex_axes) = plt.subplots(
        1, 2, figsize=(10, 4.6), layout="constrained"
    )
    _plot_source(source_axes, gray)
    collection = _plot_hex(hex_axes, hex_values)
    figure.colorbar(collection, ax=hex_axes, label="luminance", shrink=0.7)
    figure.suptitle("Camera orientation and chirality survive hex resampling")

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def main() -> int:
    """Write the camera-to-hex figure and print its path."""
    args = _parse_args()
    matplotlib.use("Agg")
    with torch.no_grad():
        output = save_figure(args.output, args.dpi)
    print(f"figure: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
