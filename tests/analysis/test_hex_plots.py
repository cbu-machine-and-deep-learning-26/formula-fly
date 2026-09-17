"""Tests for the shared hexagonal map plotting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.analysis.hex_plots import (
    HexRaster,
    hex_raster_index,
    hex_scatter,
    split_readout_maps,
)
from fly_driver.eyes.hex_resampler import (
    HEX_COLUMN_COUNT,
    hex_coordinates,
    hex_receptor_centers,
)


def test_split_readout_maps_preserves_order_and_leading_axes() -> None:
    """Each readout gets its own 721-column slice, in concatenation order."""
    readouts = ("T4a", "T5b")
    features = np.arange(3 * 2 * HEX_COLUMN_COUNT, dtype=np.float32).reshape(
        3, 2 * HEX_COLUMN_COUNT
    )

    maps = split_readout_maps(features, readouts)

    assert list(maps) == list(readouts)
    assert maps["T4a"].shape == (3, HEX_COLUMN_COUNT)
    assert np.array_equal(maps["T4a"], features[:, :HEX_COLUMN_COUNT])
    assert np.array_equal(maps["T5b"], features[:, HEX_COLUMN_COUNT:])
    assert split_readout_maps(features[0], readouts)["T5b"].shape == (HEX_COLUMN_COUNT,)


def test_split_readout_maps_rejects_wrong_width() -> None:
    """A feature vector that is not readouts x 721 wide is an error."""
    with pytest.raises(ValueError, match="expected 2 readouts"):
        split_readout_maps(np.zeros(HEX_COLUMN_COUNT), ("T4a", "T4b"))


def test_hex_raster_index_covers_every_column_in_camera_orientation() -> None:
    """Every column owns pixels, the field is hexagonal, and axes point right/down."""
    resolution = 128
    index = hex_raster_index(resolution)
    u_coordinates, v_coordinates = hex_coordinates()

    assert index.shape == (resolution, resolution)
    assert index.min() == -1
    assert set(np.unique(index[index >= 0])) == set(range(HEX_COLUMN_COUNT))
    assert index[0, 0] == -1 and index[0, -1] == -1
    assert index[-1, 0] == -1 and index[-1, -1] == -1
    inside_fraction = (index >= 0).mean()
    assert 0.7 < inside_fraction < 0.8

    center = index[resolution // 2, resolution // 2]
    assert (u_coordinates[center], v_coordinates[center]) == (0, 0)
    right_of_center = index[resolution // 2, resolution // 2 + 14]
    assert v_coordinates[right_of_center] > 0
    below_center = index[resolution // 2 + 14, resolution // 2]
    row_offset = u_coordinates[below_center] + v_coordinates[below_center] / 2
    assert row_offset > 0


def test_hex_raster_renders_values_and_masks_background() -> None:
    """Pixels take their column's value; background pixels are masked."""
    raster = HexRaster(resolution=64)
    values = np.arange(HEX_COLUMN_COUNT, dtype=np.float32)

    image = raster.render(values)

    assert image.shape == (64, 64)
    assert np.array_equal(np.ma.getmaskarray(image), raster.index < 0)
    inside = raster.index >= 0
    assert np.array_equal(np.asarray(image)[inside], values[raster.index[inside]])
    with pytest.raises(ValueError, match="shape"):
        raster.render(values[:-1])


def test_hex_scatter_draws_columns_in_camera_orientation() -> None:
    """Column offsets go right and row offsets go down, one marker per column."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    values = np.linspace(-1, 1, HEX_COLUMN_COUNT, dtype=np.float32)

    scatter = hex_scatter(axes, values, cmap="coolwarm", vmin=-1, vmax=1)

    offsets = scatter.get_offsets()
    centers = hex_receptor_centers().numpy()
    assert offsets.shape == (HEX_COLUMN_COUNT, 2)
    assert np.array_equal(offsets[:, 0], centers[:, 1])
    assert np.array_equal(offsets[:, 1], -centers[:, 0])
    assert np.array_equal(scatter.get_array(), values)
    assert axes.get_aspect() == 1.0
    plt.close(figure)
