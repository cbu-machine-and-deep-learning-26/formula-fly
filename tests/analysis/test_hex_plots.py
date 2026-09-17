"""Tests for the shared hexagonal map plotting helpers."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.analysis.hex_plots import hex_scatter, split_readout_maps
from fly_driver.eyes.hex_resampler import HEX_COLUMN_COUNT, hex_receptor_centers


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
