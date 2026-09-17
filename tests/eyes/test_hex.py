"""Tests for the Cartesian-camera to flyvis-lattice transform."""

import numpy as np
import pytest

from fly_driver.eyes.hex import (
    DEFAULT_FRAME_SHAPE,
    HEX_EXTENT,
    NUM_HEX_COLUMNS,
    HexResampler,
)
from fly_driver.eyes.stimuli import generate_drifting_grating, generate_moving_edge


def test_default_lattice_has_flyvis_column_count_and_order() -> None:
    resampler = HexResampler()

    assert len(resampler.u_coordinates) == NUM_HEX_COLUMNS == 721
    assert list(
        zip(
            resampler.u_coordinates[:4],
            resampler.v_coordinates[:4],
            strict=True,
        )
    ) == [(-15, 0), (-15, 1), (-15, 2), (-15, 3)]
    assert (
        resampler.u_coordinates[360],
        resampler.v_coordinates[360],
    ) == (0, 0)
    assert (
        resampler.u_coordinates[-1],
        resampler.v_coordinates[-1],
    ) == (15, 0)
    assert resampler.column_index(0, 0) == 360


def test_every_coordinate_is_inside_extent_15_hexagon() -> None:
    resampler = HexResampler()
    u_coordinates = resampler.u_coordinates
    v_coordinates = resampler.v_coordinates

    assert np.all(np.abs(u_coordinates) <= HEX_EXTENT)
    assert np.all(np.abs(v_coordinates) <= HEX_EXTENT)
    assert np.all(np.abs(u_coordinates + v_coordinates) <= HEX_EXTENT)
    assert len(set(zip(u_coordinates, v_coordinates, strict=True))) == 721


def test_horizontal_and_vertical_gradients_fix_orientation_and_chirality() -> None:
    height, width, _ = DEFAULT_FRAME_SHAPE
    resampler = HexResampler(channel="green")
    horizontal_gradient = np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.uint8)
    horizontal_gradient[:, :, 1] = np.arange(width, dtype=np.uint8)
    vertical_gradient = np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.uint8)
    vertical_gradient[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]

    horizontal_values = resampler.resample_frame(horizontal_gradient)
    vertical_values = resampler.resample_frame(vertical_gradient)
    flyvis_vertical_coordinate = resampler.u_coordinates + resampler.v_coordinates / 2.0

    assert np.corrcoef(horizontal_values, resampler.v_coordinates)[0, 1] > 0.99
    assert np.corrcoef(vertical_values, flyvis_vertical_coordinate)[0, 1] > 0.99
    assert (
        horizontal_values[resampler.column_index(0, -10)]
        < horizontal_values[resampler.column_index(0, 10)]
    )
    assert (
        vertical_values[resampler.column_index(-10, 0)]
        < vertical_values[resampler.column_index(10, 0)]
    )


def test_right_moving_edge_advances_along_positive_v() -> None:
    resampler = HexResampler()
    frames = generate_moving_edge(
        num_frames=9,
        frame_rate_hz=8.0,
        speed_pixels_per_second=32.0,
        start_position_pixels=-16.0,
        direction_degrees=0.0,
    )

    values = resampler.resample_sequence(frames)
    newly_bright = np.maximum(np.diff(values, axis=0), 0.0)
    v_centroids = np.sum(
        newly_bright * resampler.v_coordinates[None, :], axis=1
    ) / np.sum(newly_bright, axis=1)

    assert np.all(np.diff(v_centroids) > 0)


def test_upper_left_spot_lands_in_expected_flyvis_column() -> None:
    resampler = HexResampler()
    expected_coordinate = (-8, -7)
    expected_index = resampler.column_index(*expected_coordinate)
    expected_row, expected_column = resampler.pixel_coordinates[expected_index]
    rows, columns = np.indices(DEFAULT_FRAME_SHAPE[:2])
    squared_distance = (rows - expected_row) ** 2 + (columns - expected_column) ** 2
    spot = np.rint(255.0 * np.exp(-squared_distance / 2.0)).astype(np.uint8)
    frame = np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.uint8)
    frame[:, :, 1] = spot

    values = resampler.resample_frame(frame)

    assert expected_row < DEFAULT_FRAME_SHAPE[0] / 2
    assert expected_column < DEFAULT_FRAME_SHAPE[1] / 2
    assert np.argmax(values) == expected_index


def test_center_and_boundary_columns_map_to_expected_pixels() -> None:
    resampler = HexResampler()

    center = resampler.pixel_coordinates[resampler.column_index(0, 0)]
    left = resampler.pixel_coordinates[resampler.column_index(0, -15)]
    right = resampler.pixel_coordinates[resampler.column_index(0, 15)]

    np.testing.assert_allclose(center, [47.5, 47.5])
    assert left[1] == pytest.approx(0.0)
    assert right[1] == pytest.approx(95.0)
    assert left[0] < center[0] < right[0]


def test_sampling_geometry_is_precomputed_and_read_only() -> None:
    resampler = HexResampler()
    indices = resampler.source_indices
    weights = resampler.source_weights
    frame = np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.uint8)

    resampler.resample_frame(frame)
    resampler.resample_frame(frame)

    assert resampler.source_indices is indices
    assert resampler.source_weights is weights
    assert indices.shape == (721, 4)
    assert weights.shape == (721, 4)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0)
    assert not indices.flags.writeable
    assert not weights.flags.writeable


def test_green_and_grayscale_channel_choices_are_normalized() -> None:
    frame = np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.uint8)
    frame[:, :, 0] = 100
    frame[:, :, 1] = 200
    frame[:, :, 2] = 50

    green = HexResampler(channel="green").resample_frame(frame)
    grayscale = HexResampler(channel="grayscale").resample_frame(frame)

    np.testing.assert_allclose(green, 200.0 / 255.0, rtol=1e-6)
    expected_grayscale = (0.299 * 100 + 0.587 * 200 + 0.114 * 50) / 255
    np.testing.assert_allclose(grayscale, expected_grayscale, rtol=1e-6)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((95, 96, 3), dtype=np.uint8),
        np.zeros((96, 95, 3), dtype=np.uint8),
        np.zeros((96, 96, 1), dtype=np.uint8),
    ],
)
def test_unexpected_frame_shapes_raise_without_resizing(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="no implicit resize"):
        HexResampler().resample_frame(frame)


def test_wrong_frame_container_and_dtype_raise() -> None:
    resampler = HexResampler()

    with pytest.raises(TypeError, match="numpy array"):
        resampler.resample_frame([[[0, 0, 0]]])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="uint8"):
        resampler.resample_frame(
            np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.float32)  # type: ignore[arg-type]
        )


def test_sequence_preserves_time_axis_and_matches_individual_frames() -> None:
    frames = generate_drifting_grating(num_frames=5)
    resampler = HexResampler(frame_rate_hz=50.0)

    sequence_values = resampler.resample_sequence(frames)
    individual_values = np.stack([resampler.resample_frame(frame) for frame in frames])

    assert sequence_values.shape == (5, 721)
    assert sequence_values.dtype == np.float32
    assert resampler.frame_interval_seconds == pytest.approx(0.02)
    np.testing.assert_array_equal(sequence_values, individual_values)


def test_unexpected_sequence_shape_and_dtype_raise() -> None:
    resampler = HexResampler()

    with pytest.raises(ValueError, match="no implicit resize"):
        resampler.resample_sequence(np.zeros((2, 95, 96, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="at least one"):
        resampler.resample_sequence(np.zeros((0, *DEFAULT_FRAME_SHAPE), dtype=np.uint8))
    with pytest.raises(TypeError, match="uint8"):
        resampler.resample_sequence(
            np.zeros((2, *DEFAULT_FRAME_SHAPE), dtype=np.float32)
        )


def test_repeated_resampling_is_bitwise_deterministic() -> None:
    random_generator = np.random.default_rng(13)
    frame = random_generator.integers(0, 256, size=DEFAULT_FRAME_SHAPE, dtype=np.uint8)
    first_resampler = HexResampler()
    second_resampler = HexResampler()

    first = first_resampler.resample_frame(frame)
    repeated = first_resampler.resample_frame(frame)
    independent = second_resampler.resample_frame(frame)

    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first, independent)
    np.testing.assert_array_equal(
        first_resampler.source_indices, second_resampler.source_indices
    )
    np.testing.assert_array_equal(
        first_resampler.source_weights, second_resampler.source_weights
    )


def test_drifting_grating_moves_at_requested_speed_and_direction() -> None:
    frames = generate_drifting_grating(
        num_frames=3,
        spatial_period_pixels=16.0,
        temporal_frequency_hz=5.0,
        frame_rate_hz=40.0,
        direction_degrees=0.0,
    )

    assert frames.shape == (3, *DEFAULT_FRAME_SHAPE)
    assert frames.dtype == np.uint8
    np.testing.assert_array_equal(frames[..., 0], frames[..., 1])
    np.testing.assert_array_equal(frames[..., 1], frames[..., 2])
    np.testing.assert_array_equal(
        frames[1, :, :, 1], np.roll(frames[0, :, :, 1], 2, axis=1)
    )
    np.testing.assert_array_equal(
        frames[2, :, :, 1], np.roll(frames[1, :, :, 1], 2, axis=1)
    )


def test_moving_edge_generator_advances_without_randomness() -> None:
    arguments = {
        "frame_shape": (32, 40, 3),
        "num_frames": 4,
        "speed_pixels_per_second": 8.0,
        "frame_rate_hz": 4.0,
        "direction_degrees": 0.0,
        "start_position_pixels": -3.0,
    }

    first = generate_moving_edge(**arguments)
    second = generate_moving_edge(**arguments)
    bright_counts = np.count_nonzero(first[:, :, :, 1] == 255, axis=(1, 2))

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.diff(bright_counts), [64, 64, 64])


def test_flyvis_extra_uses_identical_column_order_for_direction_checks() -> None:
    hex_utils = pytest.importorskip(
        "flyvis.utils.hex_utils",
        reason="T4/T5 direction checks require the flyvis extra",
    )
    resampler = HexResampler()

    flyvis_u, flyvis_v = hex_utils.get_hex_coords(HEX_EXTENT)

    np.testing.assert_array_equal(resampler.u_coordinates, flyvis_u)
    np.testing.assert_array_equal(resampler.v_coordinates, flyvis_v)
    grating = generate_drifting_grating(num_frames=4)
    assert resampler.resample_sequence(grating)[None, :, None, :].shape == (
        1,
        4,
        1,
        721,
    )
