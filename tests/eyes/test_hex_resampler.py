"""Tests for the camera-to-flyvis hexagonal resampler."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fly_driver.eyes.hex_resampler import (
    DEFAULT_FRAME_SHAPE,
    HEX_COLUMN_COUNT,
    HEX_EXTENT,
    HEX_KERNEL_SIZE,
    HexResampler,
    frame_to_gray,
    hex_coordinates,
    hex_receptor_centers,
)


def _random_frame(seed: int = 0) -> np.ndarray:
    random_generator = np.random.default_rng(seed)
    return random_generator.integers(0, 256, size=DEFAULT_FRAME_SHAPE, dtype=np.uint8)


def test_frame_to_gray_range_and_shape() -> None:
    """Convert RGB frames without changing geometry or numeric range."""
    gray = frame_to_gray(_random_frame())

    assert gray.shape == DEFAULT_FRAME_SHAPE[:2]
    assert gray.dtype == np.float32
    assert gray.min() >= 0.0 and gray.max() <= 1.0
    white = np.full(DEFAULT_FRAME_SHAPE, 255, dtype=np.uint8)
    assert frame_to_gray(white).max() == pytest.approx(1.0)


def test_receptor_grid_matches_flyvis_defaults() -> None:
    """Build 721 receptor centers spanning flyvis's minimum frame."""
    centers = hex_receptor_centers()

    assert centers.shape == (HEX_COLUMN_COUNT, 2)
    span = centers.max(dim=0).values - centers.min(dim=0).values + 1
    expected_size = 2 * HEX_EXTENT * HEX_KERNEL_SIZE + 1
    assert span.tolist() == [expected_size, expected_size]


def test_column_order_matches_flyvis_hex_coordinates() -> None:
    """Use flyvis's exact u-then-v ordering when it is installed."""
    flyvis_hex_utils = pytest.importorskip("flyvis.utils.hex_utils")

    expected_u, expected_v = flyvis_hex_utils.get_hex_coords(HEX_EXTENT)
    actual_u, actual_v = hex_coordinates()

    assert np.array_equal(actual_u, expected_u)
    assert np.array_equal(actual_v, expected_v)


def test_top_left_pattern_proves_orientation_and_chirality() -> None:
    """Map image top-left to negative column and row receptor offsets."""
    frame = np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.uint8)
    frame[: DEFAULT_FRAME_SHAPE[0] // 2, : DEFAULT_FRAME_SHAPE[1] // 2] = 255
    output = HexResampler().frame(frame)[0, 0, 0]
    centers = hex_receptor_centers()
    safe_margin = 2 * HEX_KERNEL_SIZE
    lattice_radius = HEX_EXTENT * HEX_KERNEL_SIZE
    interior = (centers.abs() <= lattice_radius - HEX_KERNEL_SIZE).all(dim=1)

    top_left = (
        interior & (centers[:, 0] <= -safe_margin) & (centers[:, 1] <= -safe_margin)
    )
    top_right = (
        interior & (centers[:, 0] <= -safe_margin) & (centers[:, 1] >= safe_margin)
    )
    bottom_left = (
        interior & (centers[:, 0] >= safe_margin) & (centers[:, 1] <= -safe_margin)
    )

    assert output[top_left].min() > 0.99
    assert output[top_right].max() < 0.01
    assert output[bottom_left].max() < 0.01

    u_coordinates, v_coordinates = hex_coordinates()
    assert np.array_equal(centers[:, 1].numpy(), HEX_KERNEL_SIZE * v_coordinates)
    expected_rows = HEX_KERNEL_SIZE * (u_coordinates + v_coordinates / 2)
    assert np.array_equal(centers[:, 0].numpy(), expected_rows.astype(int))


def test_contract_frame_produces_flyvis_input_shape() -> None:
    """Produce the batch, time, channel, column shape flyvis simulates."""
    output = HexResampler().frame(_random_frame())

    assert output.shape == (1, 1, 1, HEX_COLUMN_COUNT)
    assert torch.isfinite(output).all()
    assert 0.0 <= output.min() and output.max() <= 1.0


@pytest.mark.parametrize(
    "frame, error_type",
    [
        (np.zeros((95, 96, 3), dtype=np.uint8), ValueError),
        (np.zeros((96, 96), dtype=np.uint8), ValueError),
        (np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.float32), TypeError),
        ([[[0, 0, 0]]], TypeError),
    ],
)
def test_frame_validation_rejects_without_resizing(
    frame: object, error_type: type[Exception]
) -> None:
    """Reject contract mismatches rather than silently resize or cast them."""
    with pytest.raises(error_type):
        frame_to_gray(frame)  # type: ignore[arg-type]


def test_gray_sequence_rejects_undeclared_frame_size() -> None:
    """Reject direct grayscale input that bypasses RGB frame validation."""
    sequence = torch.zeros((1, 2, 95, 96), dtype=torch.float32)

    with pytest.raises(ValueError, match="No implicit resize"):
        HexResampler()(sequence)


def test_batch_and_time_axes_are_independent() -> None:
    """Keep each batch item and time step independent while resampling."""
    resampler = HexResampler()
    frames = np.stack([frame_to_gray(_random_frame(seed)) for seed in range(6)])
    sequence = torch.from_numpy(frames).reshape(2, 3, 96, 96)

    output = resampler(sequence)
    single_output = resampler(sequence[1:2, 2:3])

    assert output.shape == (2, 3, 1, HEX_COLUMN_COUNT)
    assert torch.allclose(output[1, 2], single_output[0, 0])


def test_matches_real_flyvis_box_eye() -> None:
    """Match flyvis BoxEye numerically when the optional package is installed."""
    flyvis_rendering = pytest.importorskip("flyvis.datasets.rendering")
    box_eye = flyvis_rendering.BoxEye(extent=HEX_EXTENT, kernel_size=HEX_KERNEL_SIZE)
    frames = np.stack([frame_to_gray(_random_frame(seed)) for seed in range(4)])
    sequence = torch.from_numpy(frames).reshape(2, 2, 96, 96)

    ours = HexResampler()(sequence)
    theirs = box_eye(sequence.clone(), ftype="mean", hex_sample=True)

    assert box_eye.hexals == HEX_COLUMN_COUNT
    assert ours.shape == theirs.shape
    assert torch.allclose(ours, theirs, atol=1e-5)
