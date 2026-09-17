"""Convert Cartesian camera frames to flyvis's hexagonal retina input.

The fly's compound eye is a honeycomb of small light collectors. Flyvis models
that retina by enlarging a Cartesian frame, applying a square mean filter, and
sampling one value at each of 721 hexagonal receptor centers. The resulting
flat sequence is ordered by axial ``(u, v)`` coordinates, with ``u`` changing
slowest, exactly like :func:`flyvis.utils.hex_utils.get_hex_coords`.

This implementation reproduces ``flyvis.datasets.rendering.BoxEye`` without
importing flyvis, so geometry tests can run in the base project environment.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch.nn import functional as F

__all__ = [
    "DEFAULT_FRAME_SHAPE",
    "HEX_COLUMN_COUNT",
    "HEX_EXTENT",
    "HEX_KERNEL_SIZE",
    "HexResampler",
    "frame_to_gray",
    "hex_coordinates",
    "hex_receptor_centers",
]

DEFAULT_FRAME_SHAPE = (96, 96, 3)
HEX_EXTENT = 15
HEX_KERNEL_SIZE = 13
HEX_COLUMN_COUNT = 721
RGB_LUMA_WEIGHTS = (0.299, 0.587, 0.114)

Frame = npt.NDArray[np.uint8]


def frame_to_gray(
    frame: Frame, *, expected_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE
) -> npt.NDArray[np.float32]:
    """Convert a validated uint8 RGB frame to grayscale float32 in ``[0, 1]``.

    Args:
        frame: Camera frame with shape ``expected_shape``.
        expected_shape: Declared ``(height, width, 3)`` camera output.

    Returns:
        A ``(height, width)`` grayscale array.

    Raises:
        TypeError: If the frame is not a uint8 NumPy RGB array.
        ValueError: If the frame has an unexpected shape.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"frame must be a numpy array, got {type(frame).__name__}")
    if frame.dtype != np.uint8:
        raise TypeError(
            f"frame must be uint8 in [0, 255], got {frame.dtype}. "
            "No implicit cast is performed."
        )
    if frame.shape != expected_shape:
        raise ValueError(
            f"frame shape {frame.shape} != expected {expected_shape}. "
            "No implicit resize is performed."
        )

    array = frame.astype(np.float32) / 255.0
    weights = np.asarray(RGB_LUMA_WEIGHTS, dtype=np.float32)
    return (array @ weights).astype(np.float32)


def hex_coordinates(extent: int = HEX_EXTENT) -> tuple[np.ndarray, np.ndarray]:
    """Return axial ``(u, v)`` coordinates in flyvis column order.

    Args:
        extent: Hexagonal lattice radius in receptors.

    Returns:
        Integer ``u`` and ``v`` arrays, each with one entry per hex column.
    """
    u_coordinates: list[int] = []
    v_coordinates: list[int] = []
    for u_coordinate in range(-extent, extent + 1):
        v_min = max(-extent, -extent - u_coordinate)
        v_max = min(extent, extent - u_coordinate)
        for v_coordinate in range(v_min, v_max + 1):
            u_coordinates.append(u_coordinate)
            v_coordinates.append(v_coordinate)
    return np.asarray(u_coordinates), np.asarray(v_coordinates)


def hex_receptor_centers(
    extent: int = HEX_EXTENT, kernel_size: int = HEX_KERNEL_SIZE
) -> torch.Tensor:
    """Return receptor center ``(row, column)`` offsets in flyvis order.

    Args:
        extent: Hexagonal lattice radius in receptors.
        kernel_size: Pixel spacing between adjacent receptor centers.

    Returns:
        A ``(hexals, 2)`` long tensor measured from the frame center.
    """
    u_coordinates, v_coordinates = hex_coordinates(extent)
    row_offsets = kernel_size * (u_coordinates + v_coordinates / 2)
    column_offsets = kernel_size * v_coordinates
    centers = np.stack((row_offsets, column_offsets), axis=-1)
    return torch.as_tensor(centers, dtype=torch.long)


class HexResampler:
    """Reproduce flyvis ``BoxEye`` mean filtering and hexagonal sampling.

    Args:
        extent: Hexagonal lattice radius in receptors.
        kernel_size: Mean-filter size and receptor spacing in pixels.
        expected_frame_shape: Declared camera frame shape. Other sizes error.
    """

    def __init__(
        self,
        extent: int = HEX_EXTENT,
        kernel_size: int = HEX_KERNEL_SIZE,
        expected_frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
    ) -> None:
        if expected_frame_shape[2] != 3:
            raise ValueError("expected_frame_shape must have three RGB channels")
        self.extent = extent
        self.kernel_size = kernel_size
        self.expected_frame_shape = expected_frame_shape
        self.receptor_centers = hex_receptor_centers(extent, kernel_size)
        self.hexals = int(self.receptor_centers.shape[0])
        self.min_frame_size = (
            self.receptor_centers.max(dim=0).values
            - self.receptor_centers.min(dim=0).values
            + 1
        )
        padding = (kernel_size - 1) / 2
        self._padding = (
            int(np.ceil(padding)),
            int(np.floor(padding)),
            int(np.ceil(padding)),
            int(np.floor(padding)),
        )
        self._kernel = torch.ones(1, 1, kernel_size, kernel_size)

    def __call__(self, gray_sequence: torch.Tensor) -> torch.Tensor:
        """Resample grayscale frames onto the hexagonal retina.

        Args:
            gray_sequence: Float tensor shaped ``(batch, time, height, width)``.

        Returns:
            Float tensor shaped ``(batch, time, 1, hexals)``.

        Raises:
            TypeError: If the input is not a float32 tensor.
            ValueError: If dimensions, frame size, or values are invalid.
        """
        if not isinstance(gray_sequence, torch.Tensor):
            raise TypeError(
                f"gray_sequence must be a torch tensor, got "
                f"{type(gray_sequence).__name__}"
            )
        if gray_sequence.ndim != 4:
            raise ValueError(
                f"Expected (batch, time, H, W), got {tuple(gray_sequence.shape)}"
            )
        if gray_sequence.dtype != torch.float32:
            raise TypeError(
                f"gray_sequence must be float32, got {gray_sequence.dtype}"
            )

        batch_size, frame_count, height, width = gray_sequence.shape
        expected_height, expected_width = self.expected_frame_shape[:2]
        if (height, width) != (expected_height, expected_width):
            raise ValueError(
                f"gray frame shape {(height, width)} != expected "
                f"{(expected_height, expected_width)}. No implicit resize is performed."
            )
        if batch_size == 0 or frame_count == 0:
            raise ValueError("batch and time dimensions must be non-empty")
        if not torch.isfinite(gray_sequence).all():
            raise ValueError("gray_sequence contains a non-finite value")
        if gray_sequence.min() < 0 or gray_sequence.max() > 1:
            raise ValueError("gray_sequence values must be in [0, 1]")

        if (self.min_frame_size > torch.tensor([height, width])).any():
            # This fixed 96-to-391 camera-to-retina projection is deliberate.
            # Only undeclared input sizes are rejected above.
            target = [int(size) for size in self.min_frame_size.tolist()]
            gray_sequence = F.interpolate(
                gray_sequence.reshape(batch_size * frame_count, 1, height, width),
                size=target,
                mode="bilinear",
                align_corners=False,
                antialias=False,
            ).reshape(batch_size, frame_count, *target)
            height, width = gray_sequence.shape[2:]

        padded = F.pad(gray_sequence, self._padding)
        kernel = self._kernel.to(padded)
        filtered = torch.cat(
            [
                F.conv2d(sample.unsqueeze(1), kernel)
                for sample in torch.unbind(padded, dim=0)
            ],
            dim=0,
        )
        filtered = filtered / self.kernel_size**2
        filtered = filtered.reshape(batch_size, frame_count, height, width)

        centers = self.receptor_centers + torch.tensor([height // 2, width // 2])
        sampled = filtered[:, :, centers[:, 0], centers[:, 1]]
        return sampled.view(batch_size, frame_count, 1, self.hexals)

    def frame(self, frame: Frame) -> torch.Tensor:
        """Convert one RGB camera frame to ``(1, 1, 1, hexals)``."""
        gray = torch.from_numpy(
            frame_to_gray(frame, expected_shape=self.expected_frame_shape)
        )
        return self(gray[None, None])
