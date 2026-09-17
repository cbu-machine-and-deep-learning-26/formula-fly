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
from torch.nn import functional as F  # noqa: N812

__all__ = [
    "DEFAULT_FRAME_SHAPE",
    "HEX_COLUMN_COUNT",
    "HEX_EXTENT",
    "HEX_KERNEL_SIZE",
    "HexResampler",
    "frame_to_gray",
    "hex_receptor_centers",
]

DEFAULT_FRAME_SHAPE = (96, 96, 3)
HEX_EXTENT = 15
HEX_KERNEL_SIZE = 13
HEX_COLUMN_COUNT = 721
RGB_LUMA_WEIGHTS = (0.299, 0.587, 0.114)

Frame = npt.NDArray[np.uint8]


def frame_to_gray(frame: Frame) -> npt.NDArray[np.float32]:
    """Convert a 96x96 uint8 RGB frame to grayscale float32 in ``[0, 1]``.

    Args:
        frame: Camera frame with shape ``(96, 96, 3)``.

    Returns:
        A ``(96, 96)`` grayscale array.

    Raises:
        ValueError: If the frame has an unexpected shape or dtype.
    """
    if frame.shape != DEFAULT_FRAME_SHAPE:
        raise ValueError(
            f"Expected frame shape {DEFAULT_FRAME_SHAPE}, got {frame.shape}"
        )
    if frame.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frame, got {frame.dtype}")

    array = frame.astype(np.float32) / 255.0
    weights = np.asarray(RGB_LUMA_WEIGHTS, dtype=np.float32)
    return (array @ weights).astype(np.float32)


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
    centers: list[tuple[float, float]] = []
    for u_coordinate in range(-extent, extent + 1):
        v_min = max(-extent, -extent - u_coordinate)
        v_max = min(extent, extent - u_coordinate)
        for v_coordinate in range(v_min, v_max + 1):
            centers.append(
                (
                    kernel_size * (u_coordinate + v_coordinate / 2),
                    kernel_size * v_coordinate,
                )
            )
    return torch.tensor(centers, dtype=torch.long)


class HexResampler:
    """Reproduce flyvis ``BoxEye`` mean filtering and hexagonal sampling.

    Args:
        extent: Hexagonal lattice radius in receptors.
        kernel_size: Mean-filter size and receptor spacing in pixels.
    """

    def __init__(
        self, extent: int = HEX_EXTENT, kernel_size: int = HEX_KERNEL_SIZE
    ) -> None:
        self.extent = extent
        self.kernel_size = kernel_size
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
            ValueError: If the input does not have four dimensions.
        """
        if gray_sequence.ndim != 4:
            raise ValueError(
                f"Expected (batch, time, H, W), got {tuple(gray_sequence.shape)}"
            )

        batch_size, frame_count, height, width = gray_sequence.shape
        if (self.min_frame_size > torch.tensor([height, width])).any():
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
        gray = torch.from_numpy(frame_to_gray(frame))
        return self(gray[None, None])
