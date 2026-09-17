"""Map Cartesian camera images onto flyvis's 721-column hexagonal retina.

The axial coordinates and their order reproduce ``flyvis.utils.hex_utils``:
``u`` is traversed from -15 to +15 and valid ``v`` values are traversed in
ascending order.  Camera coordinates use flyvis's rendering convention
(``x = v``, ``y = u + v / 2``), with positive image ``y`` pointing down.  See
``flyvis.datasets.rendering.utils.hex_center_coordinates`` and the hex-grid
reference linked by flyvis:
https://www.redblobgames.com/grids/hexagons/#range-coordinate.

Each receptor center is projected through the configured pinhole-camera field
of view.  Bilinear source indices and weights are computed at construction, so
resampling only gathers pixels and applies those fixed weights.  A sequence
contains one camera frame per flyvis integration step and maps from
``(T, H, W, 3)`` to ``(T, 721)`` without temporal interpolation.  The default
50 Hz therefore requires flyvis ``dt=0.02`` seconds; callers using another
camera rate must use the matching ``dt``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt

ChannelChoice = Literal["grayscale", "green"]

DEFAULT_FRAME_SHAPE = (96, 96, 3)
DEFAULT_FRAME_RATE_HZ = 50.0
DEFAULT_VERTICAL_FOV_DEGREES = 75.0
HEX_EXTENT = 15
NUM_HEX_COLUMNS = 1 + 3 * HEX_EXTENT * (HEX_EXTENT + 1)

_GRAYSCALE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _validate_frame_shape(frame_shape: tuple[int, int, int]) -> None:
    if (
        not isinstance(frame_shape, tuple)
        or len(frame_shape) != 3
        or any(
            isinstance(size, bool) or not isinstance(size, int) for size in frame_shape
        )
    ):
        raise TypeError("frame_shape must be a tuple of three integers")
    height, width, channels = frame_shape
    if height < 2 or width < 2:
        raise ValueError(
            f"frame height and width must be at least 2, got {frame_shape}"
        )
    if channels != 3:
        raise ValueError(f"frame_shape must have three RGB channels, got {frame_shape}")


def _validate_positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite, got {result!r}")
    return result


def _generate_axial_coordinates(
    extent: int,
) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]:
    coordinates = [
        (u_coordinate, v_coordinate)
        for u_coordinate in range(-extent, extent + 1)
        for v_coordinate in range(
            max(-extent, -extent - u_coordinate),
            min(extent, extent - u_coordinate) + 1,
        )
    ]
    coordinate_array = np.asarray(coordinates, dtype=np.int16)
    return coordinate_array[:, 0], coordinate_array[:, 1]


class HexResampler:
    """Resample RGB camera frames onto the extent-15 flyvis lattice.

    Args:
        frame_shape: Exact accepted ``(height, width, 3)`` camera shape.
        vertical_fov_degrees: Pinhole camera vertical field of view.  The
            lattice spans this full angle vertically; its horizontal span is
            derived from the frame aspect ratio.
        channel: ``"green"`` for the fly-relevant green channel, or
            ``"grayscale"`` for BT.601 luma.
        frame_rate_hz: Camera sampling rate.  No frames are inserted or
            dropped; flyvis must use ``dt = 1 / frame_rate_hz``.

    Raises:
        TypeError: If a configuration value has the wrong type.
        ValueError: If a configuration value is outside its valid range.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
        *,
        vertical_fov_degrees: float = DEFAULT_VERTICAL_FOV_DEGREES,
        channel: ChannelChoice = "green",
        frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    ) -> None:
        _validate_frame_shape(frame_shape)
        vertical_fov = _validate_positive_finite(
            vertical_fov_degrees, "vertical_fov_degrees"
        )
        if vertical_fov >= 180:
            raise ValueError(
                f"vertical_fov_degrees must be less than 180, got {vertical_fov!r}"
            )
        if channel not in ("grayscale", "green"):
            raise ValueError(f"channel must be 'grayscale' or 'green', got {channel!r}")

        self.frame_shape = frame_shape
        self.vertical_fov_degrees = vertical_fov
        self.channel: ChannelChoice = channel
        self.frame_rate_hz = _validate_positive_finite(frame_rate_hz, "frame_rate_hz")
        self.u_coordinates, self.v_coordinates = _generate_axial_coordinates(HEX_EXTENT)
        self.pixel_coordinates = self._calculate_pixel_coordinates()
        self.source_indices, self.source_weights = self._calculate_bilinear_geometry()
        self._coordinate_indices = {
            (int(u_coordinate), int(v_coordinate)): index
            for index, (u_coordinate, v_coordinate) in enumerate(
                zip(self.u_coordinates, self.v_coordinates, strict=True)
            )
        }
        for array in (
            self.u_coordinates,
            self.v_coordinates,
            self.pixel_coordinates,
            self.source_indices,
            self.source_weights,
        ):
            array.setflags(write=False)

    @property
    def frame_interval_seconds(self) -> float:
        """Return the flyvis integration interval implied by the camera rate."""
        return 1.0 / self.frame_rate_hz

    @property
    def horizontal_fov_degrees(self) -> float:
        """Return the horizontal FOV implied by vertical FOV and aspect ratio."""
        height, width, _ = self.frame_shape
        vertical_half_angle = np.radians(self.vertical_fov_degrees / 2.0)
        horizontal_half_angle = np.arctan(np.tan(vertical_half_angle) * width / height)
        return float(np.degrees(2.0 * horizontal_half_angle))

    def column_index(self, u_coordinate: int, v_coordinate: int) -> int:
        """Return the flyvis column index for an axial coordinate.

        Args:
            u_coordinate: Flyvis axial ``u`` coordinate.
            v_coordinate: Flyvis axial ``v`` coordinate.

        Raises:
            KeyError: If the coordinate is outside the extent-15 lattice.
        """
        try:
            return self._coordinate_indices[(u_coordinate, v_coordinate)]
        except KeyError as error:
            raise KeyError(
                f"({u_coordinate}, {v_coordinate}) is outside extent {HEX_EXTENT}"
            ) from error

    def resample_frame(self, frame: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Convert one RGB frame to normalized flyvis receptor intensities.

        Args:
            frame: RGB uint8 array with exactly the configured shape.

        Returns:
            Float32 values in ``[0, 1]`` with shape ``(721,)``.

        Raises:
            TypeError: If the frame is not a uint8 NumPy array.
            ValueError: If the frame shape differs from the configured shape.
        """
        self._validate_frame(frame)
        gathered_rgb = frame.reshape(-1, 3)[self.source_indices]
        gathered_values = self._select_channel(gathered_rgb)
        result = np.sum(
            gathered_values * self.source_weights,
            axis=1,
            dtype=np.float32,
        )
        return result / np.float32(255.0)

    def resample_sequence(
        self, frames: npt.NDArray[np.uint8]
    ) -> npt.NDArray[np.float32]:
        """Convert one frame-rate-preserving RGB sequence to ``(T, 721)``.

        Args:
            frames: RGB uint8 array with shape ``(T, H, W, 3)``.

        Returns:
            Float32 flyvis input with one row per camera frame.  Add batch and
            singleton channel axes for flyvis network input:
            ``values[None, :, None, :]``.

        Raises:
            TypeError: If frames is not a uint8 NumPy array.
            ValueError: If its spatial shape differs or it has no frames.
        """
        if not isinstance(frames, np.ndarray):
            raise TypeError(
                f"frames must be a numpy array, got {type(frames).__name__}"
            )
        if frames.dtype != np.uint8:
            raise TypeError(f"frames must be uint8, got {frames.dtype}")
        expected_trailing_shape = self.frame_shape
        if frames.ndim != 4 or frames.shape[1:] != expected_trailing_shape:
            raise ValueError(
                "frames must have shape (T, "
                f"{expected_trailing_shape[0]}, {expected_trailing_shape[1]}, 3), "
                f"got {frames.shape}; no implicit resize is performed"
            )
        if frames.shape[0] == 0:
            raise ValueError("frames must contain at least one camera frame")

        flat_frames = frames.reshape(frames.shape[0], -1, 3)
        gathered_rgb = flat_frames[:, self.source_indices, :]
        gathered_values = self._select_channel(gathered_rgb)
        result = np.sum(
            gathered_values * self.source_weights[None, :, :],
            axis=2,
            dtype=np.float32,
        )
        return result / np.float32(255.0)

    def _calculate_pixel_coordinates(self) -> npt.NDArray[np.float64]:
        height, width, _ = self.frame_shape
        normalized_x = self.v_coordinates.astype(np.float64) / HEX_EXTENT
        normalized_y = (
            self.u_coordinates.astype(np.float64)
            + self.v_coordinates.astype(np.float64) / 2.0
        ) / HEX_EXTENT

        vertical_half_angle = np.radians(self.vertical_fov_degrees / 2.0)
        horizontal_half_angle = np.radians(self.horizontal_fov_degrees / 2.0)
        ray_x = np.tan(normalized_x * horizontal_half_angle) / np.tan(
            horizontal_half_angle
        )
        ray_y = np.tan(normalized_y * vertical_half_angle) / np.tan(vertical_half_angle)
        columns = (ray_x + 1.0) * (width - 1) / 2.0
        rows = (ray_y + 1.0) * (height - 1) / 2.0
        return np.column_stack((rows, columns))

    def _calculate_bilinear_geometry(
        self,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32]]:
        height, width, _ = self.frame_shape
        rows = np.clip(self.pixel_coordinates[:, 0], 0.0, height - 1)
        columns = np.clip(self.pixel_coordinates[:, 1], 0.0, width - 1)
        row_low = np.floor(rows).astype(np.int64)
        column_low = np.floor(columns).astype(np.int64)
        row_high = np.minimum(row_low + 1, height - 1)
        column_high = np.minimum(column_low + 1, width - 1)
        row_fraction = rows - row_low
        column_fraction = columns - column_low

        source_indices = np.column_stack(
            (
                row_low * width + column_low,
                row_low * width + column_high,
                row_high * width + column_low,
                row_high * width + column_high,
            )
        )
        source_weights = np.column_stack(
            (
                (1.0 - row_fraction) * (1.0 - column_fraction),
                (1.0 - row_fraction) * column_fraction,
                row_fraction * (1.0 - column_fraction),
                row_fraction * column_fraction,
            )
        ).astype(np.float32)
        return source_indices, source_weights

    def _select_channel(self, gathered_rgb: npt.NDArray[np.uint8]) -> np.ndarray:
        if self.channel == "green":
            return gathered_rgb[..., 1].astype(np.float32)
        return np.einsum(
            "...c,c->...",
            gathered_rgb,
            _GRAYSCALE_WEIGHTS,
            dtype=np.float32,
        )

    def _validate_frame(self, frame: object) -> None:
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be a numpy array, got {type(frame).__name__}")
        if frame.dtype != np.uint8:
            raise TypeError(
                f"frame must be uint8 in [0, 255], got {frame.dtype}; "
                "no implicit cast is performed"
            )
        if frame.shape != self.frame_shape:
            raise ValueError(
                f"frame shape {frame.shape} != expected {self.frame_shape}; "
                "no implicit resize is performed"
            )
