"""Deterministic visual stimuli for fly motion-direction validation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fly_driver.eyes.hex import DEFAULT_FRAME_RATE_HZ, DEFAULT_FRAME_SHAPE


def _validate_positive_finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite, got {result!r}")
    return result


def _validate_stimulus_parameters(
    frame_shape: tuple[int, int, int],
    num_frames: int,
    frame_rate_hz: float,
) -> float:
    if (
        not isinstance(frame_shape, tuple)
        or len(frame_shape) != 3
        or any(
            isinstance(size, bool) or not isinstance(size, int) for size in frame_shape
        )
    ):
        raise TypeError("frame_shape must be a tuple of three integers")
    height, width, channels = frame_shape
    if height < 2 or width < 2 or channels != 3:
        raise ValueError(
            f"frame_shape must be (height >= 2, width >= 2, 3), got {frame_shape}"
        )
    if isinstance(num_frames, bool) or not isinstance(num_frames, int):
        raise TypeError("num_frames must be an integer")
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    return _validate_positive_finite(frame_rate_hz, "frame_rate_hz")


def generate_drifting_grating(
    frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
    *,
    num_frames: int = 50,
    spatial_period_pixels: float = 16.0,
    temporal_frequency_hz: float = 5.0,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    direction_degrees: float = 0.0,
    contrast: float = 1.0,
    mean_intensity: float = 0.5,
) -> npt.NDArray[np.uint8]:
    """Generate a deterministic RGB sinusoidal drifting grating.

    Args:
        frame_shape: Output ``(height, width, 3)`` frame shape.
        num_frames: Number of frames in the sequence.
        spatial_period_pixels: Distance between luminance peaks.
        temporal_frequency_hz: Grating cycles per second.
        frame_rate_hz: Frames per second.
        direction_degrees: Motion direction in image coordinates: 0 is right,
            90 is down, 180 is left, and 270 is up.
        contrast: Michelson contrast in ``[0, 1]``.
        mean_intensity: Mean luminance in ``[0, 1]``.

    Returns:
        RGB uint8 sequence with shape ``(T, H, W, 3)``.
    """
    validated_frame_rate = _validate_stimulus_parameters(
        frame_shape, num_frames, frame_rate_hz
    )
    period = _validate_positive_finite(spatial_period_pixels, "spatial_period_pixels")
    temporal_frequency = _validate_positive_finite(
        temporal_frequency_hz, "temporal_frequency_hz"
    )
    if not np.isfinite(direction_degrees):
        raise ValueError("direction_degrees must be finite")
    if not 0.0 <= contrast <= 1.0:
        raise ValueError(f"contrast must be in [0, 1], got {contrast!r}")
    if not 0.0 <= mean_intensity <= 1.0:
        raise ValueError(f"mean_intensity must be in [0, 1], got {mean_intensity!r}")
    amplitude = contrast * min(mean_intensity, 1.0 - mean_intensity)

    height, width, _ = frame_shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    angle_radians = np.radians(direction_degrees)
    projected_position = columns * np.cos(angle_radians) + rows * np.sin(angle_radians)
    displacement_per_frame = temporal_frequency * period / validated_frame_rate
    displacements = np.arange(num_frames, dtype=np.float64) * displacement_per_frame
    wrapped_positions = np.remainder(
        projected_position[None, :, :] - displacements[:, None, None],
        period,
    )
    phases = 2.0 * np.pi * wrapped_positions / period
    luminance = mean_intensity + amplitude * np.cos(phases)
    grayscale = np.rint(np.clip(luminance, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(grayscale[:, :, :, None], 3, axis=3)


def generate_moving_edge(
    frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
    *,
    num_frames: int = 50,
    speed_pixels_per_second: float = 40.0,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    direction_degrees: float = 0.0,
    start_position_pixels: float | None = None,
    dark_intensity: int = 0,
    bright_intensity: int = 255,
) -> npt.NDArray[np.uint8]:
    """Generate a deterministic hard edge moving across RGB frames.

    Args:
        frame_shape: Output ``(height, width, 3)`` frame shape.
        num_frames: Number of frames in the sequence.
        speed_pixels_per_second: Edge velocity along ``direction_degrees``.
        frame_rate_hz: Frames per second.
        direction_degrees: Motion direction in image coordinates: 0 is right
            and 90 is down.
        start_position_pixels: Initial edge position relative to frame center.
            Defaults to the negative half-diagonal.
        dark_intensity: uint8 intensity ahead of the moving edge.
        bright_intensity: uint8 intensity behind the moving edge.

    Returns:
        RGB uint8 sequence with shape ``(T, H, W, 3)``.
    """
    validated_frame_rate = _validate_stimulus_parameters(
        frame_shape, num_frames, frame_rate_hz
    )
    speed = _validate_positive_finite(
        speed_pixels_per_second, "speed_pixels_per_second"
    )
    if not np.isfinite(direction_degrees):
        raise ValueError("direction_degrees must be finite")
    for intensity, name in (
        (dark_intensity, "dark_intensity"),
        (bright_intensity, "bright_intensity"),
    ):
        if (
            isinstance(intensity, bool)
            or not isinstance(intensity, (int, np.integer))
            or not 0 <= intensity <= 255
        ):
            raise ValueError(f"{name} must be an integer in [0, 255]")

    height, width, _ = frame_shape
    rows, columns = np.indices((height, width), dtype=np.float64)
    centered_rows = rows - (height - 1) / 2.0
    centered_columns = columns - (width - 1) / 2.0
    angle_radians = np.radians(direction_degrees)
    projected_position = centered_columns * np.cos(
        angle_radians
    ) + centered_rows * np.sin(angle_radians)
    if start_position_pixels is None:
        initial_position = -float(np.hypot(height, width)) / 2.0
    else:
        if not np.isfinite(start_position_pixels):
            raise ValueError("start_position_pixels must be finite")
        initial_position = float(start_position_pixels)
    frame_times = np.arange(num_frames, dtype=np.float64) / validated_frame_rate
    edge_positions = initial_position + speed * frame_times
    is_behind_edge = projected_position[None, :, :] <= edge_positions[:, None, None]
    grayscale = np.where(is_behind_edge, bright_intensity, dark_intensity).astype(
        np.uint8
    )
    return np.repeat(grayscale[:, :, :, None], 3, axis=3)
