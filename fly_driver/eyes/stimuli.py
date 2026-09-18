"""Synthetic camera stimuli for checking the eye pipeline.

These generate uint8 RGB frame sequences in the camera contract, so they exercise
the full path (frame validation, luminance, hexagonal resampling, optic lobe)
rather than hand-placed hexal values. The drifting grating gates T4/T5 direction
selectivity; the moving edge is the readable demo stimulus. Both start with a
grey baseline so the network can be compared against its resting state.

Directions are named in image coordinates: ``"right"`` moves toward higher
column indices and ``"down"`` toward higher row indices.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fly_driver.eyes.hex_resampler import DEFAULT_FRAME_SHAPE

__all__ = [
    "DIRECTIONS",
    "FRAME_RATE_HZ",
    "GREY_LEVEL",
    "drifting_grating_frames",
    "grey_frames",
    "moving_edge_frames",
]

FRAME_RATE_HZ = 50.0
GREY_LEVEL = 128
DIRECTIONS = ("right", "left", "down", "up")

FrameSequence = npt.NDArray[np.uint8]


def _validate_direction(direction: str) -> None:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")


def _rgb_from_luminance(luminance: npt.NDArray[np.uint8]) -> FrameSequence:
    return np.repeat(luminance[..., None], 3, axis=-1)


def grey_frames(
    frame_count: int,
    *,
    frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
    grey_level: int = GREY_LEVEL,
) -> FrameSequence:
    """Return a uniform grey ``(frame_count, height, width, 3)`` uint8 sequence.

    Args:
        frame_count: Number of frames.
        frame_shape: Camera frame shape ``(height, width, 3)``.
        grey_level: uint8 luminance of every pixel.

    Returns:
        A uint8 array of constant frames.
    """
    if frame_count < 0:
        raise ValueError("frame_count must be non-negative")
    return np.full((frame_count, *frame_shape), grey_level, dtype=np.uint8)


def drifting_grating_frames(
    direction: str,
    *,
    prestimulus_frames: int = 25,
    drift_frames: int = 50,
    period_pixels: int = 24,
    cycles_per_second: float = 2.0,
    frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
    frame_rate_hz: float = FRAME_RATE_HZ,
) -> FrameSequence:
    """Return grey baseline frames followed by a full-contrast drifting sinusoid.

    With the defaults this is 0.5 s of grey and 1 s of grating at 2 cycles/s,
    the stimulus behind the T4/T5 direction-selectivity gate.

    Args:
        direction: One of :data:`DIRECTIONS`, in image coordinates.
        prestimulus_frames: Grey frames before the grating appears.
        drift_frames: Frames of drifting grating.
        period_pixels: Spatial period of the grating in pixels.
        cycles_per_second: Temporal frequency of the drift.
        frame_shape: Camera frame shape ``(height, width, 3)``.
        frame_rate_hz: Frame rate used to convert frame index to time.

    Returns:
        A uint8 ``(prestimulus_frames + drift_frames, height, width, 3)`` array.
    """
    _validate_direction(direction)
    height, width, _ = frame_shape
    row_grid, column_grid = np.mgrid[:height, :width]
    direction_sign = 1 if direction in ("right", "down") else -1
    coordinate_grid = column_grid if direction in ("right", "left") else row_grid

    frame_indices = np.arange(drift_frames)[:, None, None]
    time_seconds = frame_indices / frame_rate_hz
    phase = (
        2
        * np.pi
        * (
            coordinate_grid[None] / period_pixels
            - direction_sign * cycles_per_second * time_seconds
        )
    )
    luminance = np.rint(127.5 + 127.5 * np.sin(phase)).astype(np.uint8)
    baseline = grey_frames(prestimulus_frames, frame_shape=frame_shape)
    return np.concatenate([baseline, _rgb_from_luminance(luminance)], axis=0)


def moving_edge_frames(
    direction: str,
    *,
    prestimulus_frames: int = 25,
    sweep_frames: int = 50,
    hold_frames: int = 10,
    frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
) -> FrameSequence:
    """Return a grey baseline, a bright edge sweeping across the frame, and a hold.

    At the first sweep frame the field steps from grey to dark except for the
    leading column (or row); the bright region then grows in ``direction`` until
    the whole frame is bright. With the defaults this is 0.5 s of grey, a 1 s
    sweep, and a 0.2 s hold at 50 Hz, matching ``scripts/flyvis_smoke.py``.

    Args:
        direction: One of :data:`DIRECTIONS`, in image coordinates.
        prestimulus_frames: Grey frames before the sweep.
        sweep_frames: Frames the edge takes to cross the frame.
        hold_frames: Frames holding the final bright field.
        frame_shape: Camera frame shape ``(height, width, 3)``.

    Returns:
        A uint8 ``(prestimulus_frames + sweep_frames + hold_frames, height, width, 3)``
        array.
    """
    _validate_direction(direction)
    if sweep_frames < 2:
        raise ValueError("sweep_frames must be at least 2")
    height, width, _ = frame_shape
    row_grid, column_grid = np.mgrid[:height, :width]
    coordinate_grid = column_grid if direction in ("right", "left") else row_grid
    extent = float(coordinate_grid.max())

    thresholds = np.linspace(0.0, extent, sweep_frames)[:, None, None]
    if direction in ("right", "down"):
        is_bright = coordinate_grid[None] <= thresholds
    else:
        is_bright = coordinate_grid[None] >= extent - thresholds
    sweep = np.where(is_bright, 255, 0).astype(np.uint8)
    hold = np.repeat(sweep[-1:], hold_frames, axis=0)
    baseline = grey_frames(prestimulus_frames, frame_shape=frame_shape)
    return np.concatenate(
        [baseline, _rgb_from_luminance(sweep), _rgb_from_luminance(hold)], axis=0
    )
