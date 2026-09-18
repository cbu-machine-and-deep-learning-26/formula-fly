"""Tests for the synthetic camera stimuli."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.eyes.hex_resampler import DEFAULT_FRAME_SHAPE
from fly_driver.eyes.stimuli import (
    DIRECTIONS,
    GREY_LEVEL,
    drifting_grating_frames,
    grey_frames,
    moving_edge_frames,
)


def test_grey_frames_are_uniform_uint8() -> None:
    """Fill every pixel of every frame with the grey level."""
    frames = grey_frames(3)

    assert frames.shape == (3, *DEFAULT_FRAME_SHAPE)
    assert frames.dtype == np.uint8
    assert np.all(frames == GREY_LEVEL)


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_grating_has_grey_baseline_then_full_contrast_drift(direction: str) -> None:
    """Start with grey frames and drift the sinusoid in the requested direction."""
    frames = drifting_grating_frames(direction, prestimulus_frames=5, drift_frames=10)

    assert frames.shape == (15, *DEFAULT_FRAME_SHAPE)
    assert frames.dtype == np.uint8
    assert np.all(frames[:5] == GREY_LEVEL)
    drift = frames[5:, :, :, 0]
    assert drift.min() == 0 and drift.max() == 255
    assert np.array_equal(frames[..., 0], frames[..., 1])
    assert np.array_equal(frames[..., 0], frames[..., 2])

    # A quarter of a 2 Hz cycle at 50 Hz is 6.25 frames; over 5 frames the pattern
    # should have shifted by roughly period * 2 Hz * 0.1 s = 4.8 pixels.
    first, later = drift[0].astype(int), drift[5].astype(int)
    axis = 1 if direction in ("right", "left") else 0
    shift = 5 if direction in ("right", "down") else -5
    shifted = np.roll(first, shift, axis=axis)
    unshifted_error = np.abs(later - first).mean()
    shifted_error = np.abs(later - shifted).mean()
    assert shifted_error < 0.25 * unshifted_error


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_moving_edge_sweeps_monotonically_and_holds(direction: str) -> None:
    """Grow the bright region in the requested direction, then hold it."""
    frames = moving_edge_frames(direction, prestimulus_frames=4, sweep_frames=20, hold_frames=3)

    assert frames.shape == (27, *DEFAULT_FRAME_SHAPE)
    assert frames.dtype == np.uint8
    assert np.all(frames[:4] == GREY_LEVEL)
    sweep = frames[4:24, :, :, 0]
    bright_counts = (sweep == 255).sum(axis=(1, 2))
    assert np.all(np.diff(bright_counts) > 0)
    assert bright_counts[-1] == DEFAULT_FRAME_SHAPE[0] * DEFAULT_FRAME_SHAPE[1]
    assert np.all(frames[24:] == 255)

    midway = sweep[10]
    height, width = midway.shape
    if direction == "right":
        assert midway[:, 0].min() == 255 and midway[:, width - 1].max() == 0
    elif direction == "left":
        assert midway[:, width - 1].min() == 255 and midway[:, 0].max() == 0
    elif direction == "down":
        assert midway[0].min() == 255 and midway[height - 1].max() == 0
    else:
        assert midway[height - 1].min() == 255 and midway[0].max() == 0


def test_direction_names_are_validated() -> None:
    """Reject directions outside the image-coordinate vocabulary."""
    with pytest.raises(ValueError, match="direction"):
        drifting_grating_frames("ltr")
    with pytest.raises(ValueError, match="direction"):
        moving_edge_frames("rtl")
