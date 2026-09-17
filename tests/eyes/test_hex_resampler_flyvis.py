"""Pretrained flyvis gate for camera-to-retina geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from fly_driver.eyes.hex_resampler import (
    DEFAULT_FRAME_SHAPE,
    HEX_COLUMN_COUNT,
    HexResampler,
    frame_to_gray,
)

flyvis = pytest.importorskip("flyvis")

FRAME_INTERVAL_SECONDS = 1 / 50
PRESTIMULUS_FRAMES = 25
DRIFT_FRAMES = 50
SETTLING_DRIFT_FRAMES = 10
GRATING_PERIOD_PIXELS = 24
GRATING_CYCLES_PER_SECOND = 2
DIRECTIONS = ("right", "left", "down", "up")
MOTION_READOUTS = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def _create_grating_sequence(direction: str) -> torch.Tensor:
    """Create gray prestimulus plus one second of RGB drifting grating."""
    height, width, _ = DEFAULT_FRAME_SHAPE
    row_grid, column_grid = np.mgrid[:height, :width]
    frames = [
        np.full(DEFAULT_FRAME_SHAPE, 128, dtype=np.uint8)
        for _ in range(PRESTIMULUS_FRAMES)
    ]
    direction_sign = 1 if direction in ("right", "down") else -1
    coordinate_grid = column_grid if direction in ("right", "left") else row_grid

    for frame_index in range(DRIFT_FRAMES):
        time_seconds = frame_index * FRAME_INTERVAL_SECONDS
        phase = (
            2
            * np.pi
            * (
                coordinate_grid / GRATING_PERIOD_PIXELS
                - direction_sign * GRATING_CYCLES_PER_SECOND * time_seconds
            )
        )
        luminance = np.rint(127.5 + 127.5 * np.sin(phase)).astype(np.uint8)
        frames.append(np.repeat(luminance[..., None], 3, axis=-1))

    grayscale_frames = np.stack([frame_to_gray(frame) for frame in frames])
    return HexResampler()(torch.from_numpy(grayscale_frames)[None])[0]


def _measure_q95_amplitudes(responses: Any) -> dict[str, dict[str, float]]:
    """Measure a robust near-peak response for every direction and readout."""
    amplitudes: dict[str, dict[str, float]] = {}
    response_start = PRESTIMULUS_FRAMES + SETTLING_DRIFT_FRAMES
    for direction_index, direction in enumerate(DIRECTIONS):
        amplitudes[direction] = {}
        for readout in MOTION_READOUTS:
            motion_response = responses[readout][direction_index, response_start:]
            amplitudes[direction][readout] = float(
                torch.quantile(motion_response, 0.95)
            )
    return amplitudes


def _combine_t4_t5(
    amplitudes: dict[str, dict[str, float]], direction: str, subtype: str
) -> float:
    """Average ON-pathway T4 and OFF-pathway T5 amplitudes for one subtype."""
    return (
        amplitudes[direction][f"T4{subtype}"] + amplitudes[direction][f"T5{subtype}"]
    ) / 2


def test_pretrained_t4_t5_direction_selectivity() -> None:
    """Opposite T4/T5 subtype channels must prefer opposite grating motion."""
    checkpoint_dir = Path(flyvis.results_dir) / "flow" / "0000" / "000"
    if not checkpoint_dir.is_dir():
        pytest.skip("run `flyvis download-pretrained` to enable this gate")

    sequences = torch.stack(
        [_create_grating_sequence(direction) for direction in DIRECTIONS]
    )
    assert sequences.shape == (
        len(DIRECTIONS),
        PRESTIMULUS_FRAMES + DRIFT_FRAMES,
        1,
        HEX_COLUMN_COUNT,
    )
    assert sequences.dtype == torch.float32

    network = flyvis.NetworkView(checkpoint_dir).init_network()
    network.eval()
    network.requires_grad_(False)
    with torch.no_grad():
        responses = network.simulate(
            sequences,
            dt=FRAME_INTERVAL_SECONDS,
            as_layer_activity=True,
        )

    amplitudes = _measure_q95_amplitudes(responses)
    minimum_preference_ratio = 1.25
    expected_preferences = (
        ("a", "left", "right"),
        ("b", "right", "left"),
        ("c", "up", "down"),
        ("d", "down", "up"),
    )
    for subtype, preferred_direction, opposite_direction in expected_preferences:
        preferred_response = _combine_t4_t5(amplitudes, preferred_direction, subtype)
        opposite_response = _combine_t4_t5(amplitudes, opposite_direction, subtype)
        assert preferred_response > minimum_preference_ratio * opposite_response, (
            f"T4/T5{subtype} q95 response {preferred_response:.4f} for "
            f"{preferred_direction} did not exceed {opposite_direction} "
            f"{opposite_response:.4f} by {minimum_preference_ratio:.2f}x; "
            f"individual amplitudes: {amplitudes}"
        )
