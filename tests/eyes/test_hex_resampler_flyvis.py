"""Pretrained flyvis gate for camera-to-retina geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from fly_driver.eyes.hex_resampler import (
    HEX_COLUMN_COUNT,
    HexResampler,
    frame_to_gray,
)
from fly_driver.eyes.stimuli import DIRECTIONS, drifting_grating_frames
from tests.eyes.direction_selectivity import (
    DRIFT_FRAMES,
    PRESTIMULUS_FRAMES,
    assert_t4_t5_direction_selectivity,
    measure_q95_amplitudes,
)

flyvis = pytest.importorskip("flyvis")

FRAME_INTERVAL_SECONDS = 1 / 50


def _create_grating_sequence(direction: str) -> torch.Tensor:
    """Resample the shared RGB drifting grating onto the flyvis retina."""
    frames = drifting_grating_frames(
        direction, prestimulus_frames=PRESTIMULUS_FRAMES, drift_frames=DRIFT_FRAMES
    )
    grayscale_frames = np.stack([frame_to_gray(frame) for frame in frames])
    return HexResampler()(torch.from_numpy(grayscale_frames)[None])[0]


def test_pretrained_t4_t5_direction_selectivity() -> None:
    """Opposite T4/T5 subtype channels must prefer opposite grating motion."""
    checkpoint_dir = Path(flyvis.results_dir) / "flow" / "0000" / "000"
    if not checkpoint_dir.is_dir():
        pytest.skip("run `flyvis download-pretrained` to enable this gate")

    sequences = torch.stack([_create_grating_sequence(direction) for direction in DIRECTIONS])
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

    def readout_response(direction: str, readout: str) -> np.ndarray:
        return responses[readout][DIRECTIONS.index(direction)].cpu().numpy()

    assert_t4_t5_direction_selectivity(measure_q95_amplitudes(readout_response))
