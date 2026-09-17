"""Visual encoders and camera-to-retina transforms."""

from fly_driver.eyes.hex import (
    DEFAULT_FRAME_RATE_HZ,
    DEFAULT_FRAME_SHAPE,
    DEFAULT_VERTICAL_FOV_DEGREES,
    HEX_EXTENT,
    NUM_HEX_COLUMNS,
    HexResampler,
)
from fly_driver.eyes.stimuli import generate_drifting_grating, generate_moving_edge

__all__ = [
    "DEFAULT_FRAME_RATE_HZ",
    "DEFAULT_FRAME_SHAPE",
    "DEFAULT_VERTICAL_FOV_DEGREES",
    "HEX_EXTENT",
    "NUM_HEX_COLUMNS",
    "HexResampler",
    "generate_drifting_grating",
    "generate_moving_edge",
]
