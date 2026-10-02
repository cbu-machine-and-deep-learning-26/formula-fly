"""Visual encoders and camera-to-retina transforms.

Torch and flyvis are optional. Importing this package does not import them.
:class:`CnnEye`, :class:`FlyvisEye`, :class:`ShuffledConnectomeEye` and the
hexagonal resampler load torch on first use. :class:`FlyvisEye` and
:class:`ShuffledConnectomeEye` load flyvis only when constructed.
"""

from __future__ import annotations

import importlib
from typing import Any

from fly_driver.eyes.constants import FLYVIS_DEFAULT_FEATURE_DIM
from fly_driver.eyes.degree_shuffle import degree_matched_shuffle
from fly_driver.eyes.random_projection_eye import RandomProjectionEye

__all__ = [
    "DEFAULT_MOTION_READOUTS",
    "FLYVIS_DEFAULT_FEATURE_DIM",
    "HEX_COLUMN_COUNT",
    "CnnEye",
    "FlyvisEye",
    "FlyvisNotInstalledError",
    "HexResampler",
    "PixelEye",
    "RandomProjectionEye",
    "ShuffledConnectomeEye",
    "TorchNotInstalledError",
    "degree_matched_shuffle",
    "frame_to_gray",
    "hex_coordinates",
    "hex_receptor_centers",
    "resolve_checkpoint_dir",
]

# Imported on first access. Each of these modules imports torch at module
# scope; keeping them off the eager path is what lets the base install import
# the package, and run the random-projection and shuffle tests, without torch.
_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_MOTION_READOUTS": ("fly_driver.eyes.flyvis_eye", "DEFAULT_MOTION_READOUTS"),
    "HEX_COLUMN_COUNT": ("fly_driver.eyes.hex_resampler", "HEX_COLUMN_COUNT"),
    "CnnEye": ("fly_driver.eyes.cnn_eye", "CnnEye"),
    "FlyvisEye": ("fly_driver.eyes.flyvis_eye", "FlyvisEye"),
    "FlyvisNotInstalledError": ("fly_driver.eyes.flyvis_eye", "FlyvisNotInstalledError"),
    "HexResampler": ("fly_driver.eyes.hex_resampler", "HexResampler"),
    "PixelEye": ("fly_driver.eyes.pixel_eye", "PixelEye"),
    "ShuffledConnectomeEye": ("fly_driver.eyes.shuffled_connectome_eye", "ShuffledConnectomeEye"),
    "TorchNotInstalledError": ("fly_driver.eyes.cnn_eye", "TorchNotInstalledError"),
    "frame_to_gray": ("fly_driver.eyes.hex_resampler", "frame_to_gray"),
    "hex_coordinates": ("fly_driver.eyes.hex_resampler", "hex_coordinates"),
    "hex_receptor_centers": ("fly_driver.eyes.hex_resampler", "hex_receptor_centers"),
    "resolve_checkpoint_dir": ("fly_driver.eyes.flyvis_eye", "resolve_checkpoint_dir"),
}


def __getattr__(name: str) -> Any:
    """Load a torch-backed export the first time something asks for it."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = importlib.import_module(module_name)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include the lazy names so they tab-complete before they are imported."""
    return sorted(set(globals()) | set(__all__))
