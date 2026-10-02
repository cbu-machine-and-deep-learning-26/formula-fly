"""The fly's motion percept: T4/T5 direction channels fused into one coloured map.

Shared by the webcam demo (``scripts/flyvis_eye_live.py``, GH-46) and the in-viewer eye
view (:mod:`fly_driver.eye_view`, GH-75), so the two cannot drift apart in what a colour
means. Pure numpy: nothing here needs flyvis or torch, so it runs and is tested in the base
install.

The encoding: per hexagonal column a motion vector ``x = right - left``, ``y = up - down``;
hue is its direction (right red, up yellow-green, left cyan, down violet) and brightness its
strength relative to a running peak, so a still scene is black.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "METER_DIRECTIONS",
    "PEAK_DECAY",
    "PEAK_FLOOR",
    "SUBTYPE_DIRECTIONS",
    "T4_READOUTS",
    "T5_READOUTS",
    "RunningPeak",
    "compute_motion_percept",
    "direction_channels",
    "hsv_to_rgb",
    "hue_wheel_image",
    "summarise_motion",
]

FloatArray = npt.NDArray[np.floating[Any]]

T4_READOUTS = ("T4a", "T4b", "T4c", "T4d")
T5_READOUTS = ("T5a", "T5b", "T5c", "T5d")
METER_DIRECTIONS = ("left", "right", "up", "down")
#: Image-coordinate preferred direction of each T4/T5 subtype, the mapping the
#: direction-selectivity gate in tests/eyes asserts against the pretrained eye.
SUBTYPE_DIRECTIONS = {"a": "left", "b": "right", "c": "up", "d": "down"}
PEAK_DECAY = 0.995
PEAK_FLOOR = 0.05


def direction_channels(
    readouts: Mapping[str, FloatArray],
    resting: Mapping[str, FloatArray],
) -> dict[str, FloatArray]:
    """Sum rectified T4/T5 activity above rest into one channel per direction.

    Args:
        readouts: ``{cell type: (columns,) activity}``; T4/T5 subtypes are routed by
            :data:`SUBTYPE_DIRECTIONS`, anything else is ignored.
        resting: The same cell types at rest. Subtracted before rectifying, so a
            neuron's resting level never reads as motion.

    Returns:
        ``{direction: (columns,) non-negative}`` for every direction in
        :data:`METER_DIRECTIONS`; a direction with no subtype present is all zeros.
    """
    columns = len(next(iter(readouts.values())))
    channels = {direction: np.zeros(columns, dtype=np.float32) for direction in METER_DIRECTIONS}
    for name, activity in readouts.items():
        direction = SUBTYPE_DIRECTIONS.get(name[2:]) if name[:2] in ("T4", "T5") else None
        if direction is None:
            continue
        above_rest = np.asarray(activity, dtype=np.float32) - np.asarray(resting[name])
        channels[direction] += np.maximum(above_rest, 0.0)
    return channels


def hsv_to_rgb(hue: FloatArray, saturation: FloatArray, value: FloatArray) -> FloatArray:
    """Convert HSV arrays in ``[0, 1]`` to an RGB array with a trailing axis of 3."""
    hue6 = (np.asarray(hue) % 1.0) * 6.0
    sector = np.floor(hue6).astype(int) % 6
    fraction = hue6 - np.floor(hue6)
    low = value * (1 - saturation)
    falling = value * (1 - fraction * saturation)
    rising = value * (1 - (1 - fraction) * saturation)
    red = np.choose(sector, [value, falling, low, low, rising, value])
    green = np.choose(sector, [rising, value, value, falling, low, low])
    blue = np.choose(sector, [low, low, rising, value, value, falling])
    return np.stack([red, green, blue], axis=-1).astype(np.float32)


def compute_motion_percept(
    channels: Mapping[str, FloatArray], peak: float
) -> tuple[FloatArray, FloatArray]:
    """Fuse four direction channels into per-column motion vectors and colours.

    Per column ``x = right - left`` and ``y = up - down``; hue encodes the
    vector angle (right = red, up = yellow-green, left = cyan, down = violet)
    and brightness the magnitude relative to ``peak``, so still columns are
    black.

    Args:
        channels: ``{"left" | "right" | "up" | "down": (721,) non-negative}``.
        peak: Magnitude drawn at full brightness.

    Returns:
        ``(rgb, vectors)``: ``(721, 3)`` colours in ``[0, 1]`` and ``(721, 2)``
        ``(x, y)`` motion vectors.
    """
    x = np.asarray(channels["right"]) - np.asarray(channels["left"])
    y = np.asarray(channels["up"]) - np.asarray(channels["down"])
    magnitude = np.hypot(x, y)
    hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0
    value = np.clip(magnitude / max(peak, 1e-6), 0.0, 1.0)
    rgb = hsv_to_rgb(hue, np.ones_like(hue), value)
    return rgb, np.stack([x, y], axis=-1)


def summarise_motion(vectors: FloatArray) -> tuple[float, float]:
    """Return the mean motion vector as ``(angle_degrees, magnitude)``.

    Angles follow the hue wheel: 0 = right, 90 = up, 180 = left, 270 = down.
    """
    mean_x, mean_y = np.mean(vectors, axis=0)
    angle = float(np.degrees(np.arctan2(mean_y, mean_x)) % 360.0)
    return angle, float(np.hypot(mean_x, mean_y))


def hue_wheel_image(size: int = 64) -> FloatArray:
    """Return an ``(size, size, 4)`` RGBA hue wheel legend (transparent outside)."""
    coordinates = (np.arange(size) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(coordinates, -coordinates)
    radius = np.hypot(x, y)
    hue = (np.arctan2(y, x) / (2 * np.pi)) % 1.0
    saturation = np.clip(radius, 0.0, 1.0)
    rgb = hsv_to_rgb(hue, saturation, np.ones_like(hue))
    alpha = (radius <= 1.0).astype(np.float32)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


class RunningPeak:
    """Slowly decaying maximum used to normalise panels to their recent range."""

    def __init__(self, floor: float = PEAK_FLOOR, decay: float = PEAK_DECAY) -> None:
        self.value = floor
        self._floor = floor
        self._decay = decay

    def update(self, sample: float) -> float:
        """Fold one sample in and return the current peak."""
        self.value = max(self.value * self._decay, float(sample), self._floor)
        return self.value
