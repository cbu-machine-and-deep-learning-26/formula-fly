"""Square RGB frame → 721 hexagonal columns (flyvis lattice size).

Column count is the centered hexagonal number 3 n (n + 1) + 1 for radius n = 15.
Orientation: pointy-top axial coordinates (q, r). Chirality: +q is +x (east);
+r is 60 degrees toward +y.

This module does not import flyvis. T4/T5 preferred-direction checks stay behind
the optional flyvis extra (see tests/test_hex_resampler.py).
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from fly_driver.interface import DEFAULT_FRAME_SHAPE, FrameArray, require_frame

HEX_RADIUS = 15
HEX_COLUMNS = 3 * HEX_RADIUS * (HEX_RADIUS + 1) + 1  # 721


def axial_disk(radius: int = HEX_RADIUS) -> npt.NDArray[np.int32]:
    """Axial (q, r) coordinates with cube distance ≤ radius, row-major in r then q."""
    coords: list[tuple[int, int]] = []
    for r in range(-radius, radius + 1):
        q_lo = max(-radius, -r - radius)
        q_hi = min(radius, -r + radius)
        for q in range(q_lo, q_hi + 1):
            coords.append((q, r))
    out = np.asarray(coords, dtype=np.int32)
    if len(out) != 3 * radius * (radius + 1) + 1:
        raise RuntimeError("hex disk size mismatch")
    return out


def axial_to_pointy_xy(q: npt.NDArray[np.floating], r: npt.NDArray[np.floating], size: float = 1.0):
    """Pointy-top axial → cartesian. +q → +x; +r → 60° from +x."""
    x = size * (math.sqrt(3.0) * q + math.sqrt(3.0) / 2.0 * r)
    y = size * (1.5 * r)
    return x, y


def _bilinear(
    image: npt.NDArray[np.float32],
    xs: npt.NDArray[np.float32],
    ys: npt.NDArray[np.float32],
):
    """Sample a single-channel image at fractional (x, y) pixel coordinates."""
    h, w = image.shape
    x = np.clip(xs, 0.0, w - 1.001)
    y = np.clip(ys, 0.0, h - 1.001)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = x - x0
    wy = y - y0
    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bot = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return (top * (1.0 - wy) + bot * wy).astype(np.float32)


class HexResampler:
    """Sample the green channel of a 96×96 RGB frame onto 721 hex columns."""

    def __init__(
        self,
        frame_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
        radius: int = HEX_RADIUS,
        margin: float = 1.0,
    ) -> None:
        self.frame_shape = frame_shape
        self.radius = radius
        self.coords = axial_disk(radius)
        q = self.coords[:, 0].astype(np.float64)
        r = self.coords[:, 1].astype(np.float64)
        x, y = axial_to_pointy_xy(q, r)
        self.x_hex = x.astype(np.float32)
        self.y_hex = y.astype(np.float32)
        h, w = frame_shape[0], frame_shape[1]
        # Map hex bounding box into pixel coordinates with a 1px margin.
        x_min, x_max = float(x.min()), float(x.max())
        y_min, y_max = float(y.min()), float(y.max())
        self._px = margin + (x - x_min) / (x_max - x_min) * (w - 1 - 2 * margin)
        self._py = margin + (y - y_min) / (y_max - y_min) * (h - 1 - 2 * margin)

    @property
    def n_columns(self) -> int:
        return int(self.coords.shape[0])

    def resample(self, frame: FrameArray) -> npt.NDArray[np.float32]:
        rgb = require_frame(frame, self.frame_shape)
        green = rgb[:, :, 1] / 255.0
        return _bilinear(green, self._px.astype(np.float32), self._py.astype(np.float32))
