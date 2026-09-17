"""Tiny numpy CNN stand-in. Production CNN may switch to PyTorch (.[train])."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fly_driver.interface import DEFAULT_FRAME_SHAPE, Eye, FrameArray, require_frame


def _relu(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    return np.maximum(x, 0.0, dtype=np.float32)


def _conv2d(
    volume: npt.NDArray[np.float32],
    weight: npt.NDArray[np.float32],
    stride: int,
) -> npt.NDArray[np.float32]:
    """volume (H, W, Cin), weight (Cout, Kh, Kw, Cin) → (Hout, Wout, Cout)."""
    height, width, _ = volume.shape
    cout, kh, kw, _ = weight.shape
    hout = (height - kh) // stride + 1
    wout = (width - kw) // stride + 1
    if hout <= 0 or wout <= 0:
        raise ValueError(f"Convolution emptied the map: in={volume.shape} k={kh} stride={stride}")
    out = np.empty((hout, wout, cout), dtype=np.float32)
    for row in range(hout):
        rs = row * stride
        for col in range(wout):
            cs = col * stride
            patch = volume[rs : rs + kh, cs : cs + kw, :]
            out[row, col] = np.tensordot(weight, patch, axes=([1, 2, 3], [0, 1, 2]))
    return out


class SmallCnnEye(Eye):
    """Two stride-8 convolutions plus a linear readout. Weights are seeded, not trained."""

    def __init__(
        self,
        output_size: int = 256,
        input_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
        seed: int = 0,
    ) -> None:
        if output_size < 1:
            raise ValueError("output_size must be positive")
        self._input_shape = input_shape
        self._output_size = int(output_size)
        rng = np.random.default_rng(seed)
        cin = input_shape[2]
        self._conv1 = rng.normal(0.0, 0.1, size=(8, 5, 5, cin)).astype(np.float32)
        self._conv2 = rng.normal(0.0, 0.1, size=(8, 5, 5, 8)).astype(np.float32)
        dummy = np.zeros(input_shape, dtype=np.float32)
        flat = self._trunk(dummy).size
        self._readout = rng.normal(0.0, 1.0 / np.sqrt(flat), size=(flat, self._output_size)).astype(
            np.float32
        )

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self._input_shape

    @property
    def output_size(self) -> int:
        return self._output_size

    def _trunk(self, frame01: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        h1 = _relu(_conv2d(frame01, self._conv1, stride=8))
        h2 = _relu(_conv2d(h1, self._conv2, stride=2))
        return h2.reshape(-1)

    def encode(self, frame: FrameArray) -> npt.NDArray[np.float32]:
        pixels = require_frame(frame, self._input_shape) / 255.0
        return (self._trunk(pixels) @ self._readout).astype(np.float32, copy=False)
