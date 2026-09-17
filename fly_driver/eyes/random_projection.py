"""Fixed random Gaussian projection of flattened pixels."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fly_driver.interface import DEFAULT_FRAME_SHAPE, Eye, FrameArray, require_frame


class RandomProjectionEye(Eye):
    """Untrainable random matrix. Matched output size vs connectome readout."""

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
        n_in = int(np.prod(input_shape))
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(n_in)
        self._weight = rng.normal(0.0, scale, size=(n_in, self._output_size)).astype(np.float32)

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self._input_shape

    @property
    def output_size(self) -> int:
        return self._output_size

    def encode(self, frame: FrameArray) -> npt.NDArray[np.float32]:
        pixels = require_frame(frame, self._input_shape) / 255.0
        return pixels.reshape(-1) @ self._weight
