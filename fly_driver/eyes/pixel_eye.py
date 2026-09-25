"""Grey pixels, block-averaged and flattened: the smoke-test eye (GH-17).

This is **not** one of the RQ1 control eyes. Those -- a small CNN, a random projection and
the degree-matched shuffled connectome -- are GH-15's and are sized to match the flyvis
readout. This one exists so the training loop can be run, tested and profiled on any
machine, including ones with no flyvis checkpoint and no GPU, and so a "the loop learns
something" check has an eye that costs nothing. It has no parameters, so it cannot be
fine-tuned, and the loop says so if asked.

Same luminance as the hex resampler (BT.601), so what this eye sees is the grey image the
fly's retina is sampled from, minus the hexagonal geometry.
"""

from __future__ import annotations

import numpy as np

from fly_driver.eyes.hex_resampler import RGB_LUMA_WEIGHTS
from fly_driver.interface import FEATURE_DTYPE, FRAME_SHAPE, Features, Frame, validate_frame

__all__ = ["PixelEye"]


class PixelEye:
    """Block-averaged grey pixels in ``[0, 1]``, row-major.

    Args:
        frame_shape: ``(height, width, 3)`` of the frames :meth:`encode` accepts. Take it
            from the env.
        downsample: Side of the square block each feature averages over. Must divide both
            the height and the width, so no pixel is silently dropped.

    Raises:
        ValueError: On a bad shape or a block that does not tile the frame.
    """

    def __init__(
        self, frame_shape: tuple[int, int, int] = FRAME_SHAPE, downsample: int = 4
    ) -> None:
        shape = tuple(int(value) for value in frame_shape)
        if len(shape) != 3 or shape[2] != 3 or shape[0] < 1 or shape[1] < 1:
            raise ValueError(f"frame_shape must be (height, width, 3), got {frame_shape}")
        if isinstance(downsample, bool) or not isinstance(downsample, int) or downsample < 1:
            raise ValueError(f"downsample must be a positive integer, got {downsample!r}")
        if shape[0] % downsample or shape[1] % downsample:
            raise ValueError(
                f"downsample={downsample} does not tile a {shape[0]}x{shape[1]} frame; "
                "pick a divisor of both so no pixels are dropped"
            )
        self._frame_shape: tuple[int, int, int] = (shape[0], shape[1], 3)
        self.downsample = int(downsample)
        self.rows = shape[0] // self.downsample
        self.columns = shape[1] // self.downsample
        self._weights = np.asarray(RGB_LUMA_WEIGHTS, dtype=np.float32)

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Declared camera frame shape."""
        return self._frame_shape

    @property
    def feature_dim(self) -> int:
        """``(height // downsample) * (width // downsample)``."""
        return self.rows * self.columns

    def reset(self) -> None:
        """Stateless: nothing to reset."""

    def encode(self, frame: Frame) -> Features:
        """Grey, block-averaged, flattened; ``(feature_dim,)`` float32 in ``[0, 1]``."""
        validate_frame(frame, self._frame_shape)
        grey = (frame.astype(np.float32) / 255.0) @ self._weights
        blocks = grey.reshape(self.rows, self.downsample, self.columns, self.downsample)
        return blocks.mean(axis=(1, 3)).reshape(-1).astype(FEATURE_DTYPE, copy=False)
