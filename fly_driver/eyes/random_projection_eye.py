"""Fixed random projection from grey pixels to the flyvis readout width.

The RQ1 random-projection control (`AGENTS.md` §6). It has the same
:class:`~fly_driver.interface.Eye` contract and the same feature width as a
frozen :class:`~fly_driver.eyes.flyvis_eye.FlyvisEye` T4/T5 readout, and no
trainable parameters: the matrix is drawn once from ``seed`` and then held
fixed. Two runs with the same seed see the same projection, so a policy-seed
sweep does not silently change the eye.

numpy only. Torch and flyvis are not imported.
"""

from __future__ import annotations

import numpy as np

from fly_driver.eyes.constants import FLYVIS_DEFAULT_FEATURE_DIM
from fly_driver.interface import FEATURE_DTYPE, FRAME_SHAPE, Features, Frame, validate_frame

__all__ = ["RandomProjectionEye"]

#: BT.601 luminance, the same weights :func:`fly_driver.eyes.hex_resampler.frame_to_gray`
#: uses. Repeated here so this eye does not import the resampler (that module
#: imports torch). ``tests/eyes`` checks the two tuples stay equal.
_LUMA_WEIGHTS = np.array((0.299, 0.587, 0.114), dtype=np.float32)


class RandomProjectionEye:
    """Grey pixels, one frozen Gaussian matrix, ``feature_dim`` features.

    The frame is converted to BT.601 luminance in ``[0, 1]``, flattened in
    row-major order, and multiplied by a matrix of shape
    ``(height * width, feature_dim)`` with entries ``N(0, 1 / n_pixels)``.
    That scale keeps a feature O(1) when pixels are O(1).

    Args:
        frame_shape: ``(height, width, 3)`` of the frames :meth:`encode` accepts.
        seed: Seeds the matrix. The default is 0 so every condition shares one
            projection unless a config sets ``eye.params.seed``.
        feature_dim: Length of :meth:`encode`. The default matches the frozen
            flyvis T4/T5 readout.

    Raises:
        TypeError: If ``seed`` is not an integer.
        ValueError: If the frame shape, ``seed``, or ``feature_dim`` is invalid.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int, int] = FRAME_SHAPE,
        seed: int = 0,
        feature_dim: int = FLYVIS_DEFAULT_FEATURE_DIM,
    ) -> None:
        shape = tuple(int(value) for value in frame_shape)
        if len(shape) != 3 or shape[2] != 3 or shape[0] < 1 or shape[1] < 1:
            raise ValueError(f"frame_shape must be (height, width, 3), got {frame_shape}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be a non-negative integer, got {seed!r}")
        if seed < 0:
            raise ValueError(f"seed must be a non-negative integer, got {seed}")
        if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
            raise ValueError(f"feature_dim must be a positive integer, got {feature_dim!r}")
        self._frame_shape: tuple[int, int, int] = (shape[0], shape[1], 3)
        self._feature_dim = int(feature_dim)
        self.seed = int(seed)
        n_pixels = shape[0] * shape[1]
        generator = np.random.default_rng(self.seed)
        self._projection = generator.standard_normal(
            (n_pixels, self._feature_dim), dtype=np.float32
        )
        self._projection /= np.float32(np.sqrt(n_pixels))

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Declared camera frame shape."""
        return self._frame_shape

    @property
    def feature_dim(self) -> int:
        """Length of the vector :meth:`encode` returns. Default 5768."""
        return self._feature_dim

    def reset(self) -> None:
        """Stateless: the projection does not change between episodes."""

    def encode(self, frame: Frame) -> Features:
        """Project one frame. ``(feature_dim,)`` float32, finite.

        Args:
            frame: uint8 RGB frame with the declared ``frame_shape``.

        Returns:
            The flattened grey frame times the frozen matrix.
        """
        validate_frame(frame, self._frame_shape)
        grey = (frame.astype(np.float32) * (1.0 / 255.0)) @ _LUMA_WEIGHTS
        return (grey.reshape(-1) @ self._projection).astype(FEATURE_DTYPE, copy=False)
