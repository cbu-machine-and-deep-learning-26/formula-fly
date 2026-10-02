"""Small convolutional eye with the flyvis readout width.

The RQ1 CNN control (`AGENTS.md` §6): a few stride-2 convolutions, a global
pool, and a linear map to the same feature width as frozen flyvis T4/T5. The
spatial filters are the part that can learn; the linear map exists so the
policy head on top is the same size as the one on the optic lobe.

Torch is optional. Importing this module without it raises
:class:`TorchNotInstalledError`. The base install and CI do not depend on it.
"""

from __future__ import annotations

import math

import numpy as np

from fly_driver.eyes.constants import FLYVIS_DEFAULT_FEATURE_DIM
from fly_driver.interface import FEATURE_DTYPE, FRAME_SHAPE, Features, Frame, validate_frame

__all__ = ["CnnEye", "TorchNotInstalledError"]


class TorchNotInstalledError(ImportError):
    """Raised when :class:`CnnEye` is imported and torch is not installed."""


try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - CI has no torch, and skips this module
    raise TorchNotInstalledError(
        "CnnEye needs the optional torch stack. It is not part of the base install "
        "(see pyproject.toml); install torch in the training or flyvis virtualenv."
    ) from error


class CnnEye(nn.Module):
    """Three stride-2 convolutions, a 2×2 pool, then a linear map to ``feature_dim``.

    Channels are 8, 16, 16. After the pool the vector is 64 long, and the
    linear layer expands it to the flyvis width (5768 by default). That
    expansion is most of the parameter count; the convolutions stay small.

    ``seed`` initialises the weights from a private generator, so constructing
    the eye does not consume torch's global RNG and does not follow whatever
    ``torch.manual_seed`` the training loop just set. The default seed is 0:
    every frozen-CNN condition shares one eye unless ``eye.params.seed`` says
    otherwise. :meth:`encode` and :meth:`encode_batch` agree on a frame, which
    is what fine-tuning through PPO needs.

    Args:
        frame_shape: ``(height, width, 3)`` of the frames :meth:`encode` accepts.
        seed: Initialises the convolutions and the linear map.
        feature_dim: Length of the feature vector. Default matches frozen flyvis.

    Raises:
        TorchNotInstalledError: If torch is not installed (raised at import).
        TypeError: If ``seed`` is not an integer.
        ValueError: If the frame shape, ``seed``, or ``feature_dim`` is invalid.
    """

    def __init__(
        self,
        frame_shape: tuple[int, int, int] = FRAME_SHAPE,
        seed: int = 0,
        feature_dim: int = FLYVIS_DEFAULT_FEATURE_DIM,
    ) -> None:
        super().__init__()
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
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.readout = nn.Linear(16 * 2 * 2, self._feature_dim)
        self.register_buffer("_luma", torch.tensor((0.299, 0.587, 0.114), dtype=torch.float32))
        self._init_weights(self.seed)

    @property
    def frame_shape(self) -> tuple[int, int, int]:
        """Declared camera frame shape."""
        return self._frame_shape

    @property
    def feature_dim(self) -> int:
        """Length of the vector :meth:`encode` returns. Default 5768."""
        return self._feature_dim

    def reset(self) -> None:
        """Stateless: nothing carries from one episode to the next."""

    def encode(self, frame: Frame) -> Features:
        """Encode one frame under ``no_grad``. ``(feature_dim,)`` float32.

        Args:
            frame: uint8 RGB frame with the declared ``frame_shape``.

        Returns:
            The same vector :meth:`encode_batch` produces for that frame alone.
        """
        validate_frame(frame, self._frame_shape)
        batch = torch.from_numpy(np.ascontiguousarray(frame)).unsqueeze(0)
        batch = batch.to(self._luma.device)
        with torch.no_grad():
            features = self.encode_batch(batch)
        return features[0].detach().cpu().numpy().astype(FEATURE_DTYPE, copy=False)

    def encode_batch(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode a batch of uint8 frames. Differentiable.

        Args:
            frames: ``(batch, height, width, 3)`` uint8, on any device. Moved
                onto the eye's device.

        Returns:
            ``(batch, feature_dim)`` float32.

        Raises:
            TypeError: If ``frames`` is not a uint8 tensor.
            ValueError: If the spatial shape is not ``frame_shape``.
        """
        if not isinstance(frames, torch.Tensor):
            raise TypeError(f"frames must be a torch tensor, got {type(frames).__name__}")
        if frames.dtype != torch.uint8:
            raise TypeError(
                f"frames must be uint8 in [0, 255], got {frames.dtype}. No implicit cast."
            )
        if frames.ndim != 4 or tuple(frames.shape[1:]) != self._frame_shape:
            raise ValueError(
                f"frames shape {tuple(frames.shape)} != (batch,) + {self._frame_shape}. "
                "No implicit resize."
            )
        grey = frames.to(device=self._luma.device, dtype=torch.float32) * (1.0 / 255.0)
        grey = (grey * self._luma).sum(dim=-1, keepdim=True)
        image = grey.permute(0, 3, 1, 2)
        pooled = self.features(image)
        return self.readout(pooled.flatten(1))

    def _init_weights(self, seed: int) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
                if module.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
                    nn.init.uniform_(module.bias, -bound, bound, generator=generator)
