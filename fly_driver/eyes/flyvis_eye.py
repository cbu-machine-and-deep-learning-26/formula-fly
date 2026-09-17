"""flyvis optic-lobe eye. Optional extra: pip install 'fly-driver[eye]'."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from fly_driver.interface import DEFAULT_FRAME_SHAPE, Eye, FrameArray, require_frame


class FlyvisEye(Eye):
    """Frozen-pretrained flyvis wrapper. Importing this module does not import flyvis."""

    def __init__(
        self,
        output_size: int = 256,
        input_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
        readout_types: tuple[str, ...] = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"),
        frozen: bool = True,
        checkpoint: str | None = None,
    ) -> None:
        self._input_shape = input_shape
        self._output_size = int(output_size)
        self.readout_types = readout_types
        self.frozen = frozen
        self.checkpoint = checkpoint
        self._network: Any = None

    def _load(self) -> None:
        try:
            import flyvis  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "flyvis is optional. Install with: pip install 'fly-driver[eye]'. "
                "The frozen flyvis encode path is tracked separately."
            ) from exc
        raise NotImplementedError(
            "Flyvis checkpoint load is Phase 1. Geometry and interface tests use control eyes."
        )

    @property
    def input_shape(self) -> tuple[int, int, int]:
        return self._input_shape

    @property
    def output_size(self) -> int:
        return self._output_size

    def encode(self, frame: FrameArray) -> npt.NDArray[np.float32]:
        require_frame(frame, self._input_shape)
        if self._network is None:
            self._load()
        raise NotImplementedError("Flyvis encode is Phase 1.")
