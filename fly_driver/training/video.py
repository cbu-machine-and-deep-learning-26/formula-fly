"""Episode video writing with optional backends.

Video is a report nicety, not a dependency: :func:`open_video` returns a writer
when ``imageio`` (with ``imageio-ffmpeg``) or OpenCV is importable, and ``None``
with a warning when neither is. Callers treat ``None`` as "skip recording" so an
evaluation never fails for lack of a codec.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

__all__ = ["VIDEO_BACKENDS", "VideoWriter", "available_backend", "open_video"]

VIDEO_BACKENDS = ("imageio", "cv2")

Frame = npt.NDArray[np.uint8]


class VideoWriter(Protocol):
    """Minimal writer surface used by the evaluation harness."""

    backend: str
    path: Path

    def append(self, frame: Frame) -> None:
        """Add one ``(height, width, 3)`` uint8 RGB frame."""

    def close(self) -> None:
        """Finish the file."""


def _import_imageio() -> Any | None:
    try:
        import imageio.v2 as imageio
        import imageio_ffmpeg  # noqa: F401  (imageio's mp4 plugin)
    except ImportError:
        return None
    return imageio


def _import_cv2() -> Any | None:
    try:
        import cv2
    except ImportError:
        return None
    return cv2


def available_backend(preferred: str = "auto") -> str | None:
    """Return the first importable backend name, or ``None``.

    Args:
        preferred: ``"auto"`` tries :data:`VIDEO_BACKENDS` in order; a backend
            name checks only that one.

    Raises:
        ValueError: If ``preferred`` is not ``"auto"`` or a known backend.
    """
    if preferred == "auto":
        candidates: tuple[str, ...] = VIDEO_BACKENDS
    elif preferred in VIDEO_BACKENDS:
        candidates = (preferred,)
    else:
        raise ValueError(
            f"unknown video backend {preferred!r}; choose from {VIDEO_BACKENDS}"
        )
    for name in candidates:
        loader = _import_imageio if name == "imageio" else _import_cv2
        if loader() is not None:
            return name
    return None


def _validate_frame(frame: object) -> Frame:
    if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
        raise TypeError("video frames must be uint8 numpy arrays")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"video frames must be (height, width, 3), got {frame.shape}")
    return frame


class _ImageioWriter:
    backend = "imageio"

    def __init__(self, imageio: Any, path: Path, fps: float) -> None:
        self.path = path
        # macro_block_size=1 keeps the exact frame size instead of padding to 16 px.
        self._writer = imageio.get_writer(str(path), fps=fps, macro_block_size=1)

    def append(self, frame: Frame) -> None:
        self._writer.append_data(_validate_frame(frame))

    def close(self) -> None:
        self._writer.close()


class _Cv2Writer:
    backend = "cv2"

    def __init__(self, cv2: Any, path: Path, fps: float) -> None:
        self.path = path
        self._cv2 = cv2
        self._fps = fps
        self._writer: Any | None = None

    def append(self, frame: Frame) -> None:
        frame = _validate_frame(frame)
        if self._writer is None:
            height, width, _ = frame.shape
            fourcc = self._cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = self._cv2.VideoWriter(
                str(self.path), fourcc, self._fps, (width, height)
            )
        self._writer.write(self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()


def open_video(
    path: str | os.PathLike[str], fps: float, backend: str = "auto"
) -> VideoWriter | None:
    """Open an mp4 writer, or return ``None`` (with a warning) if no backend exists.

    Args:
        path: Output file; parent directories are created.
        fps: Playback frame rate, normally the environment frame rate.
        backend: ``"auto"``, ``"imageio"``, or ``"cv2"``.

    Returns:
        A writer, or ``None`` when recording must be skipped.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    name = available_backend(backend)
    if name is None:
        warnings.warn(
            "no video backend available; install `imageio imageio-ffmpeg` (or "
            "opencv-python) to record evaluation episodes. Skipping video.",
            stacklevel=2,
        )
        return None
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if name == "imageio":
        return _ImageioWriter(_import_imageio(), target, fps)
    return _Cv2Writer(_import_cv2(), target, fps)
