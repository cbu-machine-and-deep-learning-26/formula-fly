"""Tests for the optional video backends."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from fly_driver.training.video import VIDEO_BACKENDS, available_backend, open_video


def _block_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("imageio", "imageio_ffmpeg", "cv2"):
        monkeypatch.setitem(sys.modules, name, None)


def test_no_backend_returns_none_with_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing codecs never raise; the caller skips recording."""
    _block_backends(monkeypatch)
    assert available_backend() is None
    with pytest.warns(UserWarning, match="Skipping video"):
        assert open_video(tmp_path / "x.mp4", fps=50.0) is None
    assert not (tmp_path / "x.mp4").exists()


def test_unknown_backend_name_is_rejected(tmp_path: Path) -> None:
    """Typos in the config surface immediately."""
    with pytest.raises(ValueError, match="unknown video backend"):
        open_video(tmp_path / "x.mp4", fps=50.0, backend="quicktime")
    with pytest.raises(ValueError, match="fps"):
        open_video(tmp_path / "x.mp4", fps=0.0)
    assert set(VIDEO_BACKENDS) == {"imageio", "cv2"}


def test_imageio_backend_writes_a_playable_file(tmp_path: Path) -> None:
    """With imageio-ffmpeg installed, frames become a non-empty mp4."""
    imageio = pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    path = tmp_path / "nested" / "clip.mp4"

    writer = open_video(path, fps=50.0, backend="imageio")
    assert writer is not None and writer.backend == "imageio"
    for value in range(10):
        writer.append(np.full((32, 48, 3), value * 20, dtype=np.uint8))
    writer.close()

    assert path.stat().st_size > 0
    frames = imageio.mimread(str(path))
    assert len(frames) == 10 and frames[0].shape == (32, 48, 3)


def test_frames_are_validated() -> None:
    """Only uint8 RGB frames are accepted."""
    pytest.importorskip("imageio_ffmpeg")
    writer = open_video(Path("unused.mp4"), fps=50.0)
    assert writer is not None
    try:
        with pytest.raises(TypeError):
            writer.append(np.zeros((4, 4, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            writer.append(np.zeros((4, 4), dtype=np.uint8))
    finally:
        writer.close()
        Path("unused.mp4").unlink(missing_ok=True)
