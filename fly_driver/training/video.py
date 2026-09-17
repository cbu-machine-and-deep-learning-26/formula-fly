"""Eval-episode video helper. MP4 requires the optional video extra."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt


def write_episode_video(
    path: str | Path,
    frames: list[npt.NDArray[np.uint8]] | npt.NDArray[np.uint8],
    *,
    fps: int = 30,
) -> Path:
    """Write RGB frames.

    Prefers imageio (optional extra). Falls back to a ``.npz`` archive so tests
    and Spark boxes without ffmpeg still produce an artifact.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    stack = np.asarray(frames)
    if stack.ndim != 4:
        raise ValueError(f"Expected (T, H, W, C) frames, got {stack.shape}")
    try:
        import imageio.v3 as iio

        mp4 = dest.with_suffix(".mp4")
        iio.imwrite(mp4, stack, fps=fps)
        return mp4
    except ImportError:
        npz = dest.with_suffix(".npz")
        np.savez_compressed(npz, frames=stack, fps=np.int32(fps))
        return npz
