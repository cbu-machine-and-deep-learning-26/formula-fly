"""The eye works when torch's default device is a GPU (``import flyvis`` makes it CUDA).

Runs on CUDA or Apple MPS, whichever is present; skipped on a CPU-only machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext

import numpy as np
import pytest

from fly_driver.eyes.hex_resampler import DEFAULT_FRAME_SHAPE, HEX_COLUMN_COUNT, HexResampler

torch = pytest.importorskip("torch")

if torch.cuda.is_available():
    GPU = "cuda"
elif torch.backends.mps.is_available():
    GPU = "mps"
else:
    GPU = None


@pytest.fixture
def gpu_default_device() -> Iterator[str]:
    """Make the GPU torch's default device for one test, as ``import flyvis`` does."""
    if GPU is None:
        pytest.skip("needs a CUDA or MPS device")
    previous = torch.get_default_device()
    torch.set_default_device(GPU)
    try:
        yield GPU
    finally:
        torch.set_default_device(previous)


def test_resampler_and_eye_run_with_a_gpu_default_device(gpu_default_device: str) -> None:
    frame = np.random.default_rng(0).integers(0, 256, DEFAULT_FRAME_SHAPE, dtype=np.uint8)
    resampler = HexResampler(expected_frame_shape=DEFAULT_FRAME_SHAPE)
    cpu_retina = resampler.frame(frame)
    assert cpu_retina.shape == (1, 1, 1, HEX_COLUMN_COUNT)
    gpu_input = torch.rand(1, 2, *DEFAULT_FRAME_SHAPE[:2], device=gpu_default_device)
    assert resampler(gpu_input).device.type == gpu_default_device

    # The eye half. flyvis cannot build its network on MPS (it uses float64), so on a Mac
    # the eye itself runs on the CPU while the GPU default is in force at construction --
    # still the bug: resampler constants on the GPU, frames on the CPU.
    pytest.importorskip("flyvis")
    from fly_driver.eyes.flyvis_eye import FlyvisEye, resolve_checkpoint_dir

    try:
        resolve_checkpoint_dir()
    except FileNotFoundError:
        pytest.skip("run `flyvis download-pretrained` to enable the eye half")
    torch.set_default_device(gpu_default_device)  # `import flyvis` resets it
    eye_device = "cuda" if gpu_default_device == "cuda" else "cpu"
    with pytest.warns(UserWarning) if eye_device == "cpu" else nullcontext():
        eye = FlyvisEye(device=eye_device, warmup_seconds=0.1)
    features = eye.encode(frame)
    assert features.shape == (eye.feature_dim,)
    assert np.all(np.isfinite(features))


@pytest.mark.parametrize("device", [None, "cpu"])
def test_shuffled_eye_builds_whatever_the_default_device(
    gpu_default_device: str, device: str | None
) -> None:
    """flyvis's init generator and torch's default device must agree with the eye's device.

    ``None`` breaks on a Mac (default MPS, flyvis on the CPU) and ``"cpu"`` on a CUDA host
    (default CUDA, flyvis's generator on CUDA).
    """
    pytest.importorskip("flyvis")
    from fly_driver.eyes.flyvis_eye import resolve_checkpoint_dir
    from fly_driver.eyes.shuffled_connectome_eye import ShuffledConnectomeEye

    try:
        resolve_checkpoint_dir()
    except FileNotFoundError:
        pytest.skip("run `flyvis download-pretrained` to enable this test")
    torch.set_default_device(gpu_default_device)  # `import flyvis` resets it
    eye = ShuffledConnectomeEye(device=device, swap_attempts=2_000, warmup_seconds=0.1)
    assert next(eye.network.parameters()).device.type == eye.device.type
