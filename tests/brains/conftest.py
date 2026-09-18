"""Run the torch brain tests on every device this machine actually has (GH-23).

The lab's 4090 boxes are where the real numbers come from, so the GPU path has to be
exercised rather than assumed. Every test that takes the ``device`` fixture runs once per
available device: ``cpu`` everywhere, and ``cuda`` as well wherever CUDA is present. On a
CPU-only machine the suite is exactly what it was.

That matters more than usual here, because none of the CUDA code will have run before it
reaches the lab. The first CUDA execution is this suite, and it is meant to catch a device
or dtype mistake on the first try rather than to let a wrong number look plausible.
"""

from __future__ import annotations

import pytest


def _available_devices() -> list[str]:
    """``["cpu"]``, plus ``"cuda"`` when torch can see a card."""
    try:
        import torch
    except ImportError:  # pragma: no cover - the whole module skips without torch
        return ["cpu"]
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


DEVICES = _available_devices()


@pytest.fixture(params=DEVICES, ids=DEVICES)
def device(request: pytest.FixtureRequest) -> str:
    """A torch device present on this machine."""
    return request.param


@pytest.fixture(scope="session")
def cuda_device() -> str:
    """``"cuda"``, or skip. For tests that only make sense against a real card."""
    if "cuda" not in DEVICES:
        pytest.skip("no CUDA device available")
    return "cuda"
