"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from fly_driver.interface import DEFAULT_FRAME_SHAPE


@pytest.fixture
def frame() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=DEFAULT_FRAME_SHAPE, dtype=np.uint8)
