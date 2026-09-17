"""Optional heavy deps must not be imported by the core package."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from fly_driver.envs.carracing import make_carracing
from fly_driver.eyes.flyvis_eye import FlyvisEye
from fly_driver.interface import DEFAULT_FRAME_SHAPE


def test_core_import_does_not_load_flyvis() -> None:
    import fly_driver

    assert "flyvis" not in sys.modules
    assert fly_driver.DirectDriveAgent is not None


def test_flyvis_eye_encode_requires_optional_extra() -> None:
    eye = FlyvisEye(output_size=8)
    frame = np.zeros(DEFAULT_FRAME_SHAPE, dtype=np.uint8)
    with pytest.raises((ImportError, NotImplementedError)):
        eye.encode(frame)


def test_carracing_requires_optional_extra() -> None:
    try:
        import gymnasium  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="optional"):
            make_carracing()
        return
    pytest.skip("gymnasium installed; wrapper construction is Phase 2 env work.")


def test_wandb_not_imported_by_logger() -> None:
    importlib.import_module("fly_driver.training.logging")
    assert "wandb" not in sys.modules
