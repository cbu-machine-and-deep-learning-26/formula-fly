"""Contract tests for ``FlyvisEye`` that run without the optional flyvis stack."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import fly_driver.eyes
from fly_driver.eyes.flyvis_eye import (
    DEFAULT_MOTION_READOUTS,
    FlyvisEye,
    FlyvisNotInstalledError,
    resolve_checkpoint_dir,
)


def test_package_exports_eye_without_importing_flyvis() -> None:
    """Keep flyvis a lazy dependency of the eyes package."""
    assert fly_driver.eyes.FlyvisEye is FlyvisEye
    assert len(DEFAULT_MOTION_READOUTS) == 8


def test_missing_flyvis_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain how to install flyvis instead of failing with a bare ImportError."""
    monkeypatch.setitem(sys.modules, "flyvis", None)

    with pytest.raises(FlyvisNotInstalledError, match="requirements-flyvis.txt"):
        FlyvisEye()
    with pytest.raises(FlyvisNotInstalledError):
        resolve_checkpoint_dir("flow/0000/000")


def test_existing_checkpoint_path_resolves_without_flyvis(tmp_path: Path) -> None:
    """Accept an explicit checkpoint directory as given."""
    assert resolve_checkpoint_dir(tmp_path) == tmp_path


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"frame_rate_hz": 30.0}, "integration limit"),
        ({"warmup_seconds": 0.0}, "at least one grey frame"),
        ({"warmup_luminance": 1.5}, "warmup_luminance"),
        ({"readouts": ()}, "at least one"),
        ({"readouts": ("T4a", "T4a")}, "repeat"),
    ],
)
def test_configuration_is_validated_before_loading(kwargs: dict[str, object], match: str) -> None:
    """Reject bad temporal and readout settings before touching flyvis."""
    with pytest.raises(ValueError, match=match):
        FlyvisEye(**kwargs)  # type: ignore[arg-type]


def test_connectome_cache_writes_and_overwrites_arrays(tmp_path: Path) -> None:
    """Overwriting a cache array used to crash on Windows (datamate's open-handle unlink).

    The first write creates the directory, so only the second reaches the crashing path.
    """
    pytest.importorskip("flyvis")
    import datamate

    from fly_driver.eyes.flyvis_eye import _import_flyvis

    _import_flyvis()
    cache = datamate.Directory(tmp_path / "cache")
    cache["unique_cell_types"] = np.array([b"T4a", b"T5a"])
    cache["unique_cell_types"] = np.arange(3, dtype=np.float32)

    np.testing.assert_array_equal(cache["unique_cell_types"][:], np.arange(3, dtype=np.float32))
