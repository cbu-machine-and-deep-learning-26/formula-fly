"""Contract tests for ``FlyvisEye`` that run without the optional flyvis stack."""

from __future__ import annotations

import sys
from pathlib import Path

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
def test_configuration_is_validated_before_loading(
    kwargs: dict[str, object], match: str
) -> None:
    """Reject bad temporal and readout settings before touching flyvis."""
    with pytest.raises(ValueError, match=match):
        FlyvisEye(**kwargs)  # type: ignore[arg-type]
