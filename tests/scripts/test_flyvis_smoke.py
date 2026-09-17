"""Tests for the optional flyvis smoke script."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType


def _load_smoke_module() -> ModuleType:
    script_path = Path(__file__).parents[2] / "scripts" / "flyvis_smoke.py"
    module_spec = importlib.util.spec_from_file_location("flyvis_smoke", script_path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"could not load {script_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_skips_cleanly_without_optional_stack(tmp_path: os.PathLike[str]) -> None:
    """Exit successfully when flyvis or its downloaded checkpoint is absent."""
    environment = os.environ.copy()
    environment["FLYVIS_ROOT_DIR"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "scripts/flyvis_smoke.py"],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0
    assert result.stdout.startswith("SKIP:")


def test_edge_frames_sweep_monotonically() -> None:
    """Create a correctly shaped edge that only advances across receptors."""
    smoke_module = _load_smoke_module()
    horizontal_positions = [-2.0, -1.0, 0.0, 1.0, 2.0]

    frames = smoke_module._create_edge_frames(horizontal_positions, frame_count=5)

    assert len(frames) == 5
    assert all(len(frame) == len(horizontal_positions) for frame in frames)
    bright_counts = [sum(frame) for frame in frames]
    assert bright_counts == sorted(bright_counts)
    for receptor_index in range(len(horizontal_positions)):
        receptor_values = [frame[receptor_index] for frame in frames]
        assert receptor_values == sorted(receptor_values)
