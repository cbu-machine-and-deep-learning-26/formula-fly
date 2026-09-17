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


def test_edge_frames_have_timed_baseline_sweep_and_hold() -> None:
    """Create both edge directions with the configured timing and monotone sweep."""
    smoke_module = _load_smoke_module()
    horizontal_positions = [-2.0, -1.0, 0.0, 1.0, 2.0]
    expected_frame_count = (
        smoke_module.EDGE_PRE_STIMULUS_FRAMES
        + smoke_module.EDGE_SWEEP_FRAMES
        + smoke_module.EDGE_POST_STIMULUS_FRAMES
    )

    for direction in ("ltr", "rtl"):
        frames = smoke_module._create_edge_frames(horizontal_positions, direction)

        assert len(frames) == expected_frame_count
        assert all(len(frame) == len(horizontal_positions) for frame in frames)
        pre_stimulus = frames[: smoke_module.EDGE_PRE_STIMULUS_FRAMES]
        assert all(frame == [0.5] * len(horizontal_positions) for frame in pre_stimulus)

        sweep_start = smoke_module.EDGE_PRE_STIMULUS_FRAMES
        sweep_stop = sweep_start + smoke_module.EDGE_SWEEP_FRAMES
        sweep_frames = frames[sweep_start:sweep_stop]
        bright_counts = [sum(frame) for frame in sweep_frames]
        assert bright_counts == sorted(bright_counts)
        for receptor_index in range(len(horizontal_positions)):
            receptor_values = [frame[receptor_index] for frame in sweep_frames]
            assert receptor_values == sorted(receptor_values)

        post_stimulus = frames[sweep_stop:]
        assert all(frame == sweep_frames[-1] for frame in post_stimulus)
