"""Tests for the evaluation entry point."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]


def test_script_runs_the_default_config_and_writes_outputs(tmp_path: Path) -> None:
    """The CLI evaluates a baseline agent on the dummy track and writes results."""
    pytest.importorskip("yaml")
    result = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "evaluate.py"),
            "--agent",
            "constant",
            "--episodes",
            "1",
            "--seed",
            "5",
            "--output",
            str(tmp_path / "out"),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO,
    )

    assert result.returncode == 0, result.stderr
    assert "actions_digest:" in result.stdout
    assert "clean" in result.stdout
    assert (tmp_path / "out" / "episodes.csv").exists()
    assert (tmp_path / "out" / "summary.json").exists()
