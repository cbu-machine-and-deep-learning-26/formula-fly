"""Tests for the optional FlyvisEye demo script."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_skips_cleanly_without_optional_stack(tmp_path: os.PathLike[str]) -> None:
    """Exit successfully when flyvis or its downloaded checkpoint is absent."""
    environment = os.environ.copy()
    environment["FLYVIS_ROOT_DIR"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "scripts/flyvis_eye_demo.py", "--out", str(tmp_path)],
        check=False,
        capture_output=True,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("SKIP:")
    assert not list(Path(tmp_path).glob("*.png")), "skipping must not write plots"
