"""Tests for the optional flyvis smoke script."""

from __future__ import annotations

import os
import subprocess
import sys


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
