"""CNN and flyvis stay off the import path of the base install (GH-15)."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_ROOT = Path(__file__).parents[1]


def test_shuffle_and_random_projection_import_without_torch_or_flyvis() -> None:
    """A fresh interpreter with the base install must not touch either stack."""
    script = textwrap.dedent(
        """
        import sys
        import fly_driver.eyes
        from fly_driver.eyes.degree_shuffle import degree_matched_shuffle
        from fly_driver.eyes.random_projection_eye import RandomProjectionEye

        assert "torch" not in sys.modules
        assert "flyvis" not in sys.modules
        assert fly_driver.eyes.RandomProjectionEye is RandomProjectionEye
        assert fly_driver.eyes.degree_matched_shuffle is degree_matched_shuffle
        assert RandomProjectionEye(frame_shape=(2, 2, 3), feature_dim=3).feature_dim == 3
        """
    )
    completed = _run(script)
    assert completed.returncode == 0, completed.stderr


def test_cnn_without_torch_names_the_optional_stack() -> None:
    """Skip when this interpreter already has torch; CI does not."""
    probe = subprocess.run(
        [sys.executable, "-c", "import torch"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        return
    script = textwrap.dedent(
        """
        try:
            import fly_driver.eyes.cnn_eye  # noqa: F401
        except ImportError as error:
            message = str(error)
            assert "optional torch" in message, message
        else:
            raise SystemExit("cnn_eye imported without torch")
        """
    )
    completed = _run(script)
    assert completed.returncode == 0, completed.stderr


def _run(script: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(_ROOT), env.get("PYTHONPATH", "")])
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=_ROOT,
    )
