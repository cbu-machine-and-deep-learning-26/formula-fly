"""The training entry point: config in, run directories out (GH-17)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parents[2]
SCRIPT = REPO / "scripts" / "train.py"

SMALL_CONFIG = """
name: smoke
seeds: [0, 1]
total_env_steps: 128
device: cpu
eye: {type: pixels, frozen: true, params: {downsample: 8}}
policy: {hidden_sizes: [8]}
env: {type: dummy, max_steps: 30}
ppo: {rollout_steps: 64, minibatches: 4, epochs: 1}
"""


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO,
    )


def test_a_config_driven_run_writes_a_directory_per_seed(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("yaml")
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(SMALL_CONFIG, encoding="utf-8")

    result = _run("--config", str(config_path), "--output", str(tmp_path / "runs"))

    assert result.returncode == 0, result.stderr
    assert "condition smoke: eye=pixels (frozen) brain=none env=dummy seeds=[0, 1]" in result.stdout
    for seed in (0, 1):
        run_dir = tmp_path / "runs" / "smoke" / f"seed_{seed}"
        assert (run_dir / "episodes.csv").exists()
        assert (run_dir / "updates.csv").exists()
        assert (run_dir / "policy_final.pt").exists()
    assert (tmp_path / "runs" / "smoke" / "config.yaml").exists()


def test_seed_and_eye_overrides_narrow_the_run(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("yaml")
    config_path = tmp_path / "smoke.yaml"
    config_path.write_text(SMALL_CONFIG, encoding="utf-8")

    result = _run(
        "--config",
        str(config_path),
        "--output",
        str(tmp_path / "runs"),
        "--seed",
        "1",
        "--name",
        "one",
    )

    assert result.returncode == 0, result.stderr
    assert "seeds=[1]" in result.stdout
    assert (tmp_path / "runs" / "one" / "seed_1" / "episodes.csv").exists()
    assert not (tmp_path / "runs" / "one" / "seed_0").exists()


def test_the_default_config_parses_before_torch_is_needed() -> None:
    """A typo in the YAML fails fast, in any venv, before anything heavy loads."""
    pytest.importorskip("yaml")
    result = _run("--help")
    assert result.returncode == 0
    assert "--seed" in result.stdout and "--eye" in result.stdout and "--fine-tune" in result.stdout
