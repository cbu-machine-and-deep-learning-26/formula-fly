"""Installing the fly over a stock driver model keeps the original, and restore puts it back."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "install_fly_driver",
    Path(__file__).resolve().parents[2] / "scripts" / "cockpit" / "install_fly_driver.py",
)
assert _SPEC and _SPEC.loader
install_fly_driver = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(install_fly_driver)


@pytest.fixture
def ac_root(tmp_path: Path) -> Path:
    driver = tmp_path / "content" / "driver"
    driver.mkdir(parents=True)
    (driver / "2016_Driver.kn5").write_bytes(b"stock driver")
    (tmp_path / "fly.kn5").write_bytes(b"the fly")
    return tmp_path


def _run(ac_root: Path, *args: str) -> int:
    return install_fly_driver.main(["--ac-root", str(ac_root), "--driver", "2016_Driver", *args])


def test_install_backs_up_once_and_restore_puts_it_back(ac_root: Path) -> None:
    target = ac_root / "content" / "driver" / "2016_Driver.kn5"
    backup = target.with_name("2016_Driver.kn5.stock-backup")

    assert _run(ac_root, "--kn5", str(ac_root / "fly.kn5")) == 0
    assert target.read_bytes() == b"the fly" and backup.read_bytes() == b"stock driver"
    # A second install must not overwrite the stock backup with the fly.
    assert _run(ac_root, "--kn5", str(ac_root / "fly.kn5")) == 0
    assert backup.read_bytes() == b"stock driver"

    assert _run(ac_root, "--restore") == 0
    assert target.read_bytes() == b"stock driver" and not backup.exists()


def test_restore_without_a_backup_and_a_missing_kn5_are_errors(ac_root: Path) -> None:
    assert _run(ac_root, "--restore") == 1
    assert _run(ac_root, "--kn5", str(ac_root / "nope.kn5")) == 1
    assert (ac_root / "content" / "driver" / "2016_Driver.kn5").read_bytes() == b"stock driver"
