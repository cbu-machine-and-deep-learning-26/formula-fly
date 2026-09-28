"""The ``.knh`` reader and writer (GH-25). Synthetic trees only; Kunos's files stay local."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fly_driver.cockpit.knh import KnhNode, read_knh, world_matrices, write_knh


def _translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[3, :3] = (x, y, z)  # Direct3D: translation in the last row
    return matrix


def _rotation_y_90() -> np.ndarray:
    """Row-vector rotation taking +x to -z (90 degrees about +y, right-handed)."""
    matrix = np.eye(4)
    matrix[:3, :3] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    return matrix


def _tree() -> KnhNode:
    hand = KnhNode("DRIVER:RIG_HAND_L", _translation(0.0, -0.1, 0.0))
    arm = KnhNode("DRIVER:RIG_Arm_L", _translation(0.2, 0.0, 0.0), [hand])
    chest = KnhNode("DRIVER:RIG_Cest", _rotation_y_90() @ _translation(0.0, 0.5, 0.0), [arm])
    return KnhNode("DRIVER:DRIVER", np.eye(4), [chest])


def test_round_trip(tmp_path: Path) -> None:
    path = write_knh(_tree(), tmp_path / "driver_base_pos.knh")
    read = read_knh(path)
    assert [node.name for node in read.walk()] == [node.name for node in _tree().walk()]
    for original, loaded in zip(_tree().walk(), read.walk(), strict=True):
        np.testing.assert_allclose(loaded.local, original.local, atol=1e-7)


def test_world_matrices_compose_as_row_vectors() -> None:
    """``world = local @ parent``: the arm's +x offset is turned by the chest's rotation."""
    worlds = world_matrices(_tree())
    np.testing.assert_allclose(worlds["DRIVER:RIG_Cest"][3, :3], (0.0, 0.5, 0.0), atol=1e-12)
    # The arm sits 0.2 along the chest's local +x, which the chest turned to world -z.
    np.testing.assert_allclose(worlds["DRIVER:RIG_Arm_L"][3, :3], (0.0, 0.5, -0.2), atol=1e-12)
    np.testing.assert_allclose(worlds["DRIVER:RIG_HAND_L"][3, :3], (0.0, 0.4, -0.2), atol=1e-12)


def test_repeated_names_keep_the_first_occurrence() -> None:
    """Kunos's rig has ``DRIVER:HELMET_plastic`` as its own child."""
    inner = KnhNode("DRIVER:HELMET_plastic", _translation(0.0, 1.0, 0.0))
    outer = KnhNode("DRIVER:HELMET_plastic", _translation(0.0, 0.1, 0.0), [inner])
    worlds = world_matrices(KnhNode("DRIVER:DRIVER", np.eye(4), [outer]))
    np.testing.assert_allclose(worlds["DRIVER:HELMET_plastic"][3, :3], (0.0, 0.1, 0.0))


def test_truncated_and_padded_files_are_errors(tmp_path: Path) -> None:
    path = write_knh(_tree(), tmp_path / "good.knh")
    data = path.read_bytes()
    (tmp_path / "short.knh").write_bytes(data[:-10])
    (tmp_path / "long.knh").write_bytes(data + b"\x00\x00")
    with pytest.raises(ValueError):
        read_knh(tmp_path / "short.knh")
    with pytest.raises(ValueError, match="trailing bytes"):
        read_knh(tmp_path / "long.knh")


def test_a_non_4x4_matrix_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="4x4"):
        write_knh(KnhNode("bad", np.eye(3)), tmp_path / "bad.knh")
