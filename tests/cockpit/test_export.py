"""The OBJ and manifest writer that hands the fly to Blender (GH-25)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fly_driver.cockpit.export import MANIFEST_NAME, is_skeleton_node, write_parts
from fly_driver.cockpit.knh import KnhNode
from fly_driver.cockpit.seated_fly import BodyPart, MeshPiece


def _translation(x: float, y: float, z: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[3, :3] = (x, y, z)
    return matrix


def _rig() -> KnhNode:
    helmet = KnhNode("DRIVER:HELMET", np.eye(4), [KnhNode("DRIVER:HELMET_SUB0", np.eye(4))])
    head = KnhNode("DRIVER:RIG_Head", _translation(0, 0.2, 0), [helmet])
    hips = KnhNode("DRIVER:RIG_Hips", _translation(0, 0.3, 0.1), [head])
    driver = KnhNode("DRIVER:DRIVER", np.eye(4), [hips])
    # The editor's wrapper nodes, as in a real driver_base_pos.knh.
    return KnhNode("SCENE_ROOT", np.eye(4), [KnhNode("FBX: steer.fbx", np.eye(4), [driver])])


def _part(name: str, bone: str) -> BodyPart:
    piece = MeshPiece(
        material="body",
        rgba=(0.8, 0.5, 0.2, 1.0),
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        faces=np.array([[0, 1, 2]], np.int32),
    )
    return BodyPart(name=name, bone=bone, bodies=(name,), pieces=(piece, piece))


def test_the_manifest_has_the_skeleton_and_the_parts(tmp_path: Path) -> None:
    parts = [_part("DRIVER:HELMET", "DRIVER:RIG_Head"), _part("FLY:thorax", "DRIVER:RIG_Hips")]
    manifest_path = write_parts(parts, _rig(), tmp_path, scale=4.9)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == MANIFEST_NAME
    bones = [bone["name"] for bone in manifest["bones"]]
    # Only the skeleton: the stock helmet would collide with the fly's DRIVER:HELMET object.
    assert bones == ["DRIVER:DRIVER", "DRIVER:RIG_Hips", "DRIVER:RIG_Head"]
    assert [bone["parent"] for bone in manifest["bones"]] == [
        None,
        "DRIVER:DRIVER",
        "DRIVER:RIG_Hips",
    ]
    head_world = np.array(manifest["bones"][2]["world"])
    np.testing.assert_allclose(head_world[3, :3], (0, 0.5, 0.1))
    assert [part["vertices"] for part in manifest["parts"]] == [6, 6]
    assert manifest["scale_m_per_mujoco_unit"] == 4.9


def test_obj_files_carry_both_pieces_with_offset_indices(tmp_path: Path) -> None:
    write_parts([_part("FLY:thorax", "DRIVER:RIG_Hips")], _rig(), tmp_path, scale=1.0)
    lines = (tmp_path / "FLY_thorax.obj").read_text(encoding="utf-8").splitlines()
    assert "o FLY:thorax" in lines and "usemtl FLY_body" in lines
    assert [line for line in lines if line.startswith("f ")] == ["f 1 2 3", "f 4 5 6"]
    assert "Kd 0.8000 0.5000 0.2000" in (tmp_path / "FLY_thorax.mtl").read_text(encoding="utf-8")


def test_duplicate_names_and_missing_bones_are_refused(tmp_path: Path) -> None:
    twice = [_part("FLY:a", "DRIVER:RIG_Hips"), _part("FLY:a", "DRIVER:RIG_Head")]
    with pytest.raises(ValueError, match="unique"):
        write_parts(twice, _rig(), tmp_path, scale=1.0)
    with pytest.raises(ValueError, match="RIG_Arm_L"):
        write_parts([_part("FLY:a", "DRIVER:RIG_Arm_L")], _rig(), tmp_path, scale=1.0)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("DRIVER:DRIVER", True),
        ("DRIVER:RIG_Nek", True),
        ("DRIVER:HAND_Index4", True),
        ("DRIVER:HELMET", False),
        ("DRIVER:COLLARE_HANS1", False),
        ("DRIVER:Driver_Body_SUB0", False),
    ],
)
def test_the_skeleton_is_bones_only(name: str, expected: bool) -> None:
    assert is_skeleton_node(name) is expected
