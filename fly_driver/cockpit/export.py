"""Write the posed fly for Blender: one OBJ per part and a manifest of bones and parts (GH-25).

``scripts/cockpit/build_fly_driver_fbx.py`` runs inside Blender and reads what this writes.
The two halves meet only through these files, so neither needs the other's Python.

OBJ vertices are in Assetto Corsa coordinates (metres, y up, z front). Blender's OBJ importer
with its default axes (up Y, forward -Z) turns that into Blender's z-up frame, and the FBX
exporter's defaults turn it back, so the numbers that reach ksEditor are the numbers written
here.

The manifest carries the stock driver's node hierarchy with each node's world matrix in the
same Direct3D layout as the ``.knh`` it came from (translation in row 3), plus, for every
part, the bone it rides on and its OBJ file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fly_driver.cockpit.knh import KnhNode, world_matrices
from fly_driver.cockpit.seated_fly import BodyPart

__all__ = ["MANIFEST_NAME", "is_skeleton_node", "write_parts"]

MANIFEST_NAME = "manifest.json"
_MANIFEST_VERSION = 1


def write_parts(
    parts: Sequence[BodyPart],
    rig: KnhNode,
    output_dir: str | os.PathLike[str],
    *,
    scale: float,
    source: dict[str, Any] | None = None,
) -> Path:
    """Write ``parts`` as OBJ/MTL pairs and a manifest next to them.

    Args:
        parts: From :func:`~fly_driver.cockpit.seated_fly.body_parts`.
        rig: The driver's node tree from the car's ``driver_base_pos.knh``. Only its
            skeleton is written (:func:`is_skeleton_node`); the editor's wrapper nodes above
            ``DRIVER:DRIVER`` and Kunos's own meshes are not part of the fly.
        output_dir: Created if missing. Existing files with the same names are replaced.
        scale: Metres per MuJoCo unit, recorded for reference.
        source: Anything worth recording about where the inputs came from.

    Returns:
        The manifest's path.

    Raises:
        ValueError: If two parts share a name, or a part rides on a bone the rig lacks.
    """
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    driver_root = _find(rig, "DRIVER:DRIVER")
    bones = _bone_table(driver_root)
    names = [part.name for part in parts]
    if len(set(names)) != len(names):
        raise ValueError("part names must be unique; Assetto Corsa crashes on duplicates")
    missing = sorted({part.bone for part in parts} - {bone["name"] for bone in bones})
    if missing:
        raise ValueError(f"parts ride on bones the rig does not have: {missing}")

    part_rows = []
    for part in parts:
        stem = _file_stem(part.name)
        _write_obj(part, target / f"{stem}.obj", target / f"{stem}.mtl")
        part_rows.append(
            {
                "name": part.name,
                "bone": part.bone,
                "bodies": list(part.bodies),
                "obj": f"{stem}.obj",
                "vertices": int(sum(len(piece.vertices) for piece in part.pieces)),
            }
        )
    manifest = {
        "version": _MANIFEST_VERSION,
        "units": "metres",
        "axes": "Assetto Corsa: +x driver's left, +y up, +z front; Direct3D row-vector matrices",
        "scale_m_per_mujoco_unit": scale,
        "source": source or {},
        "bones": bones,
        "parts": part_rows,
    }
    path = target / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _find(node: KnhNode, name: str) -> KnhNode:
    for candidate in node.walk():
        if candidate.name == name:
            return candidate
    raise ValueError(f"the rig has no {name!r} node")


def is_skeleton_node(name: str) -> bool:
    """Whether a stock driver node is a bone rather than one of Kunos's meshes.

    The ``.knh`` also lists the stock helmet, HANS collar and skinned body. Those are meshes
    the fly replaces; exporting nulls with their names would collide with the fly's own
    ``DRIVER:HELMET``, and the game crashes on duplicate names.
    """
    return name == "DRIVER:DRIVER" or name.startswith(("DRIVER:RIG_", "DRIVER:HAND_"))


def _bone_table(driver_root: KnhNode) -> list[dict[str, Any]]:
    """The skeleton under ``DRIVER:DRIVER``, parents before children, with world matrices.

    Only :func:`is_skeleton_node` nodes; a stock mesh's subtree is skipped whole.
    """
    worlds = world_matrices(driver_root)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def visit(node: KnhNode, parent: str | None) -> None:
        if node.name in seen or not is_skeleton_node(node.name):
            return
        seen.add(node.name)
        rows.append(
            {
                "name": node.name,
                "parent": parent,
                "world": np.round(worlds[node.name], 7).tolist(),
            }
        )
        for child in node.children:
            visit(child, node.name)

    visit(driver_root, None)
    return rows


def _file_stem(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in name)


def _write_obj(part: BodyPart, obj_path: Path, mtl_path: Path) -> None:
    materials: dict[str, tuple[float, float, float, float]] = {}
    lines = [f"# {part.name} on {part.bone}", f"mtllib {mtl_path.name}", f"o {part.name}"]
    offset = 1
    for piece in part.pieces:
        materials.setdefault(piece.material, piece.rgba)
        lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in piece.vertices)
        lines.append(f"usemtl FLY_{piece.material}")
        lines.extend(f"f {a + offset} {b + offset} {c + offset}" for a, b, c in piece.faces)
        offset += len(piece.vertices)
    obj_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    mtl = []
    for name, (red, green, blue, alpha) in materials.items():
        mtl += [f"newmtl FLY_{name}", f"Kd {red:.4f} {green:.4f} {blue:.4f}", f"d {alpha:.4f}", ""]
    mtl_path.write_text("\n".join(mtl), encoding="utf-8")
