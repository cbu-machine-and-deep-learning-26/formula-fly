"""Build the fly-driver FBX for ksEditor, inside Blender (GH-25).

Reads what ``scripts/cockpit/pose_fly_driver.py`` wrote (per-part OBJs and ``manifest.json``)
and writes an FBX that ksEditor can turn into a driver KN5:

- the stock driver's skeleton as nulls, named exactly as in the car's ``driver_base_pos.knh``
  and with its orientations, so the car's own ``steer.ksanim`` moves them;
- every fly part as a rigid mesh parented to its bone -- the way the stock helmet rides on
  the head bone (Kunos car pipeline guide, "Skinned mesh": a non-skinned object may be a child
  of a null);
- decimated to about 36k vertices (the stock driver KN5 is 11 MB), one UV set per mesh and a
  4x4 colour texture per material, because every mesh must have a UV set and ksPerPixel
  takes its colour from a texture.

Run it with Blender 4.5 (LTS), not from the project's Python::

    blender -b --factory-startup -P scripts/cockpit/build_fly_driver_fbx.py -- \\
        --parts outputs/cockpit/parts --out outputs/cockpit/fly_driver

**Axes.** Assetto Corsa is y-up with +z to the front and +x to the driver's left. The OBJs are
written in those axes; Blender's OBJ importer (up Y, forward -Z) maps a point ``(x, y, z)`` to
``(x, -z, y)``, and the FBX exporter's defaults map it back. Null orientations need the same
care: the exporter converts only root objects, so a null gets ``C @ M`` in Blender (``C`` the
axis map, ``M`` its knh world matrix as column vectors) and lands in the FBX as exactly ``M``.
Conjugating instead (``C @ M @ C^-1``) would look right in Blender and leave every part
rotated about its bone in the game.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Matrix

#: Assetto Corsa axes -> Blender axes: (x, y, z) -> (x, -z, y).
AC_TO_BLENDER = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))


def _parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", type=Path, required=True, help="pose_fly_driver.py --out")
    parser.add_argument("--out", type=Path, required=True, help="folder for FBX, textures, blend")
    parser.add_argument("--decimate", type=float, default=0.25, help="fraction of faces to keep")
    parser.add_argument("--name", default="fly_driver", help="file stem for the FBX and .blend")
    return parser.parse_args(argv)


def _reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0


def _build_skeleton(bones: list[dict]) -> dict[str, bpy.types.Object]:
    empties: dict[str, bpy.types.Object] = {}
    for bone in bones:  # parents come first in the manifest
        empty = bpy.data.objects.new(bone["name"], None)
        empty.empty_display_type = "ARROWS"
        empty.empty_display_size = 0.03
        bpy.context.scene.collection.objects.link(empty)
        if bone["parent"] is not None:
            empty.parent = empties[bone["parent"]]
        empty.matrix_world = AC_TO_BLENDER @ Matrix(bone["world"]).transposed()
        empties[bone["name"]] = empty
    bpy.context.view_layer.update()
    return empties


def _import_part(obj_path: Path, name: str) -> bpy.types.Object:
    before = set(bpy.data.objects)
    bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis="NEGATIVE_Z", up_axis="Y")
    imported = [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]
    if len(imported) != 1:
        raise RuntimeError(f"{obj_path}: expected one mesh object, got {len(imported)}")
    obj = imported[0]
    obj.name = name
    obj.data.name = name
    return obj


def _decimate(obj: bpy.types.Object, ratio: float) -> None:
    if ratio >= 1.0:
        return
    modifier = obj.modifiers.new("decimate", "DECIMATE")
    modifier.ratio = ratio
    with bpy.context.temp_override(object=obj, active_object=obj, selected_objects=[obj]):
        bpy.ops.object.modifier_apply(modifier=modifier.name)


def _share_materials(obj: bpy.types.Object, materials: dict[str, bpy.types.Material]) -> None:
    """One material per flybody colour: the importer makes ``FLY_body.001`` and so on."""
    for slot in obj.material_slots:
        if slot.material is None:
            continue
        base = slot.material.name.split(".")[0]
        materials.setdefault(base, slot.material)
        slot.material = materials[base]


def _texture_materials(materials: dict[str, bpy.types.Material], out: Path) -> None:
    """Give each material a 4x4 texture of its colour: ksPerPixel takes colour from txDiffuse."""
    for name, material in materials.items():
        material.name = name
        material.use_nodes = True
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        colour = list(bsdf.inputs["Base Color"].default_value)
        colour[3] = bsdf.inputs["Alpha"].default_value
        image = bpy.data.images.new(name, width=4, height=4, alpha=True)
        image.pixels = colour * 16
        image.filepath_raw = str(out / f"{name}.png")
        image.file_format = "PNG"
        image.save()
        texture = material.node_tree.nodes.new("ShaderNodeTexImage")
        texture.image = image
        material.node_tree.links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])


def _render_preview(path: Path) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.color_type = "MATERIAL"
    scene.render.resolution_x, scene.render.resolution_y = 900, 700
    camera = bpy.data.objects.new("preview_camera", bpy.data.cameras.new("preview_camera"))
    scene.collection.objects.link(camera)
    # In front of the driver and to his right (the car's front is Blender -Y), looking back.
    camera.location = (-1.6, -3.0, 1.2)
    target = bpy.data.objects.new("preview_target", None)
    scene.collection.objects.link(target)
    target.location = (0.0, 0.0, 0.42)
    constraint = camera.constraints.new("TRACK_TO")
    constraint.target, constraint.track_axis, constraint.up_axis = (
        target,
        "TRACK_NEGATIVE_Z",
        "UP_Y",
    )
    scene.camera = camera
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(camera)
    bpy.data.objects.remove(target)


def main() -> None:
    args = _parse_args()
    # Absolute, or Blender resolves them against its own folder and the textures vanish.
    args.parts, args.out = args.parts.resolve(), args.out.resolve()
    manifest = json.loads((args.parts / "manifest.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    _reset_scene()
    empties = _build_skeleton(manifest["bones"])
    materials: dict[str, bpy.types.Material] = {}
    vertices = 0
    for part in manifest["parts"]:
        obj = _import_part(args.parts / part["obj"], part["name"])
        _decimate(obj, args.decimate)
        if not obj.data.uv_layers:
            obj.data.uv_layers.new(name="UVMap")
        _share_materials(obj, materials)
        world = obj.matrix_world.copy()
        obj.parent = empties[part["bone"]]
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_world = world
        vertices += len(obj.data.vertices)
    bpy.context.view_layer.update()
    _texture_materials(materials, args.out)

    fbx = args.out / f"{args.name}.fbx"
    bpy.ops.export_scene.fbx(
        filepath=str(fbx),
        object_types={"EMPTY", "MESH"},
        use_mesh_modifiers=True,
        mesh_smooth_type="FACE",
        apply_unit_scale=True,
        apply_scale_options="FBX_SCALE_UNITS",
        axis_forward="-Z",
        axis_up="Y",
        bake_space_transform=False,
        add_leaf_bones=False,
        path_mode="COPY",
        embed_textures=False,
    )
    bpy.ops.wm.save_as_mainfile(filepath=str(args.out / f"{args.name}.blend"))
    _render_preview(args.out / f"{args.name}_preview.png")
    print(
        f"FLY_DRIVER: {len(empties)} nulls, {len(manifest['parts'])} parts, {vertices} vertices, "
        f"{len(materials)} materials -> {fbx}"
    )


main()
