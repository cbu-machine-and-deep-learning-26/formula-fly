"""Pose flybody's fly on the SF70H's driver skeleton and write it out for Blender (GH-25).

Reads the car's ``driver_base_pos.knh`` from the local Assetto Corsa install and flybody's
``fruitfly.xml`` from a flybody checkout, and writes one OBJ per fly part plus
``manifest.json`` to ``--out`` (default ``outputs/cockpit/parts``, git-ignored). Nothing from
Assetto Corsa is copied into the repo. Next step: ``scripts/cockpit/build_fly_driver_fbx.py``
in Blender; see ``docs/fly-driver.md``.

Example:
    python scripts/cockpit/pose_fly_driver.py
    python scripts/cockpit/pose_fly_driver.py --preview outputs/cockpit/pose.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fly_driver.cockpit import body_parts, pose_seated_fly, read_knh, world_matrices  # noqa: E402
from fly_driver.cockpit.export import write_parts  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_AC_ROOT = Path(r"C:\Program Files (x86)\Steam\steamapps\common\assettocorsa")
DEFAULT_FRUITFLY_XML = Path.home() / ".cache/flybody-src/flybody/fruitfly/assets/fruitfly.xml"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ac-root", type=Path, default=DEFAULT_AC_ROOT)
    parser.add_argument("--car", default="ks_ferrari_sf70h")
    parser.add_argument("--fruitfly-xml", type=Path, default=DEFAULT_FRUITFLY_XML)
    parser.add_argument("--out", type=Path, default=REPO / "outputs/cockpit/parts")
    parser.add_argument("--preview", type=Path, help="also render the pose with MuJoCo to PNG")
    parser.add_argument(
        "--size", type=float, default=1.3, help="times the hips-to-head fit; 1.3 reaches the wheel"
    )
    parser.add_argument(
        "--recline", type=float, default=30.0, help="degrees further back than the driver's line"
    )
    return parser.parse_args(argv)


def _render_preview(seated, path: Path) -> None:  # noqa: ANN001 - SeatedFly
    import imageio.v3 as iio
    import mujoco
    import numpy as np

    renderer = mujoco.Renderer(seated.model, 600, 800)
    thorax = seated.data.xpos[seated.model.body("thorax").id]
    frames = []
    for azimuth, elevation in ((0.0, -10.0), (200.0, -25.0)):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = thorax + np.array([0.15, 0.0, 0.05]) / seated.scale
        camera.distance = 2.2 / seated.scale
        camera.azimuth, camera.elevation = azimuth, elevation
        renderer.update_scene(seated.data, camera)
        frames.append(renderer.render())
    renderer.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, np.concatenate(frames, axis=1))


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    args = _parse_args(argv)
    knh_path = args.ac_root / "content/cars" / args.car / "driver_base_pos.knh"
    for required in (knh_path, args.fruitfly_xml):
        if not required.exists():
            print(f"missing {required}; see docs/fly-driver.md", file=sys.stderr)
            return 2
    rig = read_knh(knh_path)
    seated = pose_seated_fly(
        args.fruitfly_xml, world_matrices(rig), size=args.size, recline_deg=args.recline
    )
    parts = body_parts(seated)
    manifest = write_parts(
        parts,
        rig,
        args.out,
        scale=seated.scale,
        source={
            "car": args.car,
            "fruitfly_xml": str(args.fruitfly_xml),
            "size": args.size,
            "recline_deg": args.recline,
        },
    )
    print(f"scale {seated.scale:.3f} m per MuJoCo unit; {len(parts)} parts -> {manifest}")
    for element, miss in seated.target_misses.items():
        print(f"  {element:16s} misses its joint by {miss * 100:5.1f} cm")
    if args.preview is not None:
        _render_preview(seated, args.preview)
        print(f"preview: {args.preview}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
