"""MJCF scene generation for the practice track (GH-16).

Builds the world the car drives in: a ground plane, a track ribbon swept from the
centerline, and lighting. The car itself is injected by :mod:`fly_driver.envs.car` through
the ``extra_*`` hooks, so world geometry and vehicle stay separable.

Two decisions worth knowing about before changing anything here.

**The track surface is visual, not collidable.** Physics happens on a flat infinite ground
plane; whether the car is on track is decided analytically by
:meth:`~fly_driver.envs.centerline.Centerline.project`. Silverstone is essentially flat, so
a collidable ribbon mesh would cost mesh-contact time every step and buy nothing. It also
sidesteps the car falling through a seam between triangles.

**Everything is textured on purpose.** This is the part that is easy to get wrong for this
project specifically. flyvis models motion vision: T4/T5 respond to *moving contrast*. A
uniform grey road on uniform green grass produces almost no optic flow anywhere except the
two track edges, so a fly eye looking at it would be nearly blind, and RQ1 would be
measuring the eye's response to an image that carries no motion signal. The checkerboard
ground, the textured asphalt, and the kerb stripes exist to give the retina something to
see. If you flatten these to solid colours to make the render faster, the eye stops
working and the failure looks like a training problem rather than a rendering one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fly_driver.envs.centerline import Centerline

__all__ = ["SceneConfig", "build_scene_xml"]


@dataclass(frozen=True)
class SceneConfig:
    """Tunables for the generated world.

    Args:
        mesh_spacing_m: Centerline resampling distance for the track ribbon. The raw data
            is ~5 m apart, which is finer than the visual mesh needs; 10 m halves the
            triangle count with no visible difference on straights.
        surface_height_m: How far the visual ribbon floats above the ground plane. Small
            but non-zero to avoid z-fighting with the plane.
        texture_repeat_m: Road texture repeat distance along the track, in metres. This
            sets the spatial frequency the eye sees when moving, so it interacts directly
            with T4/T5 tuning — it is a knob, not a cosmetic constant.
        kerb_width_m: Width of the striped edge strip. Set to 0 to disable kerbs.
        include_walls: Add collidable walls at the track edges. Off by default: the reward
            already penalises leaving the track, and walls add thousands of contact geoms
            that slow every step of every rollout.
        wall_height_m: Wall height when ``include_walls`` is set.
        timestep: Physics timestep in seconds. 2 ms keeps a rigid-body car with wheel
            contacts stable; larger values let the wheels tunnel under load.
        offscreen_width: Offscreen framebuffer width in pixels.
        offscreen_height: Offscreen framebuffer height in pixels. MuJoCo defaults these to
            640x480 and ``mujoco.Renderer`` **raises** if asked for anything larger, so the
            ceiling has to be declared in the model rather than discovered later. Whatever
            camera resolution GH-13 settles on must fit inside this; the env raises early
            if it does not.
        model_extent_m: Overrides MuJoCo's inferred model extent. This is not cosmetic:
            ``znear`` and ``zfar`` are expressed in *extents*, and MuJoCo infers extent
            from the scene bounding box. A 1.7 km circuit therefore puts the near clipping
            plane tens of metres in front of the camera and a cockpit view renders as an
            empty sky. Pinning extent to car scale fixes it.
        world_contype: Collision class of the ground and walls.
        world_conaffinity: What the ground and walls collide with. Defaults pair with the
            chassis and wheel classes in :mod:`fly_driver.envs.car` so that wheels touch
            the ground but never the bodywork or each other.
        znear_extents: Near clipping plane, in extents. With the default extent this is
            centimetres, which is what a camera 1 m off the ground needs.
        zfar_extents: Far clipping plane, in extents. Must cover how far down the track the
            eye should see; too small and the horizon vanishes mid-straight.
    """

    mesh_spacing_m: float = 10.0
    surface_height_m: float = 0.02
    texture_repeat_m: float = 8.0
    grass_texture_repeat_m: float = 5.0
    kerb_width_m: float = 0.75
    include_walls: bool = False
    wall_height_m: float = 0.6
    timestep: float = 0.002
    offscreen_width: int = 1280
    offscreen_height: int = 1280
    world_contype: int = 1
    world_conaffinity: int = 6
    model_extent_m: float = 10.0
    znear_extents: float = 0.005
    zfar_extents: float = 60.0

    def __post_init__(self) -> None:
        if self.mesh_spacing_m <= 0:
            raise ValueError(f"mesh_spacing_m must be positive, got {self.mesh_spacing_m}")
        if self.texture_repeat_m <= 0:
            raise ValueError(f"texture_repeat_m must be positive, got {self.texture_repeat_m}")
        if self.grass_texture_repeat_m <= 0:
            raise ValueError(
                f"grass_texture_repeat_m must be positive, got {self.grass_texture_repeat_m}"
            )
        if self.kerb_width_m < 0:
            raise ValueError(f"kerb_width_m must be non-negative, got {self.kerb_width_m}")
        if self.timestep <= 0:
            raise ValueError(f"timestep must be positive, got {self.timestep}")
        if self.offscreen_width <= 0 or self.offscreen_height <= 0:
            raise ValueError(
                f"offscreen framebuffer must be positive, got "
                f"{self.offscreen_width}x{self.offscreen_height}"
            )
        if self.model_extent_m <= 0:
            raise ValueError(f"model_extent_m must be positive, got {self.model_extent_m}")
        if not 0 < self.znear_extents < self.zfar_extents:
            raise ValueError(
                f"need 0 < znear_extents < zfar_extents, got "
                f"{self.znear_extents} and {self.zfar_extents}"
            )


def _ribbon_mesh(
    inner_left: np.ndarray,
    inner_right: np.ndarray,
    height: float,
    arclength: np.ndarray,
    texture_repeat_m: float,
) -> tuple[str, str, str]:
    """Build inline MJCF mesh data for a ribbon between two polylines.

    Vertices interleave as ``[right_0, left_0, right_1, left_1, ...]`` so segment ``i``
    uses indices ``2i..2i+3``.

    Winding is chosen so face normals point up (+z). With ``left`` on the ``(-ty, tx)``
    side of the tangent, ``(right_i, right_{i+1}, left_i)`` gives ``tangent x normal_left
    = +z``. Getting this backwards renders the surface invisible from above, which looks
    like a missing track rather than a winding bug.

    Returns:
        ``(vertex, face, texcoord)`` as space-separated MJCF attribute strings.
    """
    count = inner_left.shape[0]
    vertices = np.empty((count * 2, 3), dtype=np.float64)
    vertices[0::2, :2] = inner_right
    vertices[1::2, :2] = inner_left
    vertices[:, 2] = height

    faces = []
    for i in range(count - 1):
        r0, l0, r1, l1 = 2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3
        faces.append((r0, r1, l0))
        faces.append((r1, l1, l0))

    # u runs across the track (0 right, 1 left); v runs along it, repeating every
    # texture_repeat_m so the road carries visible motion at a controlled spatial frequency.
    v = arclength / texture_repeat_m
    texcoords = np.empty((count * 2, 2), dtype=np.float64)
    texcoords[0::2, 0] = 0.0
    texcoords[1::2, 0] = 1.0
    texcoords[0::2, 1] = v
    texcoords[1::2, 1] = v

    vertex_str = " ".join(f"{value:.4f}" for value in vertices.reshape(-1))
    face_str = " ".join(str(index) for index in np.asarray(faces, dtype=int).reshape(-1))
    texcoord_str = " ".join(f"{value:.4f}" for value in texcoords.reshape(-1))
    return vertex_str, face_str, texcoord_str


def _offset_polyline(points: np.ndarray, normals: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Offset ``points`` along ``normals`` by a per-point ``distance``."""
    return points + normals * distance[:, None]


def _segment_normals(centerline: Centerline) -> np.ndarray:
    """Left-pointing unit normal at each point of the closed polyline."""
    normals = np.empty_like(centerline.points)
    for i in range(centerline.num_segments):
        normals[i] = centerline.normal(i)
    normals[-1] = normals[0]
    return normals


def build_scene_xml(
    centerline: Centerline,
    config: SceneConfig | None = None,
    *,
    extra_assets: str = "",
    extra_bodies: str = "",
    extra_actuators: str = "",
    extra_tendons: str = "",
) -> str:
    """Generate the complete MJCF document for the practice track.

    Args:
        centerline: The track geometry. Resampled to ``config.mesh_spacing_m`` for the
            visual mesh; the original is kept for reward and off-track maths.
        config: Scene tunables. Defaults to :class:`SceneConfig`.
        extra_assets: MJCF fragment inserted into ``<asset>``, for the car's materials.
        extra_bodies: MJCF fragment inserted into ``<worldbody>``, for the car itself.
        extra_actuators: MJCF fragment inserted into ``<actuator>``.
        extra_tendons: MJCF fragment inserted into ``<tendon>``, for the car's anti-roll
            bars. The element is only emitted when non-empty.

    Returns:
        An MJCF document string, ready for ``mujoco.MjModel.from_xml_string``.
    """
    config = config or SceneConfig()
    mesh_line = centerline.resample(config.mesh_spacing_m)
    normals = _segment_normals(mesh_line)
    points = mesh_line.points
    arclength = mesh_line.arclength

    road_left = _offset_polyline(points, normals, mesh_line.half_width_left)
    road_right = _offset_polyline(points, normals, -mesh_line.half_width_right)
    road_vertex, road_face, road_texcoord = _ribbon_mesh(
        road_left, road_right, config.surface_height_m, arclength, config.texture_repeat_m
    )

    kerb_meshes = ""
    kerb_geoms = ""
    if config.kerb_width_m > 0:
        for side, sign in (("left", 1.0), ("right", -1.0)):
            width = mesh_line.half_width_left if sign > 0 else mesh_line.half_width_right
            inner = _offset_polyline(points, normals, sign * width)
            outer = _offset_polyline(points, normals, sign * (width + config.kerb_width_m))
            # _ribbon_mesh wants (left_polyline, right_polyline) in that order, where
            # "left" means further along the +normal direction. For the left kerb the
            # outer edge is the more-left one; for the right kerb it is the inner edge.
            # Swapping these flips the face winding and the kerb renders back-faced --
            # invisible from above, which looks like a missing geom rather than a bug.
            left_poly, right_poly = (outer, inner) if sign > 0 else (inner, outer)
            vertex, face, texcoord = _ribbon_mesh(
                left_poly,
                right_poly,
                config.surface_height_m + 0.005,
                arclength,
                # Short repeat: kerb stripes are the highest-contrast moving feature on
                # the track, which is what makes a corner legible to a motion detector.
                config.texture_repeat_m / 8.0,
            )
            kerb_meshes += (
                f'\n    <mesh name="kerb_{side}" inertia="shell" vertex="{vertex}" '
                f'face="{face}" texcoord="{texcoord}"/>'
            )
            kerb_geoms += (
                f'\n    <geom name="kerb_{side}_geom" type="mesh" mesh="kerb_{side}" '
                f'material="kerb" contype="0" conaffinity="0" group="1"/>'
            )

    walls = ""
    if config.include_walls:
        for side, sign in (("left", 1.0), ("right", -1.0)):
            width = mesh_line.half_width_left if sign > 0 else mesh_line.half_width_right
            edge = _offset_polyline(points, normals, sign * (width + config.kerb_width_m))
            for i in range(edge.shape[0] - 1):
                start, end = edge[i], edge[i + 1]
                mid = (start + end) / 2.0
                delta = end - start
                half_len = float(np.linalg.norm(delta)) / 2.0
                if half_len <= 0:
                    continue
                yaw = float(np.arctan2(delta[1], delta[0]))
                walls += (
                    f'\n    <geom name="wall_{side}_{i}" type="box" '
                    f'pos="{mid[0]:.3f} {mid[1]:.3f} {config.wall_height_m / 2:.3f}" '
                    f'euler="0 0 {yaw:.5f}" '
                    f'size="{half_len:.3f} 0.1 {config.wall_height_m / 2:.3f}" '
                    f'material="wall" group="2" '
                    f'contype="{config.world_contype}" conaffinity="{config.world_conaffinity}"/>'
                )

    start_position, start_yaw = centerline.pose_at(0.0)
    tendon_block = f"\n\n  <tendon>{extra_tendons}\n  </tendon>" if extra_tendons.strip() else ""
    # texuniform makes texrepeat world-scaled, so this is tiles per metre.
    grass_repeat = 1.0 / config.grass_texture_repeat_m

    return f"""<mujoco model="practice_track">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{config.timestep}" integrator="implicitfast"/>

  <!-- MuJoCo scales znear/zfar by the model's *extent*, and it infers extent from the
       bounding box unless told otherwise. A 1.7 km circuit gives extent ~1700 m, which
       pushes the default near plane to tens of metres and clips away everything a camera
       mounted 1 m off the ground can see. Pinning extent to car scale is what makes a
       cockpit view possible at all. -->
  <statistic extent="{config.model_extent_m}" center="0 0 0"/>

  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7" specular="0.1 0.1 0.1"/>
    <map znear="{config.znear_extents}" zfar="{config.zfar_extents}"/>
    <quality shadowsize="2048"/>
    <global offwidth="{config.offscreen_width}" offheight="{config.offscreen_height}"/>
  </visual>

  <asset>
    <texture name="skybox" type="skybox" builtin="gradient"
             rgb1="0.35 0.5 0.75" rgb2="0.75 0.82 0.9" width="256" height="256"/>
    <!-- Checkered grass. A solid colour would give the fly eye no optic flow off-track,
         so a car spinning into the runoff would see a blank field and the eye would
         report no motion at exactly the moment motion matters most. -->
    <!-- texuniform makes texrepeat world-scaled: {config.grass_texture_repeat_m} m per
         tile, so the checker squares are metres across. Set this too fine and the
         checker averages to flat green at any distance, which is the same as having no
         texture at all as far as a motion detector is concerned. -->
    <texture name="grass_tex" type="2d" builtin="checker"
             rgb1="0.13 0.30 0.13" rgb2="0.17 0.38 0.17" width="256" height="256"/>
    <material name="grass" texture="grass_tex"
              texrepeat="{grass_repeat:.4f} {grass_repeat:.4f}"
              texuniform="true"/>
    <texture name="asphalt_tex" type="2d" builtin="checker"
             rgb1="0.22 0.22 0.24" rgb2="0.30 0.30 0.33" width="128" height="128"/>
    <material name="asphalt" texture="asphalt_tex" specular="0.1" shininess="0.2"/>
    <texture name="kerb_tex" type="2d" builtin="checker"
             rgb1="0.85 0.1 0.1" rgb2="0.92 0.92 0.92" width="64" height="64"/>
    <material name="kerb" texture="kerb_tex"/>
    <material name="wall" rgba="0.8 0.8 0.85 1"/>

    <mesh name="road" inertia="shell"
          vertex="{road_vertex}"
          face="{road_face}"
          texcoord="{road_texcoord}"/>{kerb_meshes}
{extra_assets}
  </asset>

  <worldbody>
    <light name="sun" directional="true" pos="0 0 200" dir="0.2 0.3 -1"
           diffuse="0.8 0.8 0.8" specular="0.2 0.2 0.2" castshadow="true"/>
    <geom name="ground" type="plane" size="0 0 1" material="grass" friction="1.0 0.005 0.0001"
          contype="{config.world_contype}" conaffinity="{config.world_conaffinity}"/>
    <geom name="road_geom" type="mesh" mesh="road" material="asphalt"
          contype="0" conaffinity="0" group="1"/>{kerb_geoms}{walls}
    <site name="start" pos="{start_position[0]:.3f} {start_position[1]:.3f} 0.05"
          euler="0 0 {start_yaw:.5f}" size="0.5 0.1 0.05" type="box" rgba="1 1 1 1"/>
{extra_bodies}
  </worldbody>

  <actuator>{extra_actuators}
  </actuator>{tendon_block}
</mujoco>
"""
