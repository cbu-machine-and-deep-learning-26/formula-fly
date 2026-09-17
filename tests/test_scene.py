"""MJCF scene tests (GH-16).

Two of these exist because the bug actually happened during development and was invisible
in the numbers:

- **Face winding.** Getting the vertex order backwards renders a surface back-faced, so it
  vanishes when viewed from above. Nothing raises; the track is simply not there. The
  kerbs shipped inverted until a render caught it.
- **Clipping planes.** MuJoCo scales ``znear``/``zfar`` by *model extent*, which it infers
  from the bounding box. On a 1.7 km circuit that puts the near plane tens of metres out
  and a cockpit camera renders pure sky. Pinned here so nobody "simplifies" the
  ``<statistic>`` element away.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fly_driver.envs.centerline import Centerline
from fly_driver.envs.scene import SceneConfig, _ribbon_mesh, build_scene_xml


@pytest.fixture(scope="module")
def track() -> Centerline:
    return Centerline.load()


@pytest.fixture(scope="module")
def square() -> Centerline:
    """Small synthetic track; compiles far faster than Silverstone for option tests."""
    points = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    return Centerline(points=points, half_width_right=[5.0] * 4, half_width_left=[5.0] * 4)


def _geom_names(model: mujoco.MjModel) -> set[str]:
    return {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) for i in range(model.ngeom)}


class TestRibbonWinding:
    """Every generated face must point up. This is the bug that renders as nothing."""

    def _normals(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        arclength = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(left, axis=0), axis=1))]
        )
        vertex_str, face_str, _ = _ribbon_mesh(left, right, 0.02, arclength, 8.0)
        vertices = np.fromstring(vertex_str, sep=" ").reshape(-1, 3)
        faces = np.fromstring(face_str, sep=" ", dtype=int).reshape(-1, 3)
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        return np.cross(b - a, c - a)

    def test_straight_ribbon_faces_up(self):
        left = np.array([[0.0, 5.0], [10.0, 5.0], [20.0, 5.0]])
        right = np.array([[0.0, -5.0], [10.0, -5.0], [20.0, -5.0]])
        assert np.all(self._normals(left, right)[:, 2] > 0)

    def test_swapping_left_and_right_faces_down(self):
        """Confirms the test can actually detect the failure it guards against."""
        left = np.array([[0.0, 5.0], [10.0, 5.0], [20.0, 5.0]])
        right = np.array([[0.0, -5.0], [10.0, -5.0], [20.0, -5.0]])
        assert np.all(self._normals(right, left)[:, 2] < 0)

    def test_curved_ribbon_faces_up(self):
        """Counter-clockwise arc. Travelling CCW, the left of the direction of travel is
        the *inner* radius -- at angle 0 the tangent is +y so the left normal is -x, which
        points at the origin. Using the outer radius as "left" is the winding mistake this
        guards against."""
        angles = np.linspace(0, np.pi / 2, 12)
        inner = np.stack([45 * np.cos(angles), 45 * np.sin(angles)], axis=1)
        outer = np.stack([55 * np.cos(angles), 55 * np.sin(angles)], axis=1)
        assert np.all(self._normals(inner, outer)[:, 2] > 0)

    def test_vertex_and_face_counts(self):
        left = np.zeros((7, 2))
        left[:, 0] = np.arange(7) * 10.0
        left[:, 1] = 5.0
        right = left.copy()
        right[:, 1] = -5.0
        arclength = np.arange(7) * 10.0
        vertex_str, face_str, texcoord_str = _ribbon_mesh(left, right, 0.02, arclength, 8.0)
        assert np.fromstring(vertex_str, sep=" ").reshape(-1, 3).shape == (14, 3)
        assert np.fromstring(face_str, sep=" ", dtype=int).reshape(-1, 3).shape == (12, 3)
        assert np.fromstring(texcoord_str, sep=" ").reshape(-1, 2).shape == (14, 2)


class TestSceneCompiles:
    def test_silverstone_compiles(self, track):
        model = mujoco.MjModel.from_xml_string(build_scene_xml(track))
        assert model.ngeom > 0

    def test_expected_geoms_present(self, track):
        names = _geom_names(mujoco.MjModel.from_xml_string(build_scene_xml(track)))
        assert {"ground", "road_geom", "kerb_left_geom", "kerb_right_geom"} <= names

    def test_forward_dynamics_runs(self, square):
        model = mujoco.MjModel.from_xml_string(build_scene_xml(square))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert np.all(np.isfinite(data.qpos))

    def test_track_surface_is_not_collidable(self, square):
        """Physics is on the ground plane; the ribbon is visual only."""
        model = mujoco.MjModel.from_xml_string(build_scene_xml(square))
        road = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "road_geom")
        assert model.geom_contype[road] == 0
        assert model.geom_conaffinity[road] == 0

    def test_ground_plane_is_collidable(self, square):
        model = mujoco.MjModel.from_xml_string(build_scene_xml(square))
        ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
        assert model.geom_contype[ground] != 0


class TestClippingPlanes:
    def test_model_extent_is_pinned_to_car_scale(self, track):
        """Not the 1.7 km bounding box, or a cockpit camera sees only sky."""
        model = mujoco.MjModel.from_xml_string(build_scene_xml(track))
        assert model.stat.extent == pytest.approx(SceneConfig().model_extent_m)

    def test_near_plane_is_closer_than_the_camera_mount(self, track):
        """znear is in extents; it must resolve to well under a metre."""
        config = SceneConfig()
        model = mujoco.MjModel.from_xml_string(build_scene_xml(track, config))
        assert model.vis.map.znear * model.stat.extent < 0.5

    def test_far_plane_reaches_down_a_straight(self, track):
        config = SceneConfig()
        model = mujoco.MjModel.from_xml_string(build_scene_xml(track, config))
        assert model.vis.map.zfar * model.stat.extent > 300.0

    def test_offscreen_framebuffer_is_declared(self, square):
        config = SceneConfig(offscreen_width=800, offscreen_height=600)
        model = mujoco.MjModel.from_xml_string(build_scene_xml(square, config))
        assert model.vis.global_.offwidth == 800
        assert model.vis.global_.offheight == 600


class TestSceneOptions:
    def test_kerbs_can_be_disabled(self, square):
        names = _geom_names(
            mujoco.MjModel.from_xml_string(build_scene_xml(square, SceneConfig(kerb_width_m=0.0)))
        )
        assert not any(name.startswith("kerb") for name in names)

    def test_walls_are_off_by_default(self, square):
        names = _geom_names(mujoco.MjModel.from_xml_string(build_scene_xml(square)))
        assert not any(name.startswith("wall_") for name in names)

    def test_walls_can_be_enabled_and_are_collidable(self, square):
        model = mujoco.MjModel.from_xml_string(
            build_scene_xml(square, SceneConfig(include_walls=True, mesh_spacing_m=25.0))
        )
        wall_ids = [
            i
            for i in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("wall_")
        ]
        assert wall_ids
        assert all(model.geom_contype[i] != 0 for i in wall_ids)

    def test_mesh_spacing_controls_triangle_count(self, track):
        coarse = mujoco.MjModel.from_xml_string(
            build_scene_xml(track, SceneConfig(mesh_spacing_m=40.0))
        )
        fine = mujoco.MjModel.from_xml_string(
            build_scene_xml(track, SceneConfig(mesh_spacing_m=10.0))
        )
        assert fine.nmeshface > coarse.nmeshface

    def test_timestep_is_applied(self, square):
        model = mujoco.MjModel.from_xml_string(build_scene_xml(square, SceneConfig(timestep=0.004)))
        assert model.opt.timestep == pytest.approx(0.004)


class TestSceneConfigValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mesh_spacing_m": 0.0},
            {"texture_repeat_m": -1.0},
            {"grass_texture_repeat_m": 0.0},
            {"kerb_width_m": -0.5},
            {"timestep": 0.0},
            {"offscreen_width": 0},
            {"model_extent_m": -1.0},
            {"znear_extents": 0.0},
            {"zfar_extents": 0.001},
        ],
    )
    def test_rejects_bad_values(self, kwargs):
        with pytest.raises(ValueError):
            SceneConfig(**kwargs)


class TestExtraHooks:
    def test_extra_bodies_and_actuators_are_injected(self, square):
        xml = build_scene_xml(
            square,
            extra_bodies=(
                '<body name="probe" pos="0 0 1">'
                '<joint name="slide" type="slide" axis="0 0 1"/>'
                '<geom type="sphere" size="0.2"/></body>'
            ),
            extra_actuators='<motor name="push" joint="slide" gear="1"/>',
        )
        model = mujoco.MjModel.from_xml_string(xml)
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe") != -1
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "push") != -1
