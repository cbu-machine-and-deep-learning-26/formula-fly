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

import re

import mujoco
import numpy as np
import pytest

from fly_driver.envs.centerline import Centerline
from fly_driver.envs.scene import (
    SceneConfig,
    _ribbon_mesh,
    build_scene_xml,
    checkpoint_pillars_xml,
    scenery_xml,
)


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

    def test_the_friction_cone_is_elliptic(self, square):
        """The default pyramidal cone under-delivers lateral tyre force badly enough to
        change the car: at 250 km/h and moderate lock it produced 2.6 g where the tyre
        and aero figures say 4.5, and the steered front wheels skipped along the road.
        This is the single option that fixed cornering, so it is pinned."""
        model = mujoco.MjModel.from_xml_string(build_scene_xml(square, SceneConfig()))
        assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC

    def test_impratio_is_raised_for_accurate_friction(self, square):
        """MuJoCo's own guidance: with an elliptic cone, raise impratio well above 1 or
        friction stays soft."""
        model = mujoco.MjModel.from_xml_string(build_scene_xml(square, SceneConfig()))
        assert model.opt.impratio >= 5.0

    def test_the_cone_can_be_switched_back(self, square):
        model = mujoco.MjModel.from_xml_string(
            build_scene_xml(square, SceneConfig(friction_cone="pyramidal"))
        )
        assert model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL


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
            {"friction_cone": "conical"},
            {"scenery_spacing_m": -1.0},
            {"scenery_margin_m": -1.0},
            {"scenery_density": 1.5},
            {"grid_offset_m": -1.0},
            {"impratio": 0.5},
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


class TestScenery:
    """Payton asked for trackside objects as "a secondary reference of how fast the car is
    going". They also give the fly's motion detectors real parallax, which flat ground
    texture cannot, so they are on by default.
    """

    @pytest.fixture(scope="class")
    def scenery_model(self, track):
        return mujoco.MjModel.from_xml_string(build_scene_xml(track, SceneConfig()))

    def _scenery_geoms(self, model):
        found = []
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name.startswith("scenery_"):
                found.append(i)
        return found

    def test_there_is_scenery(self, scenery_model):
        assert len(self._scenery_geoms(scenery_model)) > 50

    def test_none_of_it_can_be_hit(self, scenery_model):
        """Visual only, like the kerbs. A tree that stopped the car dead would change
        every lap time in the record book."""
        for i in self._scenery_geoms(scenery_model):
            assert scenery_model.geom_contype[i] == 0
            assert scenery_model.geom_conaffinity[i] == 0

    def test_it_adds_no_mass(self, track):
        with_it = mujoco.MjModel.from_xml_string(build_scene_xml(track, SceneConfig()))
        without = mujoco.MjModel.from_xml_string(
            build_scene_xml(track, SceneConfig(scenery_spacing_m=0.0))
        )
        assert with_it.body_mass.sum() == pytest.approx(without.body_mass.sum())

    def test_it_is_drawn_by_the_head_camera(self, scenery_model):
        """Group 1, the same group as the road and kerbs. Hidden scenery would be a
        speed reference the fly cannot see."""
        for i in self._scenery_geoms(scenery_model):
            assert scenery_model.geom_group[i] == 1

    def test_nothing_stands_on_the_track(self, track, scenery_model):
        """Every object must clear the kerb by the configured margin, or a car running
        wide would drive through a building."""
        config = SceneConfig()
        for i in self._scenery_geoms(scenery_model):
            x, y = scenery_model.geom_pos[i][:2]
            projection = track.project(float(x), float(y))
            edge = (
                projection.half_width_left
                if projection.lateral >= 0
                else projection.half_width_right
            )
            clearance = abs(projection.lateral) - edge
            # crowds scatter a couple of metres around their anchor point
            assert clearance > config.scenery_margin_m - 3.0, (
                f"scenery geom {i} is only {clearance:.1f} m past the edge"
            )

    def test_it_stays_within_the_configured_spread(self, track, scenery_model):
        config = SceneConfig()
        reach = config.scenery_margin_m + config.scenery_spread_m + 8.0
        for i in self._scenery_geoms(scenery_model):
            x, y = scenery_model.geom_pos[i][:2]
            projection = track.project(float(x), float(y))
            edge = (
                projection.half_width_left
                if projection.lateral >= 0
                else projection.half_width_right
            )
            assert abs(projection.lateral) - edge < reach

    def test_the_same_seed_gives_the_same_circuit(self, track):
        """An eval protocol that rearranged the scenery between runs would not be
        reproducible, which AGENTS.md section 11 asks to be pinned rather than assumed."""
        first = scenery_xml(track, SceneConfig(scenery_seed=7))
        second = scenery_xml(track, SceneConfig(scenery_seed=7))
        assert first == second

    def test_a_different_seed_gives_a_different_circuit(self, track):
        assert scenery_xml(track, SceneConfig(scenery_seed=1)) != scenery_xml(
            track, SceneConfig(scenery_seed=2)
        )

    def test_it_can_be_switched_off(self, track):
        assert scenery_xml(track, SceneConfig(scenery_spacing_m=0.0)) == ""
        assert scenery_xml(track, SceneConfig(scenery_density=0.0)) == ""

    def test_density_controls_how_much_there_is(self, track):
        sparse = scenery_xml(track, SceneConfig(scenery_density=0.2)).count("<geom")
        dense = scenery_xml(track, SceneConfig(scenery_density=1.0)).count("<geom")
        assert dense > sparse > 0

    def test_all_three_kinds_appear(self, track):
        """Trees, buildings and crowds. A crowd gives human scale, which is what makes
        300 km/h read as 300 km/h."""
        xml = scenery_xml(track, SceneConfig())
        assert "_trunk" in xml and "_crown" in xml
        assert "_walls" in xml and "_roof" in xml
        assert 'type="capsule"' in xml

    def test_everything_sits_on_or_above_the_ground(self, scenery_model):
        for i in self._scenery_geoms(scenery_model):
            assert scenery_model.geom_pos[i][2] >= 0.0


class TestCheckpointPillars:
    """Grey posts at every segment boundary, marking where each timed checkpoint starts.

    The failure worth guarding is not that they look wrong -- it is that they become
    solid. A collidable post a metre off the racing line would rewrite every lap time on
    the circuit, and nothing about the geometry would look different.

    Placement is checked against Silverstone rather than the ``square`` fixture. The square
    is 100 m on a side and 10 m wide, so a post set 7.5 m outside one edge lands exactly on
    the centerline of the edge opposite -- a degenerate case for any "is this clear of the
    circuit" question, and nothing to do with whether the code is right.
    """

    @pytest.fixture(scope="class")
    def circuit(self):
        from fly_driver.envs.centerline import Centerline

        return Centerline.load()

    @staticmethod
    def _positions(xml: str) -> list[tuple[float, float, float]]:
        return [
            tuple(float(value) for value in match.split())
            for match in re.findall(r'pos="([^"]+)"', xml)
        ]

    def test_there_is_a_pair_at_every_segment_boundary(self, circuit):
        from fly_driver.envs.segments import find_segments

        xml = checkpoint_pillars_xml(circuit, SceneConfig())
        assert xml.count("<geom") == 2 * len(find_segments(circuit))

    def test_they_are_named_after_the_segment_they_mark(self, circuit):
        """So a pillar in the render can be tied back to the split it belongs to."""
        from fly_driver.envs.segments import find_segments

        xml = checkpoint_pillars_xml(circuit, SceneConfig())
        for segment in find_segments(circuit):
            name = segment.name.replace(" ", "_")
            assert f'name="pillar_{name}_left"' in xml
            assert f'name="pillar_{name}_right"' in xml

    def test_nothing_can_hit_them(self, square):
        """Visual only, like the kerbs and the scenery. A driver running wide should be
        paying the track-limits penalty, not bouncing off furniture."""
        model = mujoco.MjModel.from_xml_string(build_scene_xml(square, SceneConfig()))
        found = False
        for index in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
            if name and name.startswith("pillar_"):
                found = True
                assert model.geom_contype[index] == 0
                assert model.geom_conaffinity[index] == 0
        assert found, "no pillars in the model, so nothing was actually checked"

    def test_they_stand_clear_of_the_circuit(self, circuit):
        """Outside the kerb on both sides. Inside it they would be in the way, and a post
        on the racing line would quietly rewrite every lap time on the circuit."""
        config = SceneConfig()
        for x, y, _ in self._positions(checkpoint_pillars_xml(circuit, config)):
            projection = circuit.project(x, y)
            edge = (
                projection.half_width_left
                if projection.lateral >= 0.0
                else projection.half_width_right
            )
            assert abs(projection.lateral) > edge + config.kerb_width_m

    def test_they_do_not_stand_so_far_out_they_mark_nothing(self, circuit):
        config = SceneConfig()
        for x, y, _ in self._positions(checkpoint_pillars_xml(circuit, config)):
            projection = circuit.project(x, y)
            assert abs(projection.lateral) < 30.0

    def test_a_bigger_margin_pushes_them_further_out(self, circuit):
        def spread(margin: float) -> float:
            xml = checkpoint_pillars_xml(circuit, SceneConfig(pillar_margin_m=margin))
            return max(abs(circuit.project(x, y).lateral) for x, y, _ in self._positions(xml))

        assert spread(6.0) > spread(1.5)

    def test_one_goes_each_side(self, circuit):
        """A pair, not two posts on the same side -- which would still count right."""
        xml = checkpoint_pillars_xml(circuit, SceneConfig())
        laterals = [circuit.project(x, y).lateral for x, y, _ in self._positions(xml)]
        assert sum(value > 0 for value in laterals) == sum(value < 0 for value in laterals)

    def test_they_stand_on_the_ground(self, square):
        """Centred at half their height, or they float or sink."""
        config = SceneConfig()
        model = mujoco.MjModel.from_xml_string(build_scene_xml(square, config))
        for index in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
            if name and name.startswith("pillar_"):
                assert model.geom_pos[index][2] == pytest.approx(config.pillar_height_m / 2)

    def test_they_are_tall_and_thin(self):
        """A fly's eye reads a tall narrow high-contrast object as an expansion cue; a
        squat one gives it much less to work with (AGENTS.md section 6)."""
        config = SceneConfig()
        assert config.pillar_height_m > 4.0 * config.pillar_radius_m

    def test_they_can_be_turned_off(self, square):
        assert checkpoint_pillars_xml(square, SceneConfig(checkpoint_pillars=False)) == ""

    @pytest.mark.parametrize(
        "kwargs",
        [{"pillar_radius_m": 0.0}, {"pillar_height_m": -1.0}, {"pillar_margin_m": 0.0}],
    )
    def test_their_dimensions_must_be_positive(self, kwargs):
        with pytest.raises(ValueError, match="must be positive"):
            SceneConfig(**kwargs)
