"""Bodywork and cockpit tests (GH-16).

The bodywork is visual only. These tests exist to keep it that way -- a stray contact flag
on a wing would silently change how the car drives -- and to pin the one thing about it
that the science depends on: how much of the fly's view it takes up.

They also pin the ``fly_mount`` site, which is the frame the body track (GH-21, GH-25,
GH-26) builds against. Moving it moves the fly.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from fly_driver.envs.car import (
    CHASSIS_CONTYPE,
    CarConfig,
    assemble_model_xml,
)
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.scene import SceneConfig

CONFIG = CarConfig()


@pytest.fixture(scope="module")
def model() -> mujoco.MjModel:
    points = [(0.0, 0.0), (400.0, 0.0), (400.0, 200.0), (0.0, 200.0)]
    track = Centerline(points=points, half_width_right=[8.0] * 4, half_width_left=[8.0] * 4)
    return mujoco.MjModel.from_xml_string(
        assemble_model_xml(track, SceneConfig(mesh_spacing_m=50.0), CONFIG)
    )


@pytest.fixture(scope="module")
def data(model) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data


def _geom_ids(model, prefix: str) -> list[int]:
    return [
        i
        for i in range(model.ngeom)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith(prefix)
    ]


class TestBodyworkIsVisualOnly:
    def test_there_is_bodywork(self, model):
        assert len(_geom_ids(model, "body_")) >= 10

    def test_no_bodywork_geom_can_collide(self, model):
        """A wing with contact enabled would drag on the road and change the physics."""
        for gid in _geom_ids(model, "body_"):
            assert model.geom_contype[gid] == 0
            assert model.geom_conaffinity[gid] == 0

    def test_bodywork_carries_no_mass(self, model):
        """Mass comes from the explicit <inertial>; the shapes must not add to it."""
        car = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
        assert float(model.body_mass[car]) == pytest.approx(CONFIG.mass_kg)

    def test_collision_box_is_still_the_physics(self, model):
        chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
        assert model.geom_contype[chassis] == CHASSIS_CONTYPE

    def test_collision_box_is_hidden_from_the_camera(self, model):
        """Group 3 is not drawn by default, so the fly sees the bodywork, not the box."""
        chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "chassis")
        assert model.geom_group[chassis] >= 3

    def test_bodywork_is_drawn_by_default(self, model):
        for gid in _geom_ids(model, "body_"):
            assert model.geom_group[gid] <= 2


class TestBodyworkGeometry:
    def test_front_wing_is_ahead_of_the_front_axle(self, model, data):
        wing = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "body_front_wing")
        car = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
        forward = data.xmat[car].reshape(3, 3)[:, 0]
        offset = data.geom_xpos[wing] - data.xpos[car]
        assert float(np.dot(offset, forward)) > CONFIG.wheelbase_m / 2

    def test_rear_wing_is_behind_the_rear_axle(self, model, data):
        wing = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "body_rear_wing")
        car = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "car")
        forward = data.xmat[car].reshape(3, 3)[:, 0]
        offset = data.geom_xpos[wing] - data.xpos[car]
        assert float(np.dot(offset, forward)) < -CONFIG.wheelbase_m / 2

    def test_front_wing_clears_the_ground(self, model, data):
        wing = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "body_front_wing")
        bottom = data.geom_xpos[wing][2] - model.geom_size[wing][2]
        assert 0.03 < bottom < 0.15

    def test_nothing_sticks_out_past_the_wheels_sideways(self, model):
        """Overall width is regulated at 2000 mm and the wheels define it."""
        half_width = (CONFIG.track_width_front_m + CONFIG.wheel_width_front_m) / 2
        for gid in _geom_ids(model, "body_"):
            reach = abs(float(model.geom_pos[gid][1])) + float(model.geom_size[gid][1])
            assert reach <= half_width + 1e-6, mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_GEOM, gid
            )


class TestFlyMount:
    def test_site_exists(self, model):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fly_mount") != -1

    def test_site_is_in_the_cockpit_opening(self, model):
        """Between the back of the nose and the front of the airbox, on top of the tub."""
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fly_mount")
        nose = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "body_nose")
        airbox = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "body_airbox")
        tub = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "body_tub")
        x = float(model.site_pos[site][0])
        nose_back = float(model.geom_pos[nose][0] - model.geom_size[nose][0])
        airbox_front = float(model.geom_pos[airbox][0] + model.geom_size[airbox][0])
        assert airbox_front < x < nose_back
        tub_top = float(model.geom_pos[tub][2] + model.geom_size[tub][2])
        assert float(model.site_pos[site][2]) == pytest.approx(tub_top, abs=0.02)

    def test_site_is_on_the_centreline(self, model):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fly_mount")
        assert float(model.site_pos[site][1]) == 0.0

    def test_site_is_hidden_from_the_head_camera(self, model):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fly_mount")
        assert model.site_group[site] >= 3

    def test_head_camera_sits_above_the_fly_mount(self, model, data):
        """The eye goes where the fly's head is, not floating over the front axle."""
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fly_mount")
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fly_head")
        offset = data.cam_xpos[cam] - data.site_xpos[site]
        assert float(np.linalg.norm(offset[:2])) < 0.05
        assert offset[2] == pytest.approx(CONFIG.camera_height_above_mount_m, abs=1e-6)

    def test_head_camera_is_at_helmet_height(self, model, data):
        cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fly_head")
        assert 0.85 < float(data.cam_xpos[cam][2]) < 1.15


def _try_renderer(model):
    try:
        return mujoco.Renderer(model, 128, 128)
    except Exception as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no offscreen GL context: {exc}")


@pytest.mark.render
class TestWhatTheFlySees:
    def test_bodywork_does_not_dominate_the_frame(self, model, data):
        """A cockpit eye sees the nose and front wheels, as a real onboard camera does.
        That static region has no optic flow, so it is capped. A tall nose or a camera
        set too low blows straight through this."""
        renderer = _try_renderer(model)
        renderer.update_scene(data, camera="fly_head")
        frame = renderer.render()
        red = ((frame[:, :, 0].astype(int) > 120) & (frame[:, :, 1].astype(int) < 80)).mean()
        dark = (frame.max(axis=2) < 60).mean()
        assert red + dark < 0.35, f"{100 * (red + dark):.0f}% of the frame is the car"

    def test_horizon_is_near_the_middle(self, model, data):
        """Level camera: sky above, ground below, split near the centre row."""
        renderer = _try_renderer(model)
        renderer.update_scene(data, camera="fly_head")
        frame = renderer.render().astype(int)
        # Sky is the only blue-dominant thing in the scene.
        sky_rows = (frame[:, :, 2] > frame[:, :, 1] + 20).mean(axis=1) > 0.5
        horizon = int(np.argmin(sky_rows))
        assert 0.35 * frame.shape[0] < horizon < 0.65 * frame.shape[0]
