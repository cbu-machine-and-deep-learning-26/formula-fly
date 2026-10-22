"""The fly-to-driver mapping, the axis conversion, welding, and (with flybody) the pose (GH-25).

The pose tests need flybody's ``fruitfly.xml`` (``FLYBODY_FRUITFLY_XML``, or the checkout
``docs/fly-driver.md`` describes) and skip without it. They use a synthetic rig with human
joint positions, not Kunos's file.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np
import pytest

from fly_driver.cockpit import seated_fly
from fly_driver.cockpit.seated_fly import (
    HEAD_OBJECT,
    HINGE_NODES,
    SeatedFly,
    ac_from_mujoco,
    body_parts,
    driver_bone_for,
    hinges,
    level_gaze_pitch_deg,
    mujoco_from_ac,
    pose_seated_fly,
)

FRUITFLY_XML = Path(
    os.environ.get(
        "FLYBODY_FRUITFLY_XML",
        Path.home() / ".cache/flybody-src/flybody/fruitfly/assets/fruitfly.xml",
    )
)
needs_flybody = pytest.mark.skipif(not FRUITFLY_XML.is_file(), reason="flybody's fruitfly.xml")

#: A seated driver's joints in Assetto Corsa axes (x left, y up, z front), metres.
_JOINTS = {
    "DRIVER:RIG_Hips": (0.0, 0.27, 0.14),
    "DRIVER:RIG_Cest": (0.0, 0.38, -0.03),
    "DRIVER:RIG_Head": (0.0, 0.64, -0.13),
    "DRIVER:RIG_ForeArm_L": (0.17, 0.33, 0.27),
    "DRIVER:RIG_ForeArm_R": (-0.17, 0.33, 0.27),
    "DRIVER:HAND_Middle1": (0.16, 0.50, 0.31),
    "DRIVER:HAND_Middle4": (-0.17, 0.49, 0.31),
    "DRIVER:RIG_Shin_L": (0.08, 0.41, 0.65),
    "DRIVER:RIG_Shin_R": (-0.08, 0.41, 0.65),
    "DRIVER:RIG_Leg_L": (0.09, 0.22, 0.31),
    "DRIVER:RIG_Leg_R": (-0.09, 0.22, 0.31),
}


def _rig() -> dict[str, np.ndarray]:
    worlds = {}
    for name, position in _JOINTS.items():
        world = np.eye(4)
        world[3, :3] = position
        worlds[name] = world
    return worlds


class TestTheMapping:
    @pytest.mark.parametrize(
        "body, bone",
        [
            ("thorax", "DRIVER:RIG_Cest"),
            ("wing_left", "DRIVER:RIG_Cest"),
            ("abdomen_4", "DRIVER:RIG_Hips"),
            ("coxa_T1_left", "DRIVER:RIG_Arm_L"),
            ("tibia_T1_right", "DRIVER:RIG_ForeArm_R"),
            ("tarsus3_T1_left", "DRIVER:RIG_HAND_L"),
            ("claw_T1_right", "DRIVER:RIG_HAND_R"),
            ("femur_T2_left", "DRIVER:RIG_Leg_L"),
            ("tibia_T3_right", "DRIVER:RIG_Shin_R"),
            ("claw_T2_left", "DRIVER:RIG_Foot_L"),
        ],
    )
    def test_each_segment_rides_on_the_matching_driver_bone(self, body, bone):
        assert driver_bone_for(body) == bone

    @pytest.mark.parametrize("body", ["head", "rostrum", "antenna_left", "labrum_right"])
    def test_the_head_group_has_no_bone_of_its_own(self, body):
        assert driver_bone_for(body) is None

    def test_an_unknown_body_is_an_error_not_a_floating_part(self):
        with pytest.raises(KeyError, match="no driver bone"):
            driver_bone_for("proboscis_extension")


class TestAxes:
    def test_ac_front_up_left_become_mujoco_x_z_y(self):
        np.testing.assert_allclose(mujoco_from_ac((0, 0, 1)), (1, 0, 0))  # front
        np.testing.assert_allclose(mujoco_from_ac((0, 1, 0)), (0, 0, 1))  # up
        np.testing.assert_allclose(mujoco_from_ac((1, 0, 0)), (0, 1, 0))  # driver's left

    def test_the_conversion_round_trips_and_keeps_lengths(self):
        points = np.random.default_rng(0).normal(size=(10, 3))
        back = ac_from_mujoco(np.stack([mujoco_from_ac(p) for p in points]))
        np.testing.assert_allclose(back, points, atol=1e-12)
        assert np.isclose(np.linalg.det(seated_fly._MJ_FROM_AC), 1.0)  # a rotation


class TestWelding:
    def test_a_triangle_soup_becomes_a_connected_surface(self):
        # Two triangles sharing an edge, each with its own three vertices.
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float
        )
        faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
        welded, new_faces = seated_fly._weld(vertices, faces)
        assert len(welded) == 4 and len(new_faces) == 2
        np.testing.assert_allclose(welded[new_faces[1]], vertices[faces[1]])

    def test_faces_that_collapse_are_dropped(self):
        vertices = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]], float)
        _, faces = seated_fly._weld(vertices, np.array([[0, 1, 2]], np.int32))
        assert len(faces) == 0


@pytest.fixture(scope="module")
def seated():
    return pose_seated_fly(FRUITFLY_XML, _rig())


@needs_flybody
class TestThePose:
    def test_the_front_claws_reach_the_knuckles(self, seated):
        assert seated.target_misses["claw_T1_left"] < 0.08
        assert seated.target_misses["claw_T1_right"] < 0.08

    def test_the_fly_fits_in_the_cockpit(self, seated):
        """Above the road and no taller than a seated driver's helmet.

        The first fit (thorax-to-head = chest-to-head) made a 1.65 m fly whose abdomen and
        wings went a metre under the car; in the game only its head showed.
        """
        heights = np.concatenate(
            [piece.vertices[:, 1] for part in body_parts(seated) for piece in part.pieces]
        )
        assert heights.min() > 0.0
        assert heights.max() < 0.85
        assert 1.5 < seated.scale < 3.5  # metres per MuJoCo unit

    def test_the_head_is_where_the_drivers_head_is(self, seated):
        head = ac_from_mujoco(seated.data.xpos[seated.model.body("head").id]) * seated.scale
        np.testing.assert_allclose(head, _JOINTS["DRIVER:RIG_Head"], atol=1e-6)

    def test_every_body_is_mapped_and_every_mesh_exported(self, seated):
        model = seated.model
        for body in range(1, model.nbody):
            driver_bone_for(model.body(body).name)  # raises if unmapped
        parts = body_parts(seated)
        names = [part.name for part in parts]
        assert len(set(names)) == len(names) and HEAD_OBJECT in names
        exported = sum(len(part.pieces) for part in parts)
        assert exported == int((model.geom_type == 7).sum())  # every mesh geom
        for part in parts:
            for piece in part.pieces:
                assert np.isfinite(piece.vertices).all()
                assert piece.faces.max() < len(piece.vertices)

    def test_the_gaze_reads_the_head_s_elevation(self, seated):
        """At rest the fly faces forward (0 degrees); pitched nose-up 30 degrees, it reads 30."""
        model = seated.model
        rest = mujoco.MjData(model)
        mujoco.mj_forward(model, rest)
        resting = SeatedFly(model=model, data=rest, scale=1.0, target_misses={})
        assert level_gaze_pitch_deg(resting) == pytest.approx(0.0, abs=1e-6)
        root = model.jnt_qposadr[model.joint("free").id]
        nose_up = np.zeros(4)
        mujoco.mju_axisAngle2Quat(nose_up, [0.0, 1.0, 0.0], np.radians(-30.0))  # about the left
        rest.qpos[root + 3 : root + 7] = nose_up
        mujoco.mj_forward(model, rest)
        assert level_gaze_pitch_deg(resting) == pytest.approx(30.0, abs=1e-6)

    def test_the_fly_faces_forward_like_a_driver(self, seated):
        """Level gaze from a neck inside its range, body pointing to the front of the car.

        Laid along the driver's reclined line (the first build) it looked 141 degrees up and
        back, which no neck pose in flybody's range fixes.
        """
        model, data = seated.model, seated.data
        assert level_gaze_pitch_deg(seated) == pytest.approx(0.0, abs=0.5)
        neck = model.joint("head").id
        low, high = model.jnt_range[neck]
        assert low <= data.qpos[model.jnt_qposadr[neck]] <= high
        tail = ac_from_mujoco(seated_fly._abdomen_tip(model, data)) * seated.scale
        head = ac_from_mujoco(data.xpos[model.body("head").id]) * seated.scale
        assert head[2] > tail[2]  # head to the front, abdomen behind

    def test_the_wings_are_folded(self, seated):
        model, data = seated.model, seated.data
        for joint in ("wing_yaw_left", "wing_roll_left", "wing_pitch_left"):
            address = model.jnt_qposadr[model.joint(joint).id]
            assert data.qpos[address] == pytest.approx(model.qpos_spring[address])


@needs_flybody
class TestTheHinges:
    """GH-26: each wing and antenna under its own node at its hinge, rest pose unchanged."""

    def test_each_wing_and_antenna_rides_on_its_own_hinge(self, seated):
        found = {hinge.name: hinge for hinge in hinges(seated)}
        assert set(found) == set(HINGE_NODES)
        assert found["FLY:WING_L"].bone == found["FLY:WING_R"].bone == "DRIVER:RIG_Cest"
        assert found["FLY:ANTENNA_L"].bone == found["FLY:ANTENNA_R"].bone == "DRIVER:RIG_Head"
        parts = {part.name: part for part in body_parts(seated)}
        for node, (body, _, _) in HINGE_NODES.items():
            assert parts[f"FLY:{body}"].bone == node
        head = parts[HEAD_OBJECT].bodies
        assert "head" in head and not any(body.startswith("antenna") for body in head)

    def test_the_hinge_sits_on_the_joint_with_x_along_its_axis(self, seated):
        model, data = seated.model, seated.data
        parts = {part.name: part for part in body_parts(seated)}
        for hinge in hinges(seated):
            body, joint, _ = HINGE_NODES[hinge.name]
            index = model.joint(joint).id
            pivot = ac_from_mujoco(data.xanchor[index]) * seated.scale
            np.testing.assert_allclose(hinge.world[3, :3], pivot, atol=1e-9)
            rotation = hinge.world[:3, :3]
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-9)
            assert np.linalg.det(rotation) == pytest.approx(1.0)
            axis = ac_from_mujoco(data.xaxis[index])
            assert abs(rotation[0] @ axis) == pytest.approx(1.0)
            # The pivot is at the root of its own mesh, not somewhere on the body.
            vertices = np.concatenate([piece.vertices for piece in parts[f"FLY:{body}"].pieces])
            reach = np.linalg.norm(vertices - pivot, axis=1)
            assert reach.min() < 0.1 * reach.max()

    def test_a_positive_wing_angle_spreads_the_wing_out(self, seated):
        parts = {part.name: part for part in body_parts(seated)}
        for hinge in hinges(seated):
            if not hinge.name.startswith("FLY:WING"):
                continue
            body = HINGE_NODES[hinge.name][0]
            vertices = np.concatenate([piece.vertices for piece in parts[f"FLY:{body}"].pieces])
            turned = _turn(vertices, hinge.world, np.radians(10))
            outward = 1.0 if hinge.name.endswith("_L") else -1.0  # AC +x is the driver's left
            assert outward * (turned[:, 0].mean() - vertices[:, 0].mean()) > 0.005

    def test_the_rest_pose_is_unchanged(self, seated):
        """Baking a part into its hinge's frame and back is the identity, so with the
        nodes at rest every vertex is where the GH-25 build put it."""
        parts = {part.name: part for part in body_parts(seated)}
        for hinge in hinges(seated):
            body = HINGE_NODES[hinge.name][0]
            for piece in parts[f"FLY:{body}"].pieces:
                np.testing.assert_allclose(
                    _turn(piece.vertices, hinge.world, 0.0), piece.vertices, atol=1e-6
                )


def _turn(vertices: np.ndarray, world: np.ndarray, angle: float) -> np.ndarray:
    """Vertices baked into ``world``'s frame, turned ``angle`` about its local x, and back."""
    local = (vertices - world[3, :3]) @ world[:3, :3].T
    cos, sin = np.cos(angle), np.sin(angle)
    turn_x = np.array([[1.0, 0.0, 0.0], [0.0, cos, sin], [0.0, -sin, cos]])  # row vectors
    return local @ turn_x @ world[:3, :3] + world[3, :3]
