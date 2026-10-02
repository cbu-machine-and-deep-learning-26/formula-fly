"""Pose flybody's fly as a seated Formula 1 driver and cut it into bone-sized parts (GH-25).

The fly is the model that produces the controls (flybody, Vaxenburg et al. 2025). Its head
goes on the stock driver's head and its body axis (abdomen tip to head) along the driver's
hips-to-head line, leaned 30 degrees further back and 1.3 times as long: an F1 driver lies
almost flat, and that is what lets the front legs reach the wheel while the abdomen and
folded wings stay inside the tub. Measured in the SF70H: the whole fly sits between 7 cm and
78 cm off the road, inside the bodywork, with the front claws about 5 cm from the knuckles.
It is then posed by inverse kinematics against the stock driver's joints, read from the
car's ``driver_base_pos.knh`` (:data:`IK_TARGETS` says how hard each one pulls):

- front legs (T1) are arms: claws on the knuckles, knees drawn towards the elbows, so the
  front legs hold the wheel;
- middle legs (T2) reach towards the driver's knees;
- hind legs (T3) tuck towards the hip joints.

Every fly body is then assigned to the driver bone it should ride on
(:func:`driver_bone_for`). Assetto Corsa moves those bones with the car's own steering and
shift animations, so the parts follow rigidly -- the same way the stock helmet rides on the
head bone -- and nothing in the car's data has to change.

Two coordinate frames meet here. MuJoCo is z-up with the fly facing +x. Assetto Corsa is
y-up with +z to the front of the car and +x to the driver's left (see
:mod:`fly_driver.cockpit.knh`). :func:`mujoco_from_ac` and :func:`ac_from_mujoco` convert
between the two; both are rotations, so lengths are kept.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import numpy.typing as npt

__all__ = [
    "DRIVER_BONE_FOR_BODY",
    "HEAD_OBJECT",
    "IK_TARGETS",
    "BodyPart",
    "MeshPiece",
    "SeatedFly",
    "ac_from_mujoco",
    "body_parts",
    "driver_bone_for",
    "mujoco_from_ac",
    "pose_seated_fly",
]

Vector = npt.NDArray[np.float64]

#: The fly's head, mouthparts and antennae become one object with the stock helmet's name,
#: on the stock head bone. Cars hide ``DRIVER:HELMET`` in the cockpit camera
#: (``[HIDE_OBJECT_0]`` in ``driver3d.ini``), which keeps the camera out of the fly's head.
HEAD_OBJECT = "DRIVER:HELMET"

_HEAD_BODIES = ("head", "rostrum", "haustellum", "labrum_left", "labrum_right")
_HEAD_PREFIXES = ("antenna_",)
_LEG = re.compile(r"^(coxa|femur|tibia|tarsus\d?|claw)_(T[123])_(left|right)$")

#: Leg segment -> driver bone stem, per leg pair. Front legs are arms, the rest are legs.
_LEG_BONES = {
    "T1": {"coxa": "RIG_Arm", "femur": "RIG_Arm", "tibia": "RIG_ForeArm", "distal": "RIG_HAND"},
    "T2": {"coxa": "RIG_Leg", "femur": "RIG_Leg", "tibia": "RIG_Shin", "distal": "RIG_Foot"},
    "T3": {"coxa": "RIG_Leg", "femur": "RIG_Leg", "tibia": "RIG_Shin", "distal": "RIG_Foot"},
}

#: Explicit assignments for the bodies that are not legs.
DRIVER_BONE_FOR_BODY: dict[str, str] = {
    "thorax": "DRIVER:RIG_Cest",
    "wing_left": "DRIVER:RIG_Cest",
    "wing_right": "DRIVER:RIG_Cest",
    "haltere_left": "DRIVER:RIG_Cest",
    "haltere_right": "DRIVER:RIG_Cest",
}

#: ``(fly element, kind, driver bone, weight)``. ``kind`` is ``site`` or ``body`` (the body's
#: origin, which is the joint it hangs from).
#:
#: The claws go on the knuckles, where the stock hands hold the rim, and they are the only
#: hard target: the wheel is what the cameras see. The fly's tarsus is too long and too stiff
#: for its base to reach the wrist as well, and pinning claws alone lets the solver fold the
#: leg into a zigzag (knee by the shoulder), so a light pull puts the knee towards the elbow.
#: Measured on the SF70H rig: claws within ~5 cm of the knuckles, knees ~16 cm from the
#: elbows. Stronger knee pulls make the solve diverge. The legs only have to point into the
#: footwell, which no camera shows.
_KNUCKLES = {"L": "DRIVER:HAND_Middle1", "R": "DRIVER:HAND_Middle4"}

IK_TARGETS: tuple[tuple[str, str, str, float], ...] = tuple(
    item
    for side, s in (("left", "L"), ("right", "R"))
    for item in (
        (f"claw_T1_{side}", "site", _KNUCKLES[s], 1.0),
        (f"tibia_T1_{side}", "body", f"DRIVER:RIG_ForeArm_{s}", 0.1),
        (f"claw_T2_{side}", "site", f"DRIVER:RIG_Shin_{s}", 0.2),
        (f"claw_T3_{side}", "site", f"DRIVER:RIG_Leg_{s}", 0.2),
    )
)

# MuJoCo (x front, y left, z up) <- Assetto Corsa (x left, y up, z front); a pure rotation.
_MJ_FROM_AC = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def mujoco_from_ac(point: npt.ArrayLike) -> Vector:
    """An Assetto Corsa point (x left, y up, z front) in MuJoCo's axes (x front, y left, z up)."""
    return _MJ_FROM_AC @ np.asarray(point, dtype=np.float64)


def ac_from_mujoco(points: npt.ArrayLike) -> Vector:
    """MuJoCo points ``(..., 3)`` in Assetto Corsa's axes; the inverse of :func:`mujoco_from_ac`."""
    return np.asarray(points, dtype=np.float64) @ _MJ_FROM_AC


def driver_bone_for(body: str) -> str | None:
    """The stock driver bone a fly body rides on, or ``None`` for the head group.

    The head group (head, mouthparts, antennae) is one object, :data:`HEAD_OBJECT`, on
    ``DRIVER:RIG_Head``; :func:`body_parts` handles it.

    Raises:
        KeyError: For a body this mapping does not know, so a new flybody release cannot
            silently leave a part floating at the origin.
    """
    if body in _HEAD_BODIES or body.startswith(_HEAD_PREFIXES):
        return None
    if body in DRIVER_BONE_FOR_BODY:
        return DRIVER_BONE_FOR_BODY[body]
    if body.startswith("abdomen"):
        return "DRIVER:RIG_Hips"
    match = _LEG.match(body)
    if match is None:
        raise KeyError(f"no driver bone for fly body {body!r}")
    segment, pair, side = match.groups()
    stem = _LEG_BONES[pair]["distal" if segment.startswith(("tarsus", "claw")) else segment]
    return f"DRIVER:{stem}_{'L' if side == 'left' else 'R'}"


@dataclass(frozen=True)
class MeshPiece:
    """One mesh geom of a body, in Assetto Corsa coordinates (metres).

    Args:
        material: flybody's material name, e.g. ``body`` or ``red``.
        rgba: Its colour.
        vertices: ``(n, 3)`` float32 positions.
        faces: ``(m, 3)`` int32 vertex indices, counter-clockwise from outside.
    """

    material: str
    rgba: tuple[float, float, float, float]
    vertices: npt.NDArray[np.float32]
    faces: npt.NDArray[np.int32]


@dataclass(frozen=True)
class BodyPart:
    """What becomes one object in the FBX: a name, the bone it rides on, and its meshes."""

    name: str
    bone: str
    bodies: tuple[str, ...]
    pieces: tuple[MeshPiece, ...]


@dataclass
class SeatedFly:
    """A posed fly.

    Args:
        model: The flybody model.
        data: Its state in the seated pose (``mj_forward`` has been run).
        scale: Metres of car per MuJoCo length unit.
        target_misses: Metres between each IK target and where its fly element ended up.
    """

    model: mujoco.MjModel
    data: mujoco.MjData
    scale: float
    target_misses: dict[str, float]


def pose_seated_fly(
    fruitfly_xml: str | os.PathLike[str],
    driver_rig: Mapping[str, npt.ArrayLike],
    *,
    iterations: int = 800,
    size: float = 1.3,
    recline_deg: float = 30.0,
) -> SeatedFly:
    """Scale, place and pose the fly on the stock driver's skeleton.

    Args:
        fruitfly_xml: flybody's ``fruitfly/assets/fruitfly.xml``.
        driver_rig: World matrices of the driver's nodes, from
            :func:`~fly_driver.cockpit.knh.world_matrices` (translation in row 3).
        iterations: Damped least-squares steps. Joint limits are enforced after each one.
        size: Multiplies the fitted scale. ``1.0`` makes abdomen-tip-to-head equal the
            driver's hips-to-head; larger reaches the wheel more easily but sits lower.
        recline_deg: Leans the fly further back than the driver's hips-to-head line, about
            its head, so a larger fly's abdomen goes back under the seat instead of down.

    Raises:
        KeyError: If the rig lacks a bone the pose needs.
    """
    model = mujoco.MjModel.from_xml_path(str(Path(fruitfly_xml)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    def joint_position(bone: str) -> Vector:
        return mujoco_from_ac(np.asarray(driver_rig[bone])[3, :3])

    # The fly's body axis -- abdomen tip to head -- goes where the driver's body is: hips to
    # head, reclined along the seat. Matching that length sets the scale. Matching
    # thorax-to-head to chest-to-head instead made a 1.65 m fly whose abdomen and folded
    # wings hung through the floor of the car and whose thorax filled the cockpit camera.
    hips, head = joint_position("DRIVER:RIG_Hips"), joint_position("DRIVER:RIG_Head")
    fly_head = data.xpos[model.body("head").id].copy()
    fly_tail = _abdomen_tip(model, data)
    scale = size * float(np.linalg.norm(head - hips) / np.linalg.norm(fly_head - fly_tail))
    lean = np.radians(recline_deg)  # about the driver's left (+y): head stays, tail swings back
    tilt = np.array(
        [[np.cos(lean), 0.0, -np.sin(lean)], [0.0, 1.0, 0.0], [np.sin(lean), 0.0, np.cos(lean)]]
    )
    rotation = tilt @ _frame(head - hips) @ _frame(fly_head - fly_tail).T
    quaternion = np.zeros(4)
    mujoco.mju_mat2Quat(quaternion, rotation.flatten())
    root = model.jnt_qposadr[model.joint("free").id]
    thorax = data.xpos[model.body("thorax").id].copy()
    # Put the fly's head on the driver's head; the root (thorax) follows from the rotation.
    data.qpos[root : root + 3] = head / scale - rotation @ (fly_head - thorax)
    data.qpos[root + 3 : root + 7] = quaternion

    # Wings folded along the back: their spring reference is flybody's resting pose.
    wing_dofs = set()
    for joint in range(model.njnt):
        if model.joint(joint).name.startswith("wing_"):
            data.qpos[model.jnt_qposadr[joint]] = model.qpos_spring[model.jnt_qposadr[joint]]
            wing_dofs.add(int(model.jnt_dofadr[joint]))
    free_start = int(model.jnt_dofadr[model.joint("free").id])
    fixed = wing_dofs | set(range(free_start, free_start + 6))  # the root stays where it was put
    movable = [dof for dof in range(model.nv) if dof not in fixed]
    targets = [
        (element, kind, joint_position(bone) / scale, weight)
        for element, kind, bone, weight in IK_TARGETS
    ]

    for _ in range(iterations):
        mujoco.mj_forward(model, data)
        rows, errors = [], []
        for element, kind, target, weight in targets:
            jacobian = np.zeros((3, model.nv))
            if kind == "site":
                index = model.site(element).id
                mujoco.mj_jacSite(model, data, jacobian, None, index)
                position = data.site_xpos[index]
            else:
                index = model.body(element).id
                mujoco.mj_jacBody(model, data, jacobian, None, index)
                position = data.xpos[index]
            rows.append(weight * jacobian[:, movable])
            errors.append(weight * (target - position))
        stacked, error = np.vstack(rows), np.concatenate(errors)
        step = stacked.T @ np.linalg.solve(stacked @ stacked.T + 1e-4 * np.eye(error.size), error)
        velocity = np.zeros(model.nv)
        velocity[movable] = step
        mujoco.mj_integratePos(model, data.qpos, velocity, 0.5)
        _clamp_to_joint_limits(model, data)

    mujoco.mj_forward(model, data)
    misses = {}
    for element, kind, target, _ in targets:
        position = (
            data.site_xpos[model.site(element).id]
            if kind == "site"
            else data.xpos[model.body(element).id]
        )
        misses[element] = float(np.linalg.norm(position - target) * scale)
    return SeatedFly(model=model, data=data, scale=scale, target_misses=misses)


def body_parts(seated: SeatedFly) -> list[BodyPart]:
    """The fly as FBX objects: one per body, plus the head group as :data:`HEAD_OBJECT`.

    Vertices are in Assetto Corsa coordinates, in metres. Object names are unique, as the
    game requires (Kunos car pipeline guide, section 1.B).
    """
    model, data = seated.model, seated.data
    pieces_by_body: dict[str, list[MeshPiece]] = {}
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh = model.geom_dataid[geom]
        first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        local = model.mesh_vert[first : first + count].astype(np.float64)
        world = local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        face_first, face_count = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
        faces = model.mesh_face[face_first : face_first + face_count].astype(np.int32)
        material = model.geom_matid[geom]
        name = model.material(material).name if material >= 0 else "default"
        rgba = model.mat_rgba[material] if material >= 0 else model.geom_rgba[geom]
        body = model.body(model.geom_bodyid[geom]).name
        vertices, faces = _weld(ac_from_mujoco(world) * seated.scale, faces)
        pieces_by_body.setdefault(body, []).append(
            MeshPiece(
                material=name,
                rgba=tuple(float(value) for value in rgba),
                vertices=vertices.astype(np.float32),
                faces=faces,
            )
        )

    parts: list[BodyPart] = []
    head_bodies: list[str] = []
    head_pieces: list[MeshPiece] = []
    for body in sorted(pieces_by_body):
        bone = driver_bone_for(body)
        if bone is None:
            head_bodies.append(body)
            head_pieces.extend(pieces_by_body[body])
            continue
        parts.append(
            BodyPart(
                name=f"FLY:{body}", bone=bone, bodies=(body,), pieces=tuple(pieces_by_body[body])
            )
        )
    if head_pieces:
        parts.append(
            BodyPart(
                name=HEAD_OBJECT,
                bone="DRIVER:RIG_Head",
                bodies=tuple(head_bodies),
                pieces=tuple(head_pieces),
            )
        )
    return parts


def _frame(axis: Vector) -> npt.NDArray[np.float64]:
    """Right-handed frame whose first column is ``axis`` and second the world's +y (left).

    Built for both the fly (axis abdomen tip -> head, at rest) and the driver (hips -> head),
    so the rotation between them keeps the fly's left on the driver's left and its back
    towards the seat.
    """
    forward = axis / np.linalg.norm(axis)
    left = np.array([0.0, 1.0, 0.0]) - forward * forward[1]
    left /= np.linalg.norm(left)
    return np.column_stack([forward, left, np.cross(forward, left)])


def _abdomen_tip(model: mujoco.MjModel, data: mujoco.MjData) -> Vector:
    """The point of the last abdominal segment's mesh furthest from the head."""
    body = model.body("abdomen_7").id
    head = data.xpos[model.body("head").id]
    points = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == body and model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[geom]
            first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[first : first + count].astype(np.float64)
            points.append(local @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom])
    if not points:
        return data.xpos[body].copy()
    stacked = np.concatenate(points)
    return stacked[np.argmax(np.linalg.norm(stacked - head, axis=1))]


def _weld(
    vertices: npt.NDArray[np.float64], faces: npt.NDArray[np.int32], *, tolerance: float = 1e-6
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Merge coincident vertices and drop the faces that collapse.

    flybody's meshes reach us as a triangle soup -- every face with three vertices of its own
    (thorax: 42,970 faces, 128,910 vertices). Nothing is connected, so Blender's decimator can
    only shred it into loose edges. Welding first gives it a surface to simplify.
    """
    keys = np.round(vertices / tolerance).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    welded = inverse.reshape(-1)[faces]
    keep = (
        (welded[:, 0] != welded[:, 1])
        & (welded[:, 1] != welded[:, 2])
        & (welded[:, 0] != welded[:, 2])
    )
    return vertices[first], welded[keep].astype(np.int32)


def _clamp_to_joint_limits(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for joint in range(model.njnt):
        if model.jnt_limited[joint] and model.jnt_type[joint] in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            address = model.jnt_qposadr[joint]
            low, high = model.jnt_range[joint]
            data.qpos[address] = min(max(data.qpos[address], low), high)
