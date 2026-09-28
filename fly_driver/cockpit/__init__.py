"""The fly as the Assetto Corsa driver (GH-25): pose it, export it, rig it to AC's driver.

Assetto Corsa seats its driver with the car's ``driver_base_pos.knh`` and moves the arms with
``steer.ksanim``, both keyed on the stock driver's bone names (``DRIVER:RIG_Arm_L`` and so
on). Hang the fly's body parts on those same bones and the game animates the fly with no
changes to the car: its front legs turn the wheel.

numpy and mujoco only. The Blender half lives in ``scripts/cockpit/`` because it runs inside
Blender's own Python.
"""

from fly_driver.cockpit.knh import KnhNode, read_knh, world_matrices, write_knh
from fly_driver.cockpit.seated_fly import (
    DRIVER_BONE_FOR_BODY,
    SeatedFly,
    body_parts,
    driver_bone_for,
    pose_seated_fly,
)

__all__ = [
    "DRIVER_BONE_FOR_BODY",
    "KnhNode",
    "SeatedFly",
    "body_parts",
    "driver_bone_for",
    "pose_seated_fly",
    "read_knh",
    "world_matrices",
    "write_knh",
]
