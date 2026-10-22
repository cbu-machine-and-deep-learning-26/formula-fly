"""Read and write Assetto Corsa ``.knh`` node hierarchies.

A ``.knh`` file is what ksEditor's "Save Driver Base Pos" writes: the driver's node tree with
one local transform per node. The game seats the driver with it (Kunos car pipeline guide,
"Driver position and mesh"). Each node is a length-prefixed name, a 4x4 float32 matrix and a
child count, depth first, little-endian.

**The matrix convention is Direct3D's**: row vectors, translation in the last row, so a
node's world matrix is ``local @ parent_world``. Assetto Corsa's axes are +y up and +z towards
the front of the car, right-handed, which makes +x the driver's left -- the stock rig has
``DRIVER:RIG_Arm_L`` at positive x.

Kunos's own ``.knh`` files are game data and are never committed; the tests write their own.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["KnhNode", "read_knh", "world_matrices", "write_knh"]

Matrix = npt.NDArray[np.float64]


@dataclass
class KnhNode:
    """One node of a ``.knh`` tree.

    Args:
        name: Node name, e.g. ``DRIVER:RIG_Arm_L``.
        local: 4x4 transform relative to the parent, Direct3D layout (translation in row 3).
        children: Child nodes, in file order.
    """

    name: str
    local: Matrix
    children: list[KnhNode] = field(default_factory=list)

    def walk(self) -> list[KnhNode]:
        """This node and every descendant, depth first, in file order."""
        nodes = [self]
        for child in self.children:
            nodes.extend(child.walk())
        return nodes


def read_knh(path: str | os.PathLike[str]) -> KnhNode:
    """Parse a ``.knh`` file.

    Raises:
        ValueError: If the file ends early or has bytes left over after the tree.
    """
    data = Path(path).read_bytes()
    root, end = _read_node(data, 0)
    if end != len(data):
        raise ValueError(f"{path}: {len(data) - end} trailing bytes after the node tree")
    return root


def write_knh(root: KnhNode, path: str | os.PathLike[str]) -> Path:
    """Write ``root`` in the ``.knh`` layout :func:`read_knh` reads."""
    chunks: list[bytes] = []

    def emit(node: KnhNode) -> None:
        encoded = node.name.encode("latin-1")
        matrix = np.asarray(node.local, dtype=np.float32)
        if matrix.shape != (4, 4):
            raise ValueError(f"{node.name}: local transform must be 4x4, got {matrix.shape}")
        chunks.append(struct.pack("<i", len(encoded)) + encoded)
        chunks.append(matrix.astype("<f4").tobytes())
        chunks.append(struct.pack("<i", len(node.children)))
        for child in node.children:
            emit(child)

    emit(root)
    target = Path(path)
    target.write_bytes(b"".join(chunks))
    return target


def world_matrices(root: KnhNode) -> dict[str, Matrix]:
    """World transform of every node, keyed by name.

    Names can repeat in Kunos's files (``DRIVER:HELMET_plastic`` is its own child); the first
    occurrence wins, as it is the one a name lookup finds first.
    """
    worlds: dict[str, Matrix] = {}

    def visit(node: KnhNode, parent: Matrix) -> None:
        world = np.asarray(node.local, dtype=np.float64) @ parent
        worlds.setdefault(node.name, world)
        for child in node.children:
            visit(child, world)

    visit(root, np.eye(4))
    return worlds


def _read_node(data: bytes, offset: int) -> tuple[KnhNode, int]:
    try:
        (length,) = struct.unpack_from("<i", data, offset)
        offset += 4
        if length < 0 or offset + length > len(data):
            raise ValueError(f"bad name length {length} at byte {offset - 4}")
        name = data[offset : offset + length].decode("latin-1")
        offset += length
        local = np.frombuffer(data, dtype="<f4", count=16, offset=offset).reshape(4, 4)
        offset += 64
        (count,) = struct.unpack_from("<i", data, offset)
        offset += 4
    except struct.error as error:
        raise ValueError(f"knh data ends early at byte {offset}") from error
    node = KnhNode(name=name, local=local.astype(np.float64))
    for _ in range(count):
        child, offset = _read_node(data, offset)
        node.children.append(child)
    return node, offset
