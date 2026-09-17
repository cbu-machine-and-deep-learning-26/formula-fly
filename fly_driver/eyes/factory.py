"""Construct an Eye from experiment config."""

from __future__ import annotations

from fly_driver.eyes.cnn import SmallCnnEye
from fly_driver.eyes.flyvis_eye import FlyvisEye
from fly_driver.eyes.random_projection import RandomProjectionEye
from fly_driver.eyes.shuffled import ShuffledConnectomeEye
from fly_driver.interface import DEFAULT_FRAME_SHAPE, Eye

_EYE_TYPES = ("flyvis", "cnn", "random_projection", "shuffled")


def build_eye(
    eye_type: str,
    *,
    output_size: int = 256,
    input_shape: tuple[int, int, int] = DEFAULT_FRAME_SHAPE,
    seed: int = 0,
    frozen: bool = True,
) -> Eye:
    kind = eye_type.strip().lower()
    if kind == "random_projection":
        return RandomProjectionEye(output_size=output_size, input_shape=input_shape, seed=seed)
    if kind == "cnn":
        return SmallCnnEye(output_size=output_size, input_shape=input_shape, seed=seed)
    if kind == "shuffled":
        return ShuffledConnectomeEye(output_size=output_size, input_shape=input_shape, seed=seed)
    if kind == "flyvis":
        return FlyvisEye(output_size=output_size, input_shape=input_shape, frozen=frozen)
    raise ValueError(f"Unknown eye_type {eye_type!r}. Expected one of {_EYE_TYPES}.")
