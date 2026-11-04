"""Rescale a trained policy's outputs inside its checkpoint (GH-104).

The best checkpoint's raw outputs are far past the car's range, so nearly everything reaches the
car as full lock or a closed or fully open pedal. Multiplying them down at run time kept most of
its distance on the stiff-steering car, which makes that a better place to start training than a
collapsed policy. This writes that scaling into the weights, so training starts from it.

The exploration noise is scaled by the same gain, or a smaller mean would be lost in noise of the
old size and the rollouts wouldn't be the drives that were measured.

Imports torch at module scope, like ``mlp_policy``, so take it by its full path.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import torch

from fly_driver.policies.mlp_policy import MlpPolicy

__all__ = ["rescale_checkpoint", "rescale_outputs"]


def rescale_outputs(
    policy: MlpPolicy, *, steer: float = 1.0, throttle: float = 1.0, brake: float = 1.0
) -> None:
    """Multiply a policy's raw action means and noise by per-channel gains, in place.

    Args:
        policy: The policy to change.
        steer: Gain on steering.
        throttle: Gain on throttle.
        brake: Gain on brake.

    Raises:
        ValueError: If a gain is not positive.
    """
    for name, gain in (("steer", steer), ("throttle", throttle), ("brake", brake)):
        if gain <= 0.0:
            raise ValueError(f"{name} gain must be positive, got {gain}")
    last_layer = policy.actor[-1]
    gains = torch.tensor([steer, throttle, brake], dtype=last_layer.weight.dtype).to(policy.device)
    with torch.no_grad():
        last_layer.weight.mul_(gains[:, None])
        last_layer.bias.mul_(gains)
        policy.log_std.add_(torch.log(gains))


def rescale_checkpoint(
    source: str | os.PathLike[str],
    target: str | os.PathLike[str],
    *,
    steer: float = 1.0,
    throttle: float = 1.0,
    brake: float = 1.0,
) -> Path:
    """Write a rescaled copy of a ``policy_*.pt`` checkpoint, with its brain copied beside it.

    Args:
        source: A ``policy_*.pt`` file. Its ``brain_*.pt`` sibling is copied if it exists.
        target: Where to write. Its name must start ``policy_``, as warm starts expect.
        steer: Gain on steering.
        throttle: Gain on throttle.
        brake: Gain on brake.

    Returns:
        The written policy's path.

    Raises:
        ValueError: If ``target`` isn't a ``policy_*`` name or would overwrite ``source``.
    """
    source, target = Path(source), Path(target)
    if not target.name.startswith("policy_"):
        raise ValueError(f"{target} must be named policy_*, or a warm start can't find its brain")
    if target.resolve() == source.resolve():
        raise ValueError(f"{target} would overwrite the checkpoint it was made from")
    policy = MlpPolicy.load(source)
    rescale_outputs(policy, steer=steer, throttle=throttle, brake=brake)
    written = policy.save(
        target,
        extra={
            **getattr(policy, "extra", {}),
            "rescaled_from": str(source),
            "rescaled_by": {"steer": steer, "throttle": throttle, "brake": brake},
        },
    )
    brain = source.with_name(source.name.replace("policy_", "brain_", 1))
    if brain.exists():
        shutil.copyfile(brain, target.with_name(target.name.replace("policy_", "brain_", 1)))
    return written
