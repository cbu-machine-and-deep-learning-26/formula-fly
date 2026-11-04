"""A trained policy with its exploration noise left on, for measuring (GH-104).

:meth:`~fly_driver.policies.mlp_policy.MlpPolicy.act` returns the clipped mean, so evaluation
never sees what training does. This wrapper draws a fresh sample every frame and clips it, as
the training loop does, so noise-on and noise-off drives can be compared from the same starts.

Imports torch at module scope, like ``mlp_policy``, so take it by its full path.
"""

from __future__ import annotations

import torch

from fly_driver.interface import ControlVector, Features, validate_features
from fly_driver.policies.mlp_policy import MlpPolicy

__all__ = ["SampledPolicy"]


class SampledPolicy:
    """Implements the ``Policy`` protocol by sampling from an :class:`MlpPolicy`.

    Args:
        policy: The trained policy. Its weights are not changed.
    """

    def __init__(self, policy: MlpPolicy) -> None:
        self.policy = policy

    @property
    def feature_dim(self) -> int:
        """Length of the vector :meth:`act` expects."""
        return int(self.policy.feature_dim)

    def reset(self, seed: int | None = None) -> None:
        """Start an episode. A seed makes the noise repeatable, from torch's global generator.

        Args:
            seed: Seeds torch's global generator. ``None`` leaves it as it is.
        """
        if seed is not None:
            torch.manual_seed(seed)

    def act(self, features: Features) -> ControlVector:
        """A clipped sample for one feature vector, the same way the training loop takes it."""
        validate_features(features, self.feature_dim)
        with torch.no_grad():
            batch = torch.as_tensor(features, device=self.policy.device)[None]
            raw, _, _ = self.policy.sample(batch)
        steer, throttle, brake = raw[0].tolist()
        return ControlVector.clipped(steer, throttle, brake)
