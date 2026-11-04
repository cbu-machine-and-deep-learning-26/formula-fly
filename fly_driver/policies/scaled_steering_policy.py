"""A trained policy with its steering gain turned down at run time (GH-104).

The best checkpoint's raw steering mean is typically far outside the car's range (median about
1.7, 95th percentile about 4.5, against a range of 1), so the clip turns a smooth signal into
lock-to-lock steering. Scaling the raw mean before the clip asks whether that gain is what
hurts, without adding the lag a low-pass filter would.

It is a one-sided test, as smoothing is: the policy never trained with the scaled steering.

Imports torch at module scope, like ``mlp_policy``, so take it by its full path.
"""

from __future__ import annotations

import torch

from fly_driver.interface import ControlVector, Features, validate_features
from fly_driver.policies.mlp_policy import MlpPolicy

__all__ = ["ScaledSteeringPolicy"]


class ScaledSteeringPolicy:
    """Implements the ``Policy`` protocol, multiplying the raw steering mean before the clip.

    Args:
        policy: The trained policy. Its weights are not changed.
        gain: Multiplies the raw steering mean. 1 changes nothing.

    Raises:
        ValueError: If ``gain`` is not positive.
    """

    def __init__(self, policy: MlpPolicy, gain: float) -> None:
        if gain <= 0.0:
            raise ValueError(f"gain must be positive, got {gain}")
        self.policy = policy
        self.gain = float(gain)

    @property
    def feature_dim(self) -> int:
        """Length of the vector :meth:`act` expects."""
        return int(self.policy.feature_dim)

    def reset(self, seed: int | None = None) -> None:
        """Start an episode; forwarded to the wrapped policy."""
        self.policy.reset(seed=seed)

    def act(self, features: Features) -> ControlVector:
        """The clipped mean action, with the steering scaled first."""
        validate_features(features, self.feature_dim)
        with torch.no_grad():
            batch = torch.as_tensor(features, device=self.policy.device)[None]
            mean, _ = self.policy(batch)
        steer, throttle, brake = mean[0].tolist()
        return ControlVector.clipped(self.gain * steer, throttle, brake)
