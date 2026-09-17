"""Deterministic given seed: used for env smoke tests, not for learning."""

from __future__ import annotations

import numpy as np

from fly_driver.interface import ControlVector, FeatureArray, Policy, require_features


class RandomPolicy(Policy):
    def __init__(self, feature_size: int, seed: int = 0) -> None:
        self.feature_size = int(feature_size)
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self._seed)

    def act(self, features: FeatureArray) -> ControlVector:
        require_features(features, self.feature_size)
        steer, throttle, brake = self._rng.uniform([-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        return ControlVector(steer=float(steer), throttle=float(throttle), brake=float(brake))
