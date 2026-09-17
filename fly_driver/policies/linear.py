"""Seeded linear readout: features → (steer, throttle, brake)."""

from __future__ import annotations

import numpy as np

from fly_driver.interface import ControlVector, FeatureArray, Policy, require_features


class LinearPolicy(Policy):
    def __init__(self, feature_size: int, seed: int = 0) -> None:
        if feature_size < 1:
            raise ValueError("feature_size must be positive")
        self.feature_size = int(feature_size)
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(self.feature_size)
        self.weight = rng.normal(0.0, scale, size=(self.feature_size, 3)).astype(np.float32)
        self.bias = np.array([0.0, 0.5, 0.0], dtype=np.float32)

    def act(self, features: FeatureArray) -> ControlVector:
        x = require_features(features, self.feature_size)
        raw = x @ self.weight + self.bias
        return ControlVector(
            steer=float(np.tanh(raw[0])),
            throttle=float(1.0 / (1.0 + np.exp(-raw[1]))),
            brake=float(1.0 / (1.0 + np.exp(-raw[2]))),
        )
