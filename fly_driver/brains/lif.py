"""Minimal Euler leaky-integrate-and-fire layer for numerical-stability tests.

This is not the Shiu et al. whole-brain model. It exists so CLAUDE.md §11 LIF
checks can fail loudly before Brian2 or a GPU LIF lands.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from fly_driver.interface import Brain, FeatureArray, require_features


class LeakyIntegrateFireLayer(Brain):
    """Vectorized current-based LIF with reset-to-zero."""

    def __init__(
        self,
        n_neurons: int,
        tau: float = 20.0,
        v_th: float = 1.0,
        dt: float = 1.0,
        input_scale: float = 0.05,
    ) -> None:
        if n_neurons < 1:
            raise ValueError("n_neurons must be positive")
        if tau <= 0 or dt <= 0:
            raise ValueError("tau and dt must be positive")
        self.n_neurons = int(n_neurons)
        self.tau = float(tau)
        self.v_th = float(v_th)
        self.dt = float(dt)
        self.input_scale = float(input_scale)
        self.v = np.zeros(self.n_neurons, dtype=np.float32)
        self.last_spikes = np.zeros(self.n_neurons, dtype=np.float32)

    def reset(self) -> None:
        self.v[:] = 0.0
        self.last_spikes[:] = 0.0

    def step(self, features: FeatureArray) -> npt.NDArray[np.float32]:
        x = require_features(features, self.n_neurons)
        current = self.input_scale * x
        self.v = self.v + self.dt * (-self.v / self.tau + current)
        spikes = (self.v >= self.v_th).astype(np.float32)
        self.v = np.where(spikes > 0, 0.0, self.v).astype(np.float32)
        self.last_spikes = spikes
        if not np.isfinite(self.v).all():
            raise FloatingPointError("LIF voltage non-finite")
        return self.v.copy()
