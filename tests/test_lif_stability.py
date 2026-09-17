"""LIF numerical stability over a long rollout (CLAUDE.md §11)."""

from __future__ import annotations

import numpy as np

from fly_driver.brains import LeakyIntegrateFireLayer


def test_lif_no_nan_over_long_rollout() -> None:
    n = 64
    layer = LeakyIntegrateFireLayer(n_neurons=n, tau=10.0, v_th=1.0, dt=1.0, input_scale=0.2)
    rng = np.random.default_rng(0)
    spike_count = 0
    for _ in range(2000):
        x = rng.normal(0.0, 1.0, size=n).astype(np.float32)
        v = layer.step(x)
        assert np.isfinite(v).all()
        assert v.shape == (n,)
        spike_count += int(layer.last_spikes.sum())
    rate = spike_count / (2000 * n)
    assert 0.0 <= rate < 0.5


def test_lif_reset_clears_voltage() -> None:
    layer = LeakyIntegrateFireLayer(n_neurons=8)
    layer.step(np.ones(8, dtype=np.float32) * 10)
    layer.reset()
    np.testing.assert_array_equal(layer.v, np.zeros(8, dtype=np.float32))
