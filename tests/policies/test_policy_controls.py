"""A policy's in-between outputs have to reach the car unchanged (GH-104)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from fly_driver.interface import ControlVector  # noqa: E402
from fly_driver.policies.mlp_policy import MlpPolicy  # noqa: E402
from fly_driver.policies.sampled_policy import SampledPolicy  # noqa: E402

pytestmark = pytest.mark.usefixtures("cpu_default_device")

FEATURE_DIM = 4


def _policy_with_raw_mean(steer: float, throttle: float, brake: float, log_std: float) -> MlpPolicy:
    """A policy whose raw mean is exactly these numbers for zero features."""
    policy = MlpPolicy(FEATURE_DIM, hidden_sizes=(), normalize_features=False, init_log_std=log_std)
    with torch.no_grad():
        for name, parameter in policy.named_parameters():
            if name != "log_std":
                parameter.zero_()
        for module in policy.modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == 3:
                module.bias.copy_(torch.tensor([steer, throttle, brake]))
    return policy.eval()


def _features() -> np.ndarray:
    return np.zeros(FEATURE_DIM, dtype=np.float32)


@pytest.mark.parametrize("level", [0.01, 0.1, 0.25, 0.5, 0.75, 0.99])
def test_the_evaluation_path_passes_in_between_values_through(level: float) -> None:
    control = _policy_with_raw_mean(level, level, level, 0.0).act(_features())
    assert control.steer == pytest.approx(level)
    assert control.throttle == pytest.approx(level)
    assert control.brake == pytest.approx(level)


@pytest.mark.parametrize("level", [0.1, 0.25, 0.5, 0.75])
def test_the_training_path_passes_in_between_values_through(level: float) -> None:
    # Almost no noise, so the sample is the mean: nothing rounds or thresholds it on the way.
    sampled = SampledPolicy(_policy_with_raw_mean(level, level, level, -20.0))
    sampled.reset(seed=0)
    control = sampled.act(_features())
    assert control.steer == pytest.approx(level, abs=1e-4)
    assert control.throttle == pytest.approx(level, abs=1e-4)
    assert control.brake == pytest.approx(level, abs=1e-4)


def test_only_values_past_the_edge_are_clipped() -> None:
    control = _policy_with_raw_mean(1.5, 1.5, -0.5, 0.0).act(_features())
    assert control == ControlVector(steer=1.0, throttle=1.0, brake=0.0)
