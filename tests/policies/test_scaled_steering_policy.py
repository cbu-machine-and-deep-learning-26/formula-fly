"""The run-time steering gain used to test what chatter costs (GH-104)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from fly_driver.policies.mlp_policy import MlpPolicy  # noqa: E402
from fly_driver.policies.scaled_steering_policy import ScaledSteeringPolicy  # noqa: E402

pytestmark = pytest.mark.usefixtures("cpu_default_device")

FEATURE_DIM = 4


def _policy_with_raw_mean(steer: float, throttle: float = 0.5, brake: float = 0.25) -> MlpPolicy:
    """A policy whose raw mean is exactly these numbers for zero features."""
    policy = MlpPolicy(FEATURE_DIM, hidden_sizes=(), normalize_features=False)
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        for module in policy.modules():
            if isinstance(module, torch.nn.Linear) and module.out_features == 3:
                module.bias.copy_(torch.tensor([steer, throttle, brake]))
    return policy.eval()


def _features() -> np.ndarray:
    return np.zeros(FEATURE_DIM, dtype=np.float32)


def test_the_gain_scales_the_raw_mean_before_the_clip() -> None:
    # A raw 3.0 clips to 1 unscaled. At gain 0.25 it is 0.75, inside the range.
    policy = _policy_with_raw_mean(3.0)
    assert policy.act(_features()).steer == 1.0
    assert ScaledSteeringPolicy(policy, 0.25).act(_features()).steer == pytest.approx(0.75)


def test_a_scaled_steer_past_lock_still_clips() -> None:
    policy = _policy_with_raw_mean(-3.0)
    assert ScaledSteeringPolicy(policy, 0.5).act(_features()).steer == -1.0


def test_a_gain_of_one_matches_the_policy() -> None:
    policy = _policy_with_raw_mean(0.4)
    assert ScaledSteeringPolicy(policy, 1.0).act(_features()) == policy.act(_features())


def test_throttle_and_brake_are_left_alone() -> None:
    control = ScaledSteeringPolicy(_policy_with_raw_mean(0.5), 0.1).act(_features())
    assert control.throttle == pytest.approx(0.5)
    assert control.brake == pytest.approx(0.25)
    assert control.steer == pytest.approx(0.05)


def test_the_pedal_gain_scales_throttle_and_brake_but_not_steering() -> None:
    wrapped = ScaledSteeringPolicy(_policy_with_raw_mean(0.5, 3.0, 2.0), 1.0, pedal_gain=0.25)
    control = wrapped.act(_features())
    assert control.steer == pytest.approx(0.5)
    assert control.throttle == pytest.approx(0.75)
    assert control.brake == pytest.approx(0.5)


def test_a_scaled_pedal_past_its_range_still_clips() -> None:
    wrapped = ScaledSteeringPolicy(_policy_with_raw_mean(0.0, 4.0, -4.0), 1.0, pedal_gain=0.5)
    control = wrapped.act(_features())
    assert control.throttle == 1.0
    assert control.brake == 0.0


def test_a_pedal_gain_that_isnt_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="pedal_gain"):
        ScaledSteeringPolicy(_policy_with_raw_mean(0.0), 1.0, pedal_gain=0.0)


def test_it_reports_the_wrapped_policys_feature_width() -> None:
    assert ScaledSteeringPolicy(_policy_with_raw_mean(0.0), 0.5).feature_dim == FEATURE_DIM


@pytest.mark.parametrize("gain", [0.0, -1.0])
def test_a_gain_that_isnt_positive_is_refused(gain: float) -> None:
    with pytest.raises(ValueError, match="gain"):
        ScaledSteeringPolicy(_policy_with_raw_mean(0.0), gain)


def test_features_of_the_wrong_width_are_refused() -> None:
    wrapped = ScaledSteeringPolicy(_policy_with_raw_mean(0.0), 0.5)
    with pytest.raises(ValueError):
        wrapped.act(np.zeros(FEATURE_DIM + 1, dtype=np.float32))
