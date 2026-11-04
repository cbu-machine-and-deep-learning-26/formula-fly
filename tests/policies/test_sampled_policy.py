"""The noise-on policy used to measure steering chatter (GH-104)."""

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


def _policy(log_std: float) -> MlpPolicy:
    torch.manual_seed(0)
    policy = MlpPolicy(FEATURE_DIM, hidden_sizes=(8,), init_log_std=log_std)
    return policy.eval()


def _features() -> np.ndarray:
    return np.zeros(FEATURE_DIM, dtype=np.float32)


def test_reports_the_wrapped_policys_feature_width() -> None:
    assert SampledPolicy(_policy(0.0)).feature_dim == FEATURE_DIM


def test_the_same_seed_gives_the_same_noise_and_another_seed_does_not() -> None:
    sampled = SampledPolicy(_policy(0.0))

    def drive_ten_frames(seed: int) -> list[ControlVector]:
        sampled.reset(seed=seed)
        return [sampled.act(_features()) for _ in range(10)]

    assert drive_ten_frames(3) == drive_ten_frames(3)
    assert drive_ten_frames(3) != drive_ten_frames(4)


def test_reset_without_a_seed_does_not_reseed() -> None:
    sampled = SampledPolicy(_policy(0.0))
    sampled.reset(seed=3)
    first = sampled.act(_features())
    sampled.reset()
    assert sampled.act(_features()) != first


def test_every_frame_draws_fresh_noise() -> None:
    sampled = SampledPolicy(_policy(0.0))
    sampled.reset(seed=0)
    steers = {sampled.act(_features()).steer for _ in range(20)}
    assert len(steers) > 1


def test_large_noise_is_clipped_into_the_controls_range() -> None:
    sampled = SampledPolicy(_policy(3.0))  # std ~20: nearly every sample is out of range
    sampled.reset(seed=0)
    controls = [sampled.act(_features()) for _ in range(50)]
    assert all(-1.0 <= control.steer <= 1.0 for control in controls)
    assert all(0.0 <= control.throttle <= 1.0 for control in controls)
    assert all(0.0 <= control.brake <= 1.0 for control in controls)
    assert any(abs(control.steer) == 1.0 for control in controls)


def test_near_zero_noise_matches_the_deterministic_policy() -> None:
    policy = _policy(-20.0)
    sampled = SampledPolicy(policy)
    sampled.reset(seed=0)
    calm = sampled.act(_features())
    mean = policy.act(_features())
    assert calm.steer == pytest.approx(mean.steer, abs=1e-5)
    assert calm.throttle == pytest.approx(mean.throttle, abs=1e-5)
    assert calm.brake == pytest.approx(mean.brake, abs=1e-5)


def test_sampling_does_not_change_the_policys_weights() -> None:
    policy = _policy(0.0)
    before = {name: value.clone() for name, value in policy.state_dict().items()}
    sampled = SampledPolicy(policy)
    sampled.reset(seed=0)
    sampled.act(_features())
    after = policy.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_rejects_features_of_the_wrong_width() -> None:
    sampled = SampledPolicy(_policy(0.0))
    with pytest.raises(ValueError):
        sampled.act(np.zeros(FEATURE_DIM + 1, dtype=np.float32))
