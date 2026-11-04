"""Rescaling a policy's outputs inside its checkpoint (GH-104)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from fly_driver.policies.mlp_policy import MlpPolicy  # noqa: E402
from fly_driver.policies.rescale import rescale_checkpoint, rescale_outputs  # noqa: E402
from fly_driver.training.ppo import output_range_excess  # noqa: E402

pytestmark = pytest.mark.usefixtures("cpu_default_device")

FEATURE_DIM = 6


def _policy() -> MlpPolicy:
    torch.manual_seed(0)
    policy = MlpPolicy(FEATURE_DIM, hidden_sizes=(16, 16), init_log_std=-0.5)
    with torch.no_grad():
        policy.actor[-1].weight.mul_(100.0)  # outputs well past the range, as trained ones are
    return policy.eval()


def _features() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(32, FEATURE_DIM)


def test_each_mean_becomes_exactly_its_gain_times_the_original() -> None:
    policy = _policy()
    before, _ = policy(_features())
    rescale_outputs(policy, steer=0.3, throttle=0.7, brake=0.5)
    after, _ = policy(_features())
    expected = before * torch.tensor([0.3, 0.7, 0.5])
    assert torch.allclose(after, expected, atol=1e-5)


def test_the_noise_is_scaled_by_the_same_gain() -> None:
    policy = _policy()
    before = policy.log_std.exp().detach().clone()
    rescale_outputs(policy, steer=0.3, throttle=0.7, brake=0.5)
    expected = before * torch.tensor([0.3, 0.7, 0.5])
    assert torch.allclose(policy.log_std.exp(), expected, atol=1e-6)


def test_the_critic_and_normaliser_are_left_alone() -> None:
    policy = _policy()
    _, value_before = policy(_features())
    rescale_outputs(policy, steer=0.3, throttle=0.3, brake=0.3)
    _, value_after = policy(_features())
    assert torch.equal(value_before, value_after)


def test_gains_of_one_change_nothing() -> None:
    policy = _policy()
    before, _ = policy(_features())
    rescale_outputs(policy)
    after, _ = policy(_features())
    assert torch.equal(before, after)


def test_a_gain_that_isnt_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="steer gain"):
        rescale_outputs(_policy(), steer=0.0)
    with pytest.raises(ValueError, match="brake gain"):
        rescale_outputs(_policy(), brake=-1.0)


def test_scaling_brings_outputs_into_the_range_of_values_the_car_can_use() -> None:
    policy = _policy()
    before, _ = policy(_features())
    assert output_range_excess(before).mean() > 0.0
    rescale_outputs(policy, steer=1e-3, throttle=1e-3, brake=1e-3)
    after, _ = policy(_features())
    assert output_range_excess(after).abs().max() == 0.0


def test_a_checkpoint_round_trip_keeps_the_scaling_and_copies_the_brain(tmp_path: Path) -> None:
    source = tmp_path / "best" / "policy_best.pt"
    policy = _policy()
    policy.save(source, extra={"note": "kept"})
    (source.parent / "brain_best.pt").write_bytes(b"brain")
    before, _ = policy(_features())

    target = tmp_path / "start" / "policy_best.pt"
    rescale_checkpoint(source, target, steer=0.3, throttle=0.7, brake=0.7)

    loaded = MlpPolicy.load(target).eval()
    after, _ = loaded(_features())
    assert torch.allclose(after, before * torch.tensor([0.3, 0.7, 0.7]), atol=1e-5)
    assert (target.parent / "brain_best.pt").read_bytes() == b"brain"
    assert loaded.extra["note"] == "kept"
    assert loaded.extra["rescaled_by"] == {"steer": 0.3, "throttle": 0.7, "brake": 0.7}


def test_the_source_checkpoint_is_not_changed(tmp_path: Path) -> None:
    source = tmp_path / "policy_best.pt"
    _policy().save(source)
    contents = source.read_bytes()
    rescale_checkpoint(source, tmp_path / "out" / "policy_best.pt", steer=0.3)
    assert source.read_bytes() == contents


def test_a_target_that_would_overwrite_or_lose_the_brain_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "policy_best.pt"
    _policy().save(source)
    with pytest.raises(ValueError, match="overwrite"):
        rescale_checkpoint(source, source, steer=0.3)
    with pytest.raises(ValueError, match="policy_"):
        rescale_checkpoint(source, tmp_path / "scaled.pt", steer=0.3)


@pytest.mark.parametrize(
    ("mean", "expected"),
    [
        ([0.0, 0.0, 0.0], 0.0),
        ([1.0, -1.0, 0.5], 0.0),  # exactly at the edge is fine
        ([2.0, 0.0, 0.0], 1.0),
        ([-3.0, 0.0, 0.0], 4.0),
        ([2.0, -2.0, 3.0], 1.0 + 1.0 + 4.0),
    ],
)
def test_the_range_excess_on_hand_computed_values(mean: list[float], expected: float) -> None:
    excess = output_range_excess(torch.tensor([mean]))
    assert excess.item() == pytest.approx(expected)
    assert np.isfinite(excess.item())
