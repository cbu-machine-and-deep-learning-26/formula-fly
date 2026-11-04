"""The penalty on raw action means past the car's range (GH-104)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fly_driver.policies.mlp_policy import MlpPolicy  # noqa: E402
from fly_driver.training.config import PPOConfig, TrainConfig  # noqa: E402
from fly_driver.training.ppo import PPOLearner, RolloutBuffer, output_range_excess  # noqa: E402

pytestmark = pytest.mark.usefixtures("cpu_default_device")

DIM = 6
STEPS = 64


def _buffer(seed: int = 0) -> RolloutBuffer:
    rng = np.random.default_rng(seed)
    buffer = RolloutBuffer(STEPS, DIM)
    for step in range(STEPS):
        buffer.add(
            features=rng.normal(size=DIM).astype(np.float32),
            raw_action=rng.normal(size=3).astype(np.float32),
            log_prob=float(rng.normal()),
            reward=float(rng.normal()),
            episode_end=(step % 10 == 9),
            value=float(rng.normal()),
        )
    buffer.compute_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    return buffer


def _policy(output_size: float) -> MlpPolicy:
    """A policy whose raw means have about this size."""
    torch.manual_seed(0)
    policy = MlpPolicy(DIM, hidden_sizes=(16,), normalize_features=False)
    with torch.no_grad():
        policy.actor[-1].weight.mul_(output_size / 0.01)
        policy.actor[-1].bias.zero_()
    return policy


def _update(policy: MlpPolicy, coef: float, *, epochs: int = 2) -> None:
    config = PPOConfig(
        rollout_steps=STEPS,
        minibatches=4,
        epochs=epochs,
        anneal_learning_rate=False,
        output_range_coef=coef,
    )
    torch.manual_seed(1)
    PPOLearner(policy, config).update(_buffer(), update_index=1, total_updates=1)


def _excess(policy: MlpPolicy) -> float:
    torch.manual_seed(2)
    features = torch.randn(256, DIM)
    with torch.no_grad():
        return float(output_range_excess(policy(features)[0]).mean())


def _parameters(policy: MlpPolicy) -> list[torch.Tensor]:
    return [parameter.detach().clone() for parameter in policy.parameters()]


def test_it_is_off_by_default() -> None:
    assert PPOConfig().output_range_coef == 0.0


def test_a_negative_coefficient_is_refused() -> None:
    with pytest.raises(ValueError, match="output_range_coef"):
        PPOConfig(output_range_coef=-0.1)


def test_with_it_off_an_update_is_the_same_as_before() -> None:
    plain, off = _policy(5.0), _policy(5.0)
    config = PPOConfig(rollout_steps=STEPS, minibatches=4, epochs=2, anneal_learning_rate=False)
    torch.manual_seed(1)
    PPOLearner(plain, config).update(_buffer(), update_index=1, total_updates=1)
    _update(off, 0.0)
    for first, second in zip(_parameters(plain), _parameters(off), strict=True):
        assert torch.equal(first, second)


def test_it_changes_nothing_for_a_policy_already_inside_the_range() -> None:
    without, with_penalty = _policy(0.001), _policy(0.001)
    assert _excess(without) == 0.0
    _update(without, 0.0, epochs=1)
    _update(with_penalty, 0.1, epochs=1)
    for first, second in zip(_parameters(without), _parameters(with_penalty), strict=True):
        assert torch.allclose(first, second, atol=1e-6)


def test_it_pulls_outputs_back_towards_the_range() -> None:
    without, with_penalty = _policy(5.0), _policy(5.0)
    start = _excess(without)
    assert start > 1.0
    for _ in range(20):
        _update(without, 0.0)
        _update(with_penalty, 1.0)
    # Direction, not speed: each update here is small, and it fell from about 14.7 to 8.2.
    assert _excess(with_penalty) < 0.75 * start
    assert _excess(with_penalty) < _excess(without)


def test_a_recipe_can_set_it() -> None:
    config = TrainConfig.from_dict({"name": "t", "ppo": {"output_range_coef": 0.1}})
    assert config.ppo.output_range_coef == 0.1
