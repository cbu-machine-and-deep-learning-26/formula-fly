"""PPO numerics pinned by hand, per AGENTS.md §11 (GH-17)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from fly_driver.policies.mlp_policy import MlpPolicy  # noqa: E402
from fly_driver.training.config import PPOConfig  # noqa: E402
from fly_driver.training.ppo import DifferentiableEye, PPOLearner, RolloutBuffer  # noqa: E402

DIM = 6


def _filled_buffer(steps: int = 32, seed: int = 0, frame_shape=None) -> RolloutBuffer:
    rng = np.random.default_rng(seed)
    buffer = RolloutBuffer(steps, DIM, frame_shape=frame_shape)
    for step in range(steps):
        buffer.add(
            features=rng.normal(size=DIM).astype(np.float32),
            raw_action=rng.normal(size=3).astype(np.float32),
            log_prob=float(rng.normal()),
            reward=float(rng.normal()),
            episode_end=(step % 10 == 9),
            value=float(rng.normal()),
            frame=None
            if frame_shape is None
            else rng.integers(0, 256, size=frame_shape, dtype=np.uint8),
        )
    return buffer


class TestAdvantages:
    def test_gae_matches_a_hand_computed_case(self):
        """Two episodes: rewards 1,1 | 1 (ends) then 2 (continues), bootstrapped by 0.5."""
        buffer = RolloutBuffer(4, DIM)
        rewards = [1.0, 1.0, 1.0, 2.0]
        values = [0.5, 0.25, 0.75, 0.1]
        ends = [False, False, True, False]
        for reward, value, end in zip(rewards, values, ends, strict=True):
            buffer.add(
                features=np.zeros(DIM, np.float32),
                raw_action=np.zeros(3, np.float32),
                log_prob=0.0,
                reward=reward,
                episode_end=end,
                value=value,
            )
        gamma, lam = 0.9, 0.8
        buffer.compute_advantages(last_value=0.5, gamma=gamma, gae_lambda=lam)

        # Step 3 (last, continuing): delta = 2 + 0.9*0.5 - 0.1 = 2.35; adv = 2.35.
        # Step 2 (ends): delta = 1 - 0.75 = 0.25; adv = 0.25 (no carry across the boundary).
        # Step 1: delta = 1 + 0.9*0.75 - 0.25 = 1.425; adv = 1.425 + 0.72*0.25 = 1.605.
        # Step 0: delta = 1 + 0.9*0.25 - 0.5 = 0.725; adv = 0.725 + 0.72*1.605 = 1.8806.
        np.testing.assert_allclose(buffer.advantages, [1.8806, 1.605, 0.25, 2.35], rtol=1e-6)
        np.testing.assert_allclose(buffer.returns, buffer.advantages + np.array(values), rtol=1e-6)

    def test_a_terminal_last_step_ignores_the_bootstrap(self):
        buffer = RolloutBuffer(1, DIM)
        buffer.add(
            features=np.zeros(DIM, np.float32),
            raw_action=np.zeros(3, np.float32),
            log_prob=0.0,
            reward=3.0,
            episode_end=True,
            value=1.0,
        )
        buffer.compute_advantages(last_value=100.0, gamma=0.99, gae_lambda=0.95)
        assert buffer.advantages[0] == pytest.approx(2.0)

    def test_the_buffer_refuses_to_overfill_or_compute_early(self):
        buffer = _filled_buffer(steps=4)
        with pytest.raises(RuntimeError, match="full"):
            buffer.add(
                features=np.zeros(DIM, np.float32),
                raw_action=np.zeros(3, np.float32),
                log_prob=0.0,
                reward=0.0,
                episode_end=False,
                value=0.0,
            )
        partial = RolloutBuffer(4, DIM)
        with pytest.raises(RuntimeError, match="fill it first"):
            partial.compute_advantages(0.0, gamma=0.99, gae_lambda=0.95)

    def test_a_frame_buffer_needs_frames(self):
        buffer = RolloutBuffer(2, DIM, frame_shape=(4, 4, 3))
        with pytest.raises(ValueError, match="needs one"):
            buffer.add(
                features=np.zeros(DIM, np.float32),
                raw_action=np.zeros(3, np.float32),
                log_prob=0.0,
                reward=0.0,
                episode_end=False,
                value=0.0,
            )


class TestTheUpdate:
    def test_an_update_moves_the_policy_and_reports_finite_statistics(self):
        torch.manual_seed(0)
        config = PPOConfig(rollout_steps=32, minibatches=4, epochs=2)
        policy = MlpPolicy(DIM, hidden_sizes=(8,))
        before = [p.detach().clone() for p in policy.parameters()]
        buffer = _filled_buffer(32)
        buffer.compute_advantages(0.0, gamma=config.gamma, gae_lambda=config.gae_lambda)

        stats = PPOLearner(policy, config).update(buffer, update_index=1, total_updates=4)

        assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters(), strict=True))
        for name in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"):
            assert np.isfinite(getattr(stats, name)), name
        assert stats.learning_rate == pytest.approx(config.learning_rate)
        assert stats.epochs_run == 2

    def test_the_learning_rate_anneals_linearly_to_zero(self):
        config = PPOConfig(rollout_steps=32, minibatches=4, learning_rate=1e-3)
        learner = PPOLearner(MlpPolicy(DIM), config)
        assert learner.learning_rate_for(1, 4) == pytest.approx(1e-3)
        assert learner.learning_rate_for(3, 4) == pytest.approx(0.5e-3)
        assert learner.learning_rate_for(4, 4) == pytest.approx(0.25e-3)
        flat = PPOLearner(
            MlpPolicy(DIM), PPOConfig(rollout_steps=32, minibatches=4, anneal_learning_rate=False)
        )
        assert flat.learning_rate_for(4, 4) == pytest.approx(3e-4)
        with pytest.raises(ValueError, match="out of range"):
            learner.learning_rate_for(5, 4)

    def test_target_kl_stops_the_epochs_early(self):
        torch.manual_seed(0)
        config = PPOConfig(
            rollout_steps=32, minibatches=1, epochs=10, target_kl=1e-12, learning_rate=0.1
        )
        buffer = _filled_buffer(32)
        buffer.compute_advantages(0.0, gamma=config.gamma, gae_lambda=config.gae_lambda)
        stats = PPOLearner(MlpPolicy(DIM, hidden_sizes=(8,)), config).update(
            buffer, update_index=1, total_updates=1
        )
        assert stats.epochs_run < 10

    def test_a_non_finite_loss_is_an_error_not_a_step(self):
        torch.manual_seed(0)
        config = PPOConfig(rollout_steps=32, minibatches=4, epochs=1)
        policy = MlpPolicy(DIM, hidden_sizes=(8,))
        before = [p.detach().clone() for p in policy.parameters()]
        buffer = _filled_buffer(32)
        buffer.compute_advantages(0.0, gamma=config.gamma, gae_lambda=config.gae_lambda)
        buffer.returns[:] = np.inf
        with pytest.raises(RuntimeError, match="non-finite PPO loss"):
            PPOLearner(policy, config).update(buffer, update_index=1, total_updates=1)
        assert all(torch.equal(a, b) for a, b in zip(before, policy.parameters(), strict=True))

    def test_a_buffer_of_the_wrong_size_is_refused(self):
        config = PPOConfig(rollout_steps=64, minibatches=4)
        buffer = _filled_buffer(32)
        buffer.compute_advantages(0.0, gamma=0.99, gae_lambda=0.95)
        with pytest.raises(ValueError, match="expects 64"):
            PPOLearner(MlpPolicy(DIM), config).update(buffer, update_index=1, total_updates=1)


class LinearEye(nn.Module):
    """A feed-forward eye with parameters: frames → features through one linear layer."""

    def __init__(self, frame_shape=(4, 4, 3), feature_dim=DIM):
        super().__init__()
        self._frame_shape = frame_shape
        self._feature_dim = feature_dim
        self.linear = nn.Linear(int(np.prod(frame_shape)), feature_dim)

    @property
    def frame_shape(self):
        return self._frame_shape

    @property
    def feature_dim(self):
        return self._feature_dim

    def reset(self):
        pass

    def encode_batch(self, frames: torch.Tensor) -> torch.Tensor:
        flat = frames.reshape(frames.shape[0], -1).to(torch.float32) / 255.0
        return self.linear(flat)

    def encode(self, frame):
        with torch.no_grad():
            return self.encode_batch(torch.as_tensor(frame)[None])[0].numpy().astype(np.float32)


class TestFineTuningAnEye:
    def test_a_differentiable_eye_is_recognised_and_trained(self):
        torch.manual_seed(0)
        eye = LinearEye()
        assert isinstance(eye, DifferentiableEye)
        config = PPOConfig(rollout_steps=32, minibatches=4, epochs=1)
        buffer = _filled_buffer(32, frame_shape=(4, 4, 3))
        buffer.compute_advantages(0.0, gamma=config.gamma, gae_lambda=config.gae_lambda)
        before = eye.linear.weight.detach().clone()

        PPOLearner(MlpPolicy(DIM, hidden_sizes=(8,)), config, eye=eye).update(
            buffer, update_index=1, total_updates=1
        )
        assert not torch.equal(before, eye.linear.weight)

    def test_an_eye_without_trainable_parameters_is_refused(self):
        eye = LinearEye()
        eye.requires_grad_(False)
        with pytest.raises(ValueError, match="no trainable parameters"):
            PPOLearner(MlpPolicy(DIM), PPOConfig(rollout_steps=32, minibatches=4), eye=eye)

    def test_fine_tuning_needs_frames_in_the_buffer(self):
        config = PPOConfig(rollout_steps=32, minibatches=4, epochs=1)
        buffer = _filled_buffer(32)
        buffer.compute_advantages(0.0, gamma=config.gamma, gae_lambda=config.gae_lambda)
        learner = PPOLearner(MlpPolicy(DIM), config, eye=LinearEye())
        with pytest.raises(ValueError, match="stores frames"):
            learner.update(buffer, update_index=1, total_updates=1)
