"""The MLP readout: a Policy for the harness, a Gaussian for PPO, and a checkpoint (GH-17)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fly_driver.drivers import DirectDriveAgent  # noqa: E402
from fly_driver.eyes import PixelEye  # noqa: E402
from fly_driver.interface import ControlVector, Policy  # noqa: E402
from fly_driver.policies.mlp_policy import MlpPolicy, RunningNormalizer  # noqa: E402

#: flyvis may have made CUDA the default device earlier in this process; see tests/conftest.py.
pytestmark = pytest.mark.usefixtures("cpu_default_device")

DIM = 12


def _features(count: int = 1, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(count, DIM)).astype(np.float32)


class TestThePolicyProtocol:
    def test_it_is_a_policy_and_composes_with_an_eye(self):
        eye = PixelEye(frame_shape=(8, 8, 3), downsample=4)
        policy = MlpPolicy(eye.feature_dim, hidden_sizes=(8,))
        assert isinstance(policy, Policy)
        driver = DirectDriveAgent(eye, policy)
        control = driver.act(np.full((8, 8, 3), 120, dtype=np.uint8))
        assert isinstance(control, ControlVector)

    def test_act_is_deterministic_and_in_range_whatever_the_features(self):
        policy = MlpPolicy(DIM, hidden_sizes=(8,), normalize_features=False)
        with torch.no_grad():
            policy.actor[-1].bias.fill_(50.0)  # push the raw mean far out of range
        wild = (_features()[0] * 1e3).astype(np.float32)
        first, second = policy.act(wild), policy.act(wild)
        assert first == second
        assert first.steer == 1.0 and first.throttle == 1.0 and first.brake == 1.0

    def test_act_validates_its_input(self):
        policy = MlpPolicy(DIM)
        with pytest.raises(ValueError, match="feature_dim"):
            policy.act(np.zeros(DIM + 1, dtype=np.float32))
        with pytest.raises(TypeError, match="float32"):
            policy.act(np.zeros(DIM, dtype=np.float64))

    def test_bad_construction_fails(self):
        with pytest.raises(ValueError, match="feature_dim"):
            MlpPolicy(0)
        with pytest.raises(ValueError, match="activation"):
            MlpPolicy(DIM, activation="gelu")


class TestTheGaussian:
    def test_sample_log_prob_matches_evaluate_actions(self):
        policy = MlpPolicy(DIM, hidden_sizes=(8, 8))
        batch = torch.as_tensor(_features(5))
        raw, log_prob, value = policy.sample(batch)
        evaluated_log_prob, entropy, evaluated_value = policy.evaluate_actions(batch, raw)
        assert raw.shape == (5, 3) and log_prob.shape == (5,) and value.shape == (5,)
        torch.testing.assert_close(evaluated_log_prob, log_prob)
        torch.testing.assert_close(evaluated_value, value)
        assert torch.all(entropy > 0)

    def test_sampling_follows_the_global_seed(self):
        policy = MlpPolicy(DIM)
        batch = torch.as_tensor(_features(3))
        torch.manual_seed(1)
        first, *_ = policy.sample(batch)
        torch.manual_seed(1)
        second, *_ = policy.sample(batch)
        torch.manual_seed(2)
        third, *_ = policy.sample(batch)
        torch.testing.assert_close(first, second)
        assert not torch.allclose(first, third)

    def test_the_initial_mean_is_near_zero_with_the_configured_spread(self):
        """CleanRL's initialisation: small last layer, log_std as configured."""
        policy = MlpPolicy(DIM, init_log_std=-0.5, normalize_features=False)
        mean, value = policy(torch.as_tensor(_features(64)))
        assert mean.abs().max() < 0.2
        assert value.shape == (64,)
        torch.testing.assert_close(policy.log_std.exp(), torch.full((3,), float(np.exp(-0.5))))


class TestTheNormalizer:
    def test_running_statistics_match_numpy_over_several_batches(self):
        normalizer = RunningNormalizer(DIM, epsilon=0.0)
        batches = [_features(50, seed=i) * (i + 1) + i for i in range(3)]
        for batch in batches:
            normalizer.update(torch.as_tensor(batch))
        stacked = np.concatenate(batches)
        np.testing.assert_allclose(
            normalizer.mean.cpu().numpy(), stacked.mean(0), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(
            normalizer.var.cpu().numpy(), stacked.var(0), rtol=1e-3, atol=1e-3
        )
        assert float(normalizer.count) == 150.0

    def test_normalised_output_is_standardised_and_clipped(self):
        normalizer = RunningNormalizer(DIM, epsilon=0.0, clip=3.0)
        batch = torch.as_tensor(_features(200, seed=4) * 5 + 2)
        normalizer.update(batch)
        out = normalizer(batch)
        torch.testing.assert_close(out.mean(0), torch.zeros(DIM), atol=0.05, rtol=0)
        assert out.abs().max() <= 3.0

    def test_update_rejects_the_wrong_width(self):
        with pytest.raises(ValueError, match="normalizer expects"):
            RunningNormalizer(DIM).update(torch.zeros(4, DIM + 1))

    def test_the_policy_is_unchanged_by_a_batch_of_the_same_values_until_updated(self):
        policy = MlpPolicy(DIM, hidden_sizes=(8,))
        batch = torch.as_tensor(_features(20, seed=9) * 10)
        before, _ = policy(batch)
        policy.update_normalizer(batch)
        after, _ = policy(batch)
        assert not torch.allclose(before, after)


class TestCheckpoints:
    def test_save_and_load_give_the_same_policy(self, tmp_path: Path):
        policy = MlpPolicy(DIM, hidden_sizes=(16, 8), activation="relu", init_log_std=-1.0)
        policy.update_normalizer(torch.as_tensor(_features(30, seed=3) * 4 + 1))
        path = policy.save(tmp_path / "p.pt", extra={"env_steps": 123, "seed": 0})

        loaded = MlpPolicy.load(path)
        assert loaded.extra == {"env_steps": 123, "seed": 0}
        assert loaded.hidden_sizes == (16, 8) and loaded.activation == "relu"
        features = _features(1, seed=5)[0]
        assert loaded.act(features) == policy.act(features)
        batch = torch.as_tensor(_features(7, seed=6))
        torch.testing.assert_close(loaded(batch)[1], policy(batch)[1])
        torch.testing.assert_close(loaded.normalizer.mean, policy.normalizer.mean)

    def test_loading_something_else_is_refused(self, tmp_path: Path):
        path = tmp_path / "junk.pt"
        torch.save({"weights": torch.zeros(2)}, path)
        with pytest.raises(ValueError, match="not an MlpPolicy checkpoint"):
            MlpPolicy.load(path)
