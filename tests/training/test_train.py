"""The loop trains, logs per-term rewards, runs every seed, and is reproducible (GH-17).

Everything here runs on the numpy dummy track with the pixel eye, so it needs torch and
nothing else. The one test on the real practice track is marked ``render``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from fly_driver.envs.dummy_track import DummyTrackEnv  # noqa: E402
from fly_driver.policies.mlp_policy import MlpPolicy  # noqa: E402
from fly_driver.training import (  # noqa: E402
    KNOWN_BRAINS,
    KNOWN_ENV_TYPES,
    KNOWN_EYE_TYPES,
    EnvConfig,
    EyeConfig,
    PeriodicEvalConfig,
    PolicyConfig,
    PPOConfig,
    TrainConfig,
)
from fly_driver.training import train as train_module  # noqa: E402
from fly_driver.training.loggers import EPISODES_FILENAME, UPDATES_FILENAME  # noqa: E402
from fly_driver.training.train import (  # noqa: E402
    BRAIN_BUILDERS,
    ENV_BUILDERS,
    EYE_BUILDERS,
    FINAL_CHECKPOINT,
    build_env,
    train,
    train_seed,
)

#: flyvis may have made CUDA the default device earlier in this process; see tests/conftest.py.
pytestmark = pytest.mark.usefixtures("cpu_default_device")

TERM_COLUMNS = ["reward_lap_bonus", "reward_lateral", "reward_off_track", "reward_progress"]
NEEDS_GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA or ROCm GPU")


def _config(tmp_path: Path, **overrides) -> TrainConfig:
    base = TrainConfig(
        name="test",
        output_dir=str(tmp_path / "runs"),
        seeds=(0,),
        total_env_steps=256,
        device="cpu",
        eye=EyeConfig(type="pixels", frozen=True, params={"downsample": 8}),
        policy=PolicyConfig(hidden_sizes=(8,)),
        env=EnvConfig(type="dummy", max_steps=40),
        ppo=PPOConfig(rollout_steps=64, minibatches=4, epochs=2),
    )
    return replace(base, **overrides)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


#: Wall-clock columns of ``updates.csv``; everything else must reproduce exactly.
TIMING_COLUMNS = {"steps_per_second", "elapsed_s"}


def _logged_numbers(run_dir: Path) -> tuple[str, list[dict[str, str]]]:
    """``episodes.csv`` verbatim and ``updates.csv`` without its timing columns."""
    episodes = (run_dir / EPISODES_FILENAME).read_text(encoding="utf-8")
    updates = [
        {key: value for key, value in row.items() if key not in TIMING_COLUMNS}
        for row in _rows(run_dir / UPDATES_FILENAME)
    ]
    return episodes, updates


class TestTheRegistriesMatchTheConfig:
    def test_every_name_a_config_accepts_has_a_builder(self):
        assert set(EYE_BUILDERS) == set(KNOWN_EYE_TYPES)
        assert set(ENV_BUILDERS) == set(KNOWN_ENV_TYPES)
        assert set(BRAIN_BUILDERS) == set(KNOWN_BRAINS)

    def test_the_dummy_env_is_built_from_its_section(self):
        env = build_env(EnvConfig(type="dummy", max_steps=7, params={"wind_std": 0.0}))
        assert isinstance(env, DummyTrackEnv) and env.max_steps == 7 and env.wind_std == 0.0


class TestARunWritesWhatTheIssueAsksFor:
    def test_config_driven_run_writes_per_term_rewards_to_csv(self, tmp_path: Path):
        config = _config(tmp_path)
        result = train_seed(config, 0)

        assert result.run_dir == config.run_dir(0)
        assert result.env_steps == 256 and result.updates == 4
        episodes = _rows(result.run_dir / EPISODES_FILENAME)
        assert result.episodes == len(episodes) >= 4  # 40-step episodes, 256 steps
        assert [key for key in episodes[0] if key.startswith("reward_")] == TERM_COLUMNS
        for row in episodes:
            terms = sum(float(row[column]) for column in TERM_COLUMNS)
            assert terms == pytest.approx(float(row["total_return"]), abs=1e-6)
            assert int(row["steps"]) <= 40 and int(row["seed"]) == 0
        env_steps = [int(row["env_steps"]) for row in episodes]
        assert env_steps == sorted(env_steps) and env_steps[-1] <= 256

        updates = _rows(result.run_dir / UPDATES_FILENAME)
        assert [int(row["update"]) for row in updates] == [1, 2, 3, 4]
        assert [int(row["env_steps"]) for row in updates] == [64, 128, 192, 256]
        assert all(np.isfinite(float(row["policy_loss"])) for row in updates)
        assert float(updates[-1]["learning_rate"]) < float(updates[0]["learning_rate"])

        assert (result.run_dir / "config.yaml").exists()
        assert result.checkpoint == result.run_dir / FINAL_CHECKPOINT
        loaded = MlpPolicy.load(result.checkpoint)
        assert loaded.extra["env_steps"] == 256 and loaded.extra["seed"] == 0
        assert loaded.extra["config"]["eye"]["type"] == "pixels"

    def test_train_runs_every_seed_in_the_config(self, tmp_path: Path):
        config = _config(tmp_path, seeds=(0, 1, 2))
        results = train(config)
        assert [result.seed for result in results] == [0, 1, 2]
        assert (config.run_root / "config.yaml").exists()
        for seed in (0, 1, 2):
            assert (config.run_dir(seed) / EPISODES_FILENAME).exists()
            assert (config.run_dir(seed) / FINAL_CHECKPOINT).exists()

    def test_a_seed_outside_the_config_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not in the config's seeds"):
            train_seed(_config(tmp_path), 5)

    def test_an_existing_run_is_not_overwritten_unless_asked(self, tmp_path: Path):
        config = _config(tmp_path)
        train_seed(config, 0)
        with pytest.raises(FileExistsError, match="already holds a run"):
            train_seed(config, 0)
        train_seed(config, 0, overwrite=True)


class TestSeedsAreReproducible:
    def test_the_same_seed_gives_identical_logs(self, tmp_path: Path):
        """Byte-identical episodes, identical update numbers; only the clock differs."""
        first = train_seed(_config(tmp_path / "a"), 0)
        second = train_seed(_config(tmp_path / "b"), 0)
        assert _logged_numbers(first.run_dir) == _logged_numbers(second.run_dir)

    def test_a_different_seed_gives_different_episodes(self, tmp_path: Path):
        config = _config(tmp_path, seeds=(0, 1))
        zero = train_seed(config, 0)
        one = train_seed(config, 1)
        assert _rows(zero.run_dir / EPISODES_FILENAME) != _rows(one.run_dir / EPISODES_FILENAME)

    def test_periodic_evaluation_does_not_disturb_training(self, tmp_path: Path):
        """The eval protocol reseeds every generator; the loop puts them all back."""
        plain = train_seed(_config(tmp_path / "plain"), 0)
        evaluated = train_seed(
            _config(tmp_path / "eval", eval=PeriodicEvalConfig(every_updates=2, episodes=1)), 0
        )
        assert len(evaluated.eval_reports) == 2
        assert _logged_numbers(plain.run_dir) == _logged_numbers(evaluated.run_dir)


class TestPeriodicEvaluation:
    def test_evaluations_land_under_the_run_with_their_env_steps(self, tmp_path: Path):
        config = _config(
            tmp_path, eval=PeriodicEvalConfig(every_updates=2, episodes=2, max_steps=10)
        )
        result = train_seed(config, 0)
        assert [report.config.env_steps for report in result.eval_reports] == [128, 256]
        for report in result.eval_reports:
            summary = json.loads(
                (
                    result.run_dir / "eval" / f"step_{report.config.env_steps:09d}" / "summary.json"
                ).read_text(encoding="utf-8")
            )
            assert summary["env_steps"] == report.config.env_steps
            assert summary["config"]["seed"] == 0 and summary["config"]["episodes"] == 2
            assert all(row["steps"] <= 10 for row in summary["episodes"])


class TestFrozenVersusFineTuned:
    def test_the_pixel_eye_cannot_be_fine_tuned_and_says_so(self, tmp_path: Path):
        config = _config(
            tmp_path, eye=EyeConfig(type="pixels", frozen=False, params={"downsample": 8})
        )
        with pytest.raises(ValueError, match="cannot be fine-tuned"):
            train_seed(config, 0)

    @pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=NEEDS_GPU)])
    def test_fine_tuning_trains_the_eye_and_frozen_leaves_it_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str
    ):
        """A feed-forward eye with parameters, registered as the pixels builder for the test.

        On the GPU the loop has to move the eye there itself: the builder makes it on the CPU.
        """
        built: list[nn.Module] = []

        class LinearEye(nn.Module):
            def __init__(self, frame_shape):
                super().__init__()
                self._frame_shape = tuple(frame_shape)
                self.linear = nn.Linear(int(np.prod(frame_shape)), 6)
                built.append(self)

            @property
            def frame_shape(self):
                return self._frame_shape

            @property
            def feature_dim(self):
                return 6

            def reset(self):
                pass

            def encode_batch(self, frames):
                flat = frames.reshape(frames.shape[0], -1).to(torch.float32) / 255.0
                return self.linear(flat)

            def encode(self, frame):
                with torch.no_grad():
                    batch = torch.as_tensor(frame, device=self.linear.weight.device)[None]
                    out = self.encode_batch(batch)[0]
                return out.cpu().numpy().astype(np.float32)

        monkeypatch.setitem(
            train_module.EYE_BUILDERS, "pixels", lambda config, env: LinearEye(env.frame_shape)
        )

        frozen = EyeConfig(type="pixels", frozen=True)
        train_seed(_config(tmp_path / "frozen", eye=frozen, device=device), 0)
        frozen_eye = built[-1]
        tuned = EyeConfig(type="pixels", frozen=False)
        result = train_seed(_config(tmp_path / "tuned", eye=tuned, device=device), 0)
        tuned_eye = built[-1]
        # Seed 0 is what the run seeds torch with just before it builds the eye, so this is
        # exactly the eye both runs started from.
        torch.manual_seed(0)
        fresh = LinearEye(tuned_eye.frame_shape)

        assert not any(p.requires_grad for p in frozen_eye.parameters())
        assert tuned_eye.linear.weight.device.type == device
        assert torch.equal(fresh.linear.weight, frozen_eye.linear.weight)
        assert not torch.equal(fresh.linear.weight, tuned_eye.linear.weight.cpu())
        assert (result.run_dir / "eye_final.pt").exists()
        assert not (tmp_path / "frozen" / "runs" / "test" / "seed_0" / "eye_final.pt").exists()


class TestTheEnvContract:
    def test_an_env_without_reward_terms_is_refused(self, tmp_path: Path, monkeypatch):
        class NoTerms(DummyTrackEnv):
            def step(self, action):
                frame, reward, terminated, truncated, info = super().step(action)
                info.pop("reward_terms")
                return frame, reward, terminated, truncated, info

        monkeypatch.setitem(
            train_module.ENV_BUILDERS, "dummy", lambda config: NoTerms(max_steps=config.max_steps)
        )
        with pytest.raises(ValueError, match="reward_terms"):
            train_seed(_config(tmp_path), 0)

    def test_terms_that_do_not_add_up_are_refused(self, tmp_path: Path, monkeypatch):
        class Lying(DummyTrackEnv):
            def step(self, action):
                frame, reward, terminated, truncated, info = super().step(action)
                info["reward_terms"]["progress"] += 1.0
                return frame, reward, terminated, truncated, info

        monkeypatch.setitem(
            train_module.ENV_BUILDERS, "dummy", lambda config: Lying(max_steps=config.max_steps)
        )
        with pytest.raises(ValueError, match="must add up"):
            train_seed(_config(tmp_path), 0)


@NEEDS_GPU
def test_the_loop_trains_on_the_gpu(tmp_path: Path):
    """Same loop, same files, with the policy and the update on the GPU."""
    result = train_seed(_config(tmp_path, device="cuda"), 0)
    updates = _rows(result.run_dir / UPDATES_FILENAME)
    assert [int(row["env_steps"]) for row in updates] == [64, 128, 192, 256]
    assert all(np.isfinite(float(row["policy_loss"])) for row in updates)
    episodes = _rows(result.run_dir / EPISODES_FILENAME)
    for row in episodes:
        terms = sum(float(row[column]) for column in TERM_COLUMNS)
        assert terms == pytest.approx(float(row["total_return"]), abs=1e-6)
    loaded = MlpPolicy.load(result.checkpoint)
    assert loaded.device.type == "cpu" and loaded.extra["env_steps"] == 256


@pytest.mark.slow
def test_the_loop_actually_learns_on_the_dummy_track(tmp_path: Path):
    """Return rises over training: the loop is a learner, not just a logger.

    Lap completion is the later success signal (the issue says so); what this pins is that
    PPO with the pixel eye improves the dummy track's return well beyond the passive
    baseline within a small budget.
    """
    config = _config(
        tmp_path,
        total_env_steps=40_000,
        env=EnvConfig(type="dummy", max_steps=600),
        eye=EyeConfig(type="pixels", frozen=True, params={"downsample": 4}),
        policy=PolicyConfig(hidden_sizes=(64, 64)),
        ppo=PPOConfig(rollout_steps=1024, minibatches=16, epochs=10),
    )
    result = train_seed(config, 0)
    rows = _rows(result.run_dir / EPISODES_FILENAME)
    returns = np.array([float(row["total_return"]) for row in rows])
    early, late = returns[:5].mean(), returns[-5:].mean()
    assert late > early + 5.0, (early, late)
    assert np.isfinite(returns).all()


@pytest.mark.render
def test_the_practice_track_trains_through_the_same_loop(tmp_path: Path):
    """One short rollout on the MuJoCo circuit: frames, terms and lap keys all line up."""
    config = _config(
        tmp_path,
        total_env_steps=64,
        env=EnvConfig(type="practice_track", max_steps=32, params={"racing_line": False}),
        eye=EyeConfig(type="pixels", frozen=True, params={"downsample": 8}),
        ppo=PPOConfig(rollout_steps=64, minibatches=4, epochs=1),
    )
    try:
        result = train_seed(config, 0)
    except Exception as exc:  # pragma: no cover - depends on the machine
        if "gl" in str(exc).lower() or "render" in str(exc).lower():
            pytest.skip(f"no offscreen GL context: {exc}")
        raise
    rows = _rows(result.run_dir / EPISODES_FILENAME)
    assert rows and [key for key in rows[0] if key.startswith("reward_")] == TERM_COLUMNS
    assert result.env_steps == 64
