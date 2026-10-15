"""The training config: a condition is typed fields, seeds are first-class, typos fail (GH-17)."""

from __future__ import annotations

from pathlib import Path

import pytest

from fly_driver.training import (
    DEFAULT_SEEDS,
    KNOWN_BRAINS,
    KNOWN_ENV_TYPES,
    KNOWN_EYE_TYPES,
    BrainConfig,
    EnvConfig,
    EyeConfig,
    PPOConfig,
    TrainConfig,
)

CONFIGS = Path(__file__).parents[2] / "fly_driver" / "configs"
DEFAULT_YAML = CONFIGS / "train_default.yaml"
CX_YAML = CONFIGS / "train_flyvis_cx.yaml"


class TestTheDefaultCondition:
    def test_the_checked_in_config_is_the_locked_in_default(self):
        """Frozen flyvis eye, no brain, practice track, three seeds (AGENTS.md §6)."""
        pytest.importorskip("yaml")
        config = TrainConfig.from_yaml(DEFAULT_YAML)
        assert config.eye.type == "flyvis" and config.eye.frozen is True
        assert config.brain == BrainConfig(type="none")
        assert config.env.type == "practice_track"
        assert config.seeds == (0, 1, 2) == DEFAULT_SEEDS
        assert config.logging.wandb.enabled is False

    def test_the_brain_condition_is_the_default_with_a_brain_and_no_racing_line(self):
        pytest.importorskip("yaml")
        default = TrainConfig.from_yaml(DEFAULT_YAML)
        config = TrainConfig.from_yaml(CX_YAML)
        assert config.brain.type == "central_complex"
        assert config.brain.params["neuron_count"] == 4096
        assert config.env.params["racing_line"] is False
        assert config.eye == default.eye and config.ppo == default.ppo
        assert TrainConfig.from_dict(config.to_dict()) == config

    def test_three_seeds_are_the_default_field_not_a_comment(self):
        assert TrainConfig().seeds == (0, 1, 2)

    def test_round_trips_through_dict_and_yaml(self, tmp_path: Path):
        yaml = pytest.importorskip("yaml")
        config = TrainConfig.from_yaml(DEFAULT_YAML)
        assert TrainConfig.from_dict(config.to_dict()) == config
        written = config.to_yaml(tmp_path / "out.yaml")
        assert TrainConfig.from_yaml(written) == config
        # Plain YAML sequences, never !!python/tuple.
        assert "python/tuple" not in written.read_text(encoding="utf-8")
        assert yaml.safe_load(written.read_text(encoding="utf-8"))["seeds"] == [0, 1, 2]


class TestTheConditionFields:
    def test_eye_brain_and_frozen_are_fields(self):
        config = TrainConfig.from_dict(
            {
                "eye": {"type": "pixels", "frozen": True, "params": {"downsample": 8}},
                "brain": "none",
            }
        )
        assert config.eye == EyeConfig(type="pixels", frozen=True, params={"downsample": 8})
        assert config.brain == BrainConfig(type="none")
        brain = {"type": "central_complex", "params": {"neuron_count": 64}}
        assert TrainConfig.from_dict({"brain": brain}).brain == BrainConfig(**brain)
        assert TrainConfig.from_dict({"eye": {"frozen": False}}).eye.frozen is False

    @pytest.mark.parametrize(
        "data, match",
        [
            ({"eye": {"type": "retina"}}, "unknown eye type"),
            ({"brain": "cortex"}, "unknown brain"),
            ({"brain": {"type": "none", "params": {"neuron_count": 8}}}, "no params"),
            ({"brain": {"type": "central_complex", "params": {"input_dim": 8}}}, "input_dim"),
            ({"brain": {"type": "central_complex", "size": 8}}, "unknown"),
            ({"env": {"type": "carracing"}}, "unknown env type"),
            ({"eye": {"frozen": "yes"}}, "frozen"),
            ({"eye": {"params": {"frame_shape": [64, 64, 3]}}}, "frame_shape"),
            ({"env": {"params": {"max_steps": 3}}}, "env.max_steps"),
        ],
    )
    def test_names_the_loop_cannot_build_are_rejected(self, data, match):
        with pytest.raises(ValueError, match=match):
            TrainConfig.from_dict(data)

    def test_the_known_names_are_what_the_default_yaml_documents(self):
        assert KNOWN_EYE_TYPES == ("flyvis", "cnn", "random_projection", "shuffled", "pixels")
        assert KNOWN_BRAINS == ("none", "central_complex")
        assert set(KNOWN_ENV_TYPES) == {"practice_track", "dummy"}


class TestSeeds:
    @pytest.mark.parametrize(
        "seeds, match",
        [
            ([], "at least one seed"),
            ([0, 0, 1], "distinct"),
            ([-1], "at least 0"),
            ([0.5], "integer"),
            ([True], "integer"),
        ],
    )
    def test_bad_seed_lists_fail(self, seeds, match):
        with pytest.raises(ValueError, match=match):
            TrainConfig(seeds=seeds)

    def test_with_seeds_narrows_a_condition_to_a_subset(self):
        config = TrainConfig()
        assert config.with_seeds([1]).seeds == (1,)
        assert config.with_seeds([1]).eye == config.eye
        assert config.run_dir(2) == Path("runs") / "flyvis_frozen" / "seed_2"
        assert config.run_root == Path("runs") / "flyvis_frozen"

    def test_a_single_seed_is_accepted_as_one(self):
        assert TrainConfig(seeds=7).seeds == (7,)  # type: ignore[arg-type]


class TestUnknownKeysAreErrors:
    @pytest.mark.parametrize(
        "data, path",
        [
            ({"seed": 0}, "train config"),
            ({"ppo": {"learning_rat": 1e-3}}, "train config.ppo"),
            ({"logging": {"wandb": {"enable": True}}}, "train config.logging.wandb"),
            ({"eye": {"kind": "flyvis"}}, "train config.eye"),
        ],
    )
    def test_a_typo_at_any_level_names_where_it_is(self, data, path):
        with pytest.raises(ValueError, match=f"unknown {path} keys"):
            TrainConfig.from_dict(data)

    def test_a_non_mapping_where_a_section_belongs_is_an_error(self):
        with pytest.raises(TypeError, match="train config.ppo must be a mapping"):
            TrainConfig.from_dict({"ppo": 3})


class TestPPOValidation:
    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"rollout_steps": 100, "minibatches": 3}, "must divide"),
            ({"gamma": 1.5}, "gamma"),
            ({"gae_lambda": -0.1}, "gae_lambda"),
            ({"learning_rate": 0.0}, "learning_rate"),
            ({"clip_coef": 0.0}, "clip_coef"),
            ({"entropy_coef": -1.0}, "entropy_coef"),
            ({"max_grad_norm": 0.0}, "max_grad_norm"),
            ({"target_kl": 0.0}, "target_kl"),
            ({"epochs": 0}, "epochs"),
            ({"learning_rate": float("nan")}, "finite"),
        ],
    )
    def test_out_of_range_hyperparameters_fail(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            PPOConfig(**kwargs)

    def test_minibatch_size_and_update_count_follow(self):
        config = TrainConfig(
            total_env_steps=10_000, ppo=PPOConfig(rollout_steps=2048, minibatches=32)
        )
        assert config.ppo.minibatch_size == 64
        assert config.num_updates == 4

    def test_fewer_steps_than_one_rollout_is_refused(self):
        with pytest.raises(ValueError, match="less than one rollout"):
            TrainConfig(total_env_steps=100, ppo=PPOConfig(rollout_steps=2048))


class TestTheOtherSections:
    def test_env_max_steps_lives_at_the_top_of_its_section(self):
        env = EnvConfig(type="dummy", max_steps=50, params={"wind_std": 0.0})
        assert env.max_steps == 50 and env.params == {"wind_std": 0.0}

    @pytest.mark.parametrize(
        "data, match",
        [
            ({"policy": {"activation": "gelu"}}, "activation"),
            ({"policy": {"hidden_sizes": [64, 0]}}, "hidden_sizes"),
            ({"eval": {"every_updates": -1}}, "eval.every_updates"),
            ({"eval": {"episodes": 0}}, "eval.episodes"),
            ({"logging": {"wandb": {"mode": "sometimes"}}}, "mode"),
            ({"logging": {"checkpoint_every_updates": -3}}, "checkpoint_every_updates"),
            ({"name": "a/b"}, "single path component"),
            ({"device": ""}, "device"),
        ],
    )
    def test_bad_values_fail_with_their_key_named(self, data, match):
        with pytest.raises(ValueError, match=match):
            TrainConfig.from_dict(data)

    def test_wandb_is_off_unless_turned_on(self):
        assert TrainConfig().logging.wandb.enabled is False
        on = TrainConfig.from_dict({"logging": {"wandb": {"enabled": True, "tags": ["x"]}}})
        assert on.logging.wandb.enabled is True and on.logging.wandb.tags == ("x",)
