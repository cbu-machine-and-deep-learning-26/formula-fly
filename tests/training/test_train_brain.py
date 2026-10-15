"""Training the central-complex brain with PPO, on the dummy track with the pixel eye (GH-23)."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fly_driver.brains.blank import blank_trainable_parameter_count  # noqa: E402
from fly_driver.brains.central_complex import CentralComplexBrain  # noqa: E402
from fly_driver.brains.constants import VOLTAGE_LIMIT  # noqa: E402
from fly_driver.training import (  # noqa: E402
    BrainConfig,
    EnvConfig,
    EvalConfig,
    EyeConfig,
    PeriodicEvalConfig,
    PolicyConfig,
    PPOConfig,
    TrainConfig,
    evaluate,
)
from fly_driver.training.loggers import EPISODES_FILENAME  # noqa: E402
from fly_driver.training.ppo import PPOLearner  # noqa: E402
from fly_driver.training.train import (  # noqa: E402
    FINAL_CHECKPOINT,
    PARAMETERS_FILENAME,
    build_env,
    load_driver,
    train_seed,
)

pytestmark = pytest.mark.usefixtures("cpu_default_device")

REPO = Path(__file__).parents[2]
NEURON_COUNT = 48
OUTPUT_DIM = 8
EYE_DIM = 64  # the pixel eye at downsample 8 on the dummy track's frames


def _config(output_dir: Path, **overrides) -> TrainConfig:
    base = TrainConfig(
        name="cx",
        output_dir=str(output_dir),
        seeds=(0,),
        total_env_steps=256,
        device="cpu",
        eye=EyeConfig(type="pixels", params={"downsample": 8}),
        brain=BrainConfig(
            type="central_complex",
            params={"neuron_count": NEURON_COUNT, "output_dim": OUTPUT_DIM},
        ),
        policy=PolicyConfig(hidden_sizes=(8,)),
        env=EnvConfig(type="dummy", max_steps=40),
        ppo=PPOConfig(rollout_steps=64, minibatches=4, epochs=2),
        eval=PeriodicEvalConfig(every_updates=2, episodes=1),
    )
    return replace(base, **overrides)


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One short training run, with every PPO update's input and effect captured."""
    original = PPOLearner.update
    updates: list[dict] = []

    def capturing(self, buffer, **kwargs):
        brain = self.brain
        before = {name: value.detach().clone() for name, value in brain.state_dict().items()}
        with torch.no_grad():
            _, replayed = brain.transition(
                torch.as_tensor(buffer.brain_states), torch.as_tensor(buffer.brain_inputs)
            )
        stats = original(self, buffer, **kwargs)
        updates.append(
            {
                "states": buffer.brain_states.copy(),
                "episode_ends": buffer.episode_ends.copy(),
                "features": buffer.features.copy(),
                "replayed": replayed.numpy(),
                "before": before,
                "after": {k: v.detach().clone() for k, v in brain.state_dict().items()},
            }
        )
        return stats

    PPOLearner.update = capturing
    try:
        config = _config(tmp_path_factory.mktemp("runs"))
        result = train_seed(config, 0)
    finally:
        PPOLearner.update = original
    return {"config": config, "result": result, "updates": updates}


class TestTheRollout:
    def test_the_membrane_is_at_rest_at_every_episode_start(self, run):
        for update in run["updates"]:
            states, ends = update["states"], update["episode_ends"]
            starts = np.flatnonzero(ends[:-1]) + 1
            assert starts.size > 0
            assert np.all(states[starts] == 0.0)
            # Mid-episode the membrane is carried, not reset every frame.
            assert np.any(states[~np.isin(np.arange(len(states)), starts)] != 0.0)
        assert np.all(run["updates"][0]["states"][0] == 0.0)

    def test_replaying_the_stored_membrane_reproduces_what_the_policy_read(self, run):
        for update in run["updates"]:
            np.testing.assert_allclose(update["replayed"], update["features"], atol=1e-6)


class TestTheUpdate:
    def test_synapse_strengths_learn_and_the_mask_does_not_move(self, run):
        update = run["updates"][0]
        before, after = update["before"], update["after"]
        for name in ("input_weight", "recurrent_weight", "bias", "log_tau"):
            assert not torch.equal(before[name], after[name]), name
        for name in ("input_pre", "input_post", "recurrent_pre", "recurrent_post"):
            assert torch.equal(before[name], after[name]), name

    def test_the_mask_is_not_a_parameter(self):
        brain = CentralComplexBrain(16, neuron_count=NEURON_COUNT, output_dim=OUTPUT_DIM)
        voltage = torch.randn(4, NEURON_COUNT)
        _, output = brain.transition(voltage, torch.randn(4, 16))
        output.sum().backward()
        assert brain.input_weight.grad is not None and brain.input_weight.grad.abs().sum() > 0
        assert brain.recurrent_weight.grad.abs().sum() > 0
        parameter_names = {name for name, _ in brain.named_parameters()}
        assert not parameter_names & {"input_pre", "input_post", "recurrent_pre", "recurrent_post"}


class TestTheRun:
    def test_parameter_counts_are_logged_for_the_rq2_comparison(self, run):
        run_dir = run["result"].run_dir
        counts = json.loads((run_dir / PARAMETERS_FILENAME).read_text(encoding="utf-8"))
        brain = CentralComplexBrain(EYE_DIM, neuron_count=NEURON_COUNT, output_dim=OUTPUT_DIM)
        assert counts["brain"] == "central_complex"
        assert counts["neuron_count"] == NEURON_COUNT
        assert counts["eye_trainable"] == 0
        assert counts["brain_trainable"] == brain.trainable_parameter_count()
        assert counts["blank_trainable"] == blank_trainable_parameter_count(EYE_DIM, NEURON_COUNT)
        assert counts["head_trainable"] > 0

    def test_periodic_evaluation_does_not_disturb_the_training_rollout(self, run, tmp_path):
        quiet = _config(tmp_path, eval=PeriodicEvalConfig(every_updates=0))
        result = train_seed(quiet, 0)
        assert run["result"].eval_reports
        logged = (run["result"].run_dir / EPISODES_FILENAME).read_text(encoding="utf-8")
        assert (result.run_dir / EPISODES_FILENAME).read_text(encoding="utf-8") == logged

    def test_a_trained_brain_stays_finite_over_a_long_rollout(self, run):
        """AGENTS.md §11: 15 000 frames is the practice track's whole episode cap."""
        brain = CentralComplexBrain(EYE_DIM, neuron_count=NEURON_COUNT, output_dim=OUTPUT_DIM)
        weights = torch.load(run["result"].run_dir / "brain_final.pt", weights_only=True)
        brain.load_state_dict(weights)
        brain.reset()
        rng = np.random.default_rng(0)
        outputs = np.empty((15_000, OUTPUT_DIM), dtype=np.float32)
        for step in range(len(outputs)):
            outputs[step] = brain.step(rng.random(EYE_DIM, dtype=np.float32) * 9.0 - 3.0)
        assert np.all(np.isfinite(outputs))
        assert np.max(np.abs(brain.activity())) <= VOLTAGE_LIMIT
        assert float(outputs[-1000:].std()) > 1e-4

    def test_the_checkpoint_reloads_and_evaluates_deterministically(self, run):
        config, run_dir = run["config"], run["result"].run_dir
        env = build_env(config.env)
        driver = load_driver(config, run_dir / FINAL_CHECKPOINT, env)
        env.close()
        saved = torch.load(run_dir / "brain_final.pt", weights_only=True)
        for name, value in driver.brain.state_dict().items():
            assert torch.equal(value, saved[name]), name
        eval_config = EvalConfig(episodes=2, seed=3)
        first = evaluate(lambda: build_env(config.env), driver, eval_config)
        second = evaluate(lambda: build_env(config.env), driver, eval_config)
        assert first.actions_digest == second.actions_digest

    def test_a_missing_brain_file_is_an_error_not_an_untrained_brain(self, run, tmp_path):
        run_dir = run["result"].run_dir
        lone = tmp_path / FINAL_CHECKPOINT
        lone.write_bytes((run_dir / FINAL_CHECKPOINT).read_bytes())
        env = build_env(run["config"].env)
        with pytest.raises(FileNotFoundError, match="brain_final.pt"):
            load_driver(run["config"], lone, env)
        env.close()

    def test_the_evaluate_script_runs_a_brain_checkpoint(self, run, tmp_path):
        checkpoint = run["result"].run_dir / FINAL_CHECKPOINT
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "scripts" / "evaluate.py"),
                "--checkpoint",
                str(checkpoint),
                "--episodes",
                "1",
                "--output",
                str(tmp_path / "eval"),
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO,
        )
        assert result.returncode == 0, result.stderr
        assert "actions_digest:" in result.stdout
        assert (tmp_path / "eval" / "summary.json").exists()
