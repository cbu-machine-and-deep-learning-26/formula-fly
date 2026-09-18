"""Determinism, config, metrics, and output tests for the evaluation harness."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fly_driver.envs import DummyTrackEnv
from fly_driver.policies import ConstantAgent, RandomAgent
from fly_driver.training import CLEAN_CONDITION, EvalConfig, EvalReport, evaluate
from fly_driver.training.perturbations import PerturbationSpec

PERTURBED = EvalConfig(
    episodes=2,
    seed=11,
    max_steps=60,
    perturbations=(
        PerturbationSpec("observation_noise", {"std": 0.1}),
        PerturbationSpec("action_delay", {"steps": 2}),
    ),
)


class CentreSteeringAgent:
    """Reads the road position from the frame and steers toward it."""

    def act(self, observation: np.ndarray) -> np.ndarray:
        columns = np.flatnonzero(observation[0, :, 0] > 100)
        width = observation.shape[1]
        offset = (columns.mean() - width / 2) / (width / 2) if columns.size else 0.0
        return np.array([np.clip(2.0 * offset, -1, 1), 1.0, 0.0], dtype=np.float32)


def _make_env() -> DummyTrackEnv:
    return DummyTrackEnv()


def test_same_config_gives_identical_actions_and_metrics() -> None:
    """Same seed, same actions: the determinism guarantee for CI (AGENTS.md §11)."""
    first = evaluate(_make_env, RandomAgent(), PERTURBED)
    np.random.seed(999)  # noqa: NPY002  global RNG must not leak into the protocol
    second = evaluate(_make_env, RandomAgent(seed=12345), PERTURBED)

    assert first.actions_digest == second.actions_digest
    assert len(first.episodes) == 3 * PERTURBED.episodes
    for left, right in zip(first.episodes, second.episodes, strict=True):
        np.testing.assert_array_equal(left.actions, right.actions)
        assert left.to_row() == right.to_row()
    assert first.to_dict() == second.to_dict()


def test_different_seeds_change_the_actions() -> None:
    """The seed is actually wired through to the agent and env."""
    baseline = evaluate(_make_env, RandomAgent(), PERTURBED)
    other = evaluate(_make_env, RandomAgent(), EvalConfig(episodes=2, seed=12, max_steps=60))
    assert baseline.actions_digest != other.actions_digest
    assert baseline.episodes[0].seed == 11 and other.episodes[0].seed == 12


def test_lap_time_and_completion_are_reported() -> None:
    """A lap-completing agent yields lap time from the env's ``lap_time``."""
    report = evaluate(_make_env, CentreSteeringAgent(), EvalConfig(episodes=2, seed=0))
    clean = report.summary(CLEAN_CONDITION)

    assert clean.lap_completion_rate == 1.0
    assert clean.mean_lap_time_s is not None and clean.mean_lap_time_s > 0
    for episode in report.episodes:
        assert episode.lap_complete and episode.terminated and not episode.truncated
        assert episode.lap_time_s == pytest.approx(episode.steps / 50.0)


def test_lap_time_falls_back_to_steps_over_frame_rate() -> None:
    """Envs that only flag ``lap_complete`` still get a lap time."""

    class FlagOnlyEnv:
        def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict]:
            self.steps = 0
            return np.zeros((4, 4, 3), dtype=np.uint8), {}

        def step(self, action: object) -> tuple[np.ndarray, float, bool, bool, dict]:
            self.steps += 1
            done = self.steps == 25
            return (
                np.zeros((4, 4, 3), dtype=np.uint8),
                1.0,
                done,
                False,
                {"lap_complete": done},
            )

        def render(self) -> None:
            return None

    report = evaluate(FlagOnlyEnv, ConstantAgent(), EvalConfig(episodes=1, frame_rate_hz=50.0))
    assert report.episodes[0].lap_time_s == pytest.approx(0.5)


def test_robustness_conditions_share_seeds_and_report_ratio() -> None:
    """Perturbed conditions reuse the clean seeds and report return ratios."""
    report = evaluate(_make_env, CentreSteeringAgent(), PERTURBED)

    assert [item.condition for item in report.conditions] == list(PERTURBED.condition_labels)
    clean, noise, delay = report.conditions
    assert clean.return_ratio_to_clean is None
    assert noise.return_ratio_to_clean == pytest.approx(noise.mean_return / clean.mean_return)
    assert delay.return_ratio_to_clean == pytest.approx(delay.mean_return / clean.mean_return)
    seeds_by_condition = {
        label: [e.seed for e in report.episodes if e.condition == label]
        for label in PERTURBED.condition_labels
    }
    assert all(seeds == [11, 12] for seeds in seeds_by_condition.values())
    delayed = [e.actions for e in report.episodes if e.condition == delay.condition]
    assert all(len(actions) > 2 for actions in delayed)


def test_harness_max_steps_truncates() -> None:
    """The config cap applies even when the env would continue."""
    report = evaluate(_make_env, ConstantAgent(), EvalConfig(episodes=1, max_steps=7))
    episode = report.episodes[0]
    assert episode.steps == 7 and episode.truncated and not episode.terminated


def test_agent_output_is_validated() -> None:
    """Wrong shapes and non-finite actions are errors, not silently fixed."""

    class BadAgent:
        def act(self, observation: object) -> np.ndarray:
            return np.array([0.0, 0.0])

    with pytest.raises(ValueError, match="shape"):
        evaluate(_make_env, BadAgent(), EvalConfig(episodes=1))


def test_outputs_are_written(tmp_path: Path) -> None:
    """``episodes.csv`` and ``summary.json`` land in ``output_dir``."""
    config = EvalConfig(episodes=2, seed=1, max_steps=20, output_dir=str(tmp_path / "run"))
    report = evaluate(_make_env, ConstantAgent(), config)

    with (tmp_path / "run" / "episodes.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert {
        "condition",
        "episode",
        "seed",
        "steps",
        "total_return",
        "lap_time_s",
    } <= set(rows[0])
    summary = json.loads((tmp_path / "run" / "summary.json").read_text(encoding="utf-8"))
    assert summary["actions_digest"] == report.actions_digest
    assert summary["config"] == config.to_dict()
    assert summary["conditions"][0]["condition"] == CLEAN_CONDITION
    assert summary["video_backend"] is None


def test_video_skips_cleanly_without_a_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No codec installed: a warning, no files, and the evaluation still completes."""
    monkeypatch.setitem(sys.modules, "imageio", None)
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)
    monkeypatch.setitem(sys.modules, "cv2", None)
    config = EvalConfig(
        episodes=2, max_steps=10, record_video=True, output_dir=str(tmp_path / "run")
    )

    with pytest.warns(UserWarning, match="no video backend"):
        report = evaluate(_make_env, ConstantAgent(), config)

    assert report.video_backend is None
    assert all(episode.video_path is None for episode in report.episodes)
    assert not (tmp_path / "run" / "videos").exists()


def test_video_is_written_when_a_backend_exists(tmp_path: Path) -> None:
    """With imageio-ffmpeg installed the first episode of each condition is recorded."""
    pytest.importorskip("imageio_ffmpeg")
    config = EvalConfig(
        episodes=2,
        max_steps=10,
        record_video=True,
        video_episodes=1,
        output_dir=str(tmp_path / "run"),
        perturbations=(PerturbationSpec("brightness", {"scale": 0.5}),),
    )

    report = evaluate(_make_env, ConstantAgent(), config)

    assert report.video_backend == "imageio"
    recorded = [episode for episode in report.episodes if episode.video_path is not None]
    assert [episode.episode for episode in recorded] == [0, 0]
    for episode in recorded:
        path = Path(episode.video_path)
        assert path.exists() and path.stat().st_size > 0 and path.suffix == ".mp4"
    assert report.episodes[1].video_path is None


def test_config_from_dict_and_yaml(tmp_path: Path) -> None:
    """Config round-trips through dict and YAML and rejects unknown keys."""
    yaml = pytest.importorskip("yaml")
    data = {
        "episodes": 4,
        "seed": 3,
        "record_video": True,
        "perturbations": [{"name": "brightness", "scale": 0.7}],
    }
    config = EvalConfig.from_dict(data)
    assert config.episodes == 4 and config.record_video is True
    assert config.perturbations[0].label == "brightness(scale=0.7)"
    assert EvalConfig.from_dict(config.to_dict()) == config

    path = tmp_path / "eval.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert EvalConfig.from_yaml(path) == config

    with pytest.raises(ValueError, match="unknown eval config keys"):
        EvalConfig.from_dict({"episode": 1})


def test_default_config_file_loads() -> None:
    """The checked-in default config is valid."""
    pytest.importorskip("yaml")
    path = Path(__file__).parents[2] / "fly_driver" / "configs" / "eval_default.yaml"
    config = EvalConfig.from_yaml(path)
    assert config.episodes == 3 and len(config.perturbations) == 3


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"episodes": 0}, "episodes"),
        ({"video_episodes": 5}, "video_episodes"),
        ({"frame_rate_hz": 0.0}, "frame_rate_hz"),
        ({"max_steps": 0}, "max_steps"),
        ({"perturbations": ({"name": "unknown"},)}, "unknown perturbation"),
        (
            {
                "perturbations": (
                    {"name": "brightness", "scale": 1},
                    {"name": "brightness", "scale": 1},
                )
            },
            "unique",
        ),
    ],
)
def test_config_validation(kwargs: dict[str, object], match: str) -> None:
    """Bad protocol settings fail at construction."""
    with pytest.raises(ValueError, match=match):
        EvalConfig(**kwargs)  # type: ignore[arg-type]


def test_report_summary_lookup_errors_on_unknown_condition() -> None:
    """Asking for a condition that was not run is a KeyError."""
    report = evaluate(_make_env, ConstantAgent(), EvalConfig(episodes=1, max_steps=5))
    assert isinstance(report, EvalReport)
    with pytest.raises(KeyError):
        report.summary("nope")
