"""CSV always, W&B only when asked, per-term reward columns (GH-17)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from fly_driver.training import CsvLogger, RunLogger, TrainingEpisode, UpdateRecord, WandbLogger
from fly_driver.training.config import WandbConfig
from fly_driver.training.loggers import REWARD_COLUMN_PREFIX


def _episode(**overrides) -> TrainingEpisode:
    values = {
        "seed": 0,
        "episode": 0,
        "update": 1,
        "env_steps": 120,
        "steps": 120,
        "total_return": 7.5,
        "lap_complete": False,
        "lap_time_s": None,
        "terminated": True,
        "truncated": False,
        "reward_terms": {"progress": 9.0, "lateral": -1.5, "lap_bonus": 0.0, "off_track": 0.0},
    }
    values.update(overrides)
    return TrainingEpisode(**values)


def _update(**overrides) -> UpdateRecord:
    values = {
        "seed": 0,
        "update": 1,
        "env_steps": 512,
        "episodes_completed": 2,
        "mean_episode_return": 3.0,
        "mean_episode_steps": 256.0,
        "policy_loss": -0.01,
        "value_loss": 0.4,
        "entropy": 4.2,
        "approx_kl": 0.005,
        "clip_fraction": 0.1,
        "explained_variance": 0.8,
        "learning_rate": 3e-4,
        "steps_per_second": 900.0,
        "elapsed_s": 0.6,
    }
    values.update(overrides)
    return UpdateRecord(**values)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestEpisodeRows:
    def test_every_reward_term_is_its_own_column(self, tmp_path: Path):
        logger = CsvLogger(tmp_path)
        logger.log_episode(_episode())
        logger.close()

        rows = _rows(logger.episodes_path)
        assert len(rows) == 1
        term_columns = [key for key in rows[0] if key.startswith(REWARD_COLUMN_PREFIX)]
        assert term_columns == [
            "reward_lap_bonus",
            "reward_lateral",
            "reward_off_track",
            "reward_progress",
        ]
        assert float(rows[0]["reward_progress"]) == 9.0
        assert float(rows[0]["reward_lateral"]) == -1.5
        assert rows[0]["lap_time_s"] == ""  # None writes as empty, not "None"
        assert rows[0]["terminated"] == "True"

    def test_the_columns_come_before_the_terms_in_a_fixed_order(self, tmp_path: Path):
        logger = CsvLogger(tmp_path)
        logger.log_episode(_episode())
        logger.close()
        header = logger.episodes_path.read_text(encoding="utf-8").splitlines()[0]
        assert header.startswith(
            "seed,episode,update,env_steps,steps,total_return,lap_complete,lap_time_s,"
            "terminated,truncated,reward_"
        )

    def test_rows_are_on_disk_before_close(self, tmp_path: Path):
        """A run that dies mid-way still leaves its rows behind."""
        logger = CsvLogger(tmp_path)
        logger.log_episode(_episode())
        logger.log_episode(_episode(episode=1))
        assert len(_rows(logger.episodes_path)) == 2
        logger.close()

    def test_changing_the_set_of_terms_mid_run_is_an_error(self, tmp_path: Path):
        logger = CsvLogger(tmp_path)
        logger.log_episode(_episode())
        with pytest.raises(ValueError, match="columns changed"):
            logger.log_episode(_episode(reward_terms={"progress": 1.0}))
        logger.close()


class TestUpdateRows:
    def test_update_rows_land_in_their_own_file(self, tmp_path: Path):
        logger = CsvLogger(tmp_path)
        logger.log_update(_update())
        logger.log_update(_update(update=2, env_steps=1024, mean_episode_return=None))
        logger.close()

        rows = _rows(logger.updates_path)
        assert [row["update"] for row in rows] == ["1", "2"]
        assert rows[1]["mean_episode_return"] == ""
        assert {"policy_loss", "value_loss", "approx_kl", "steps_per_second"} <= set(rows[0])
        assert not logger.episodes_path.exists()  # no episode was logged


class TestWandb:
    def test_disabled_never_imports_wandb(self, monkeypatch: pytest.MonkeyPatch):
        """Off by default means the package is not even looked for."""
        monkeypatch.setitem(sys.modules, "wandb", None)
        with pytest.raises(ValueError, match="disabled"):
            WandbLogger(
                WandbConfig(enabled=False), run_name="x", group="g", run_config={}, run_dir="."
            )

    def test_enabled_without_the_package_says_how_to_fix_it(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setitem(sys.modules, "wandb", None)
        with pytest.raises(ImportError, match="pip install wandb"):
            WandbLogger(
                WandbConfig(enabled=True), run_name="x", group="g", run_config={}, run_dir="."
            )

    def test_enabled_mirrors_rows_to_the_run(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        calls: list[tuple[str, dict, int | None]] = []
        inits: list[dict] = []

        class FakeRun:
            def log(self, data, step=None):
                calls.append(("log", data, step))

            def finish(self):
                calls.append(("finish", {}, None))

        def init(**kwargs):
            inits.append(kwargs)
            return FakeRun()

        monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=init))
        config = WandbConfig(enabled=True, project="p", entity="e", mode="offline", tags=("t",))
        logger = WandbLogger(
            config, run_name="cond/seed_0", group="cond", run_config={"seed": 0}, run_dir=tmp_path
        )
        logger.log_episode(_episode())
        logger.log_update(_update())
        logger.close()

        assert inits[0]["project"] == "p" and inits[0]["entity"] == "e"
        assert inits[0]["mode"] == "offline" and inits[0]["tags"] == ["t"]
        assert inits[0]["name"] == "cond/seed_0" and inits[0]["group"] == "cond"
        assert inits[0]["config"] == {"seed": 0}
        assert calls[0][0] == "log" and calls[0][2] == 120
        assert calls[0][1]["episode/reward_progress"] == 9.0
        assert calls[1][1]["update/policy_loss"] == -0.01 and calls[1][2] == 512
        assert calls[-1][0] == "finish"


class TestRunLogger:
    def test_fans_out_and_closes_everything_even_when_one_fails(self, tmp_path: Path):
        class Broken:
            def log_episode(self, record):
                pass

            def log_update(self, record):
                pass

            def close(self):
                raise RuntimeError("network down")

        csv_logger = CsvLogger(tmp_path)
        composite = RunLogger([csv_logger, Broken()])
        composite.log_episode(_episode())
        composite.log_update(_update())
        with pytest.raises(RuntimeError, match="network down"):
            composite.close()
        assert len(_rows(csv_logger.episodes_path)) == 1
        assert csv_logger._episodes._handle is None  # the CSV was still closed
