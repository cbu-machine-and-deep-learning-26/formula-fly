"""Training logs: CSV always, Weights & Biases when asked for (GH-17).

Two tables per seed, one row per event, written as they happen so a run that dies still
leaves its numbers behind:

- ``episodes.csv``: one row per finished training episode, with the return **split into the
  reward's terms** (``reward_progress``, ``reward_lateral``, ...). That split is what
  `AGENTS.md` §11 asks for: an agent farming one term without progressing is visible in the
  columns rather than inferred from a curve.
- ``updates.csv``: one row per PPO update -- the losses, the approximate KL, how many
  episodes finished in the rollout and what they returned, and the wall-clock throughput.

The term columns come from the env (``info["reward_terms"]``) at the first episode and are
fixed for the rest of the file: a later row with different terms is an error, not a new
column, because a CSV whose header stops matching its rows is unreadable downstream.

W&B mirrors the same rows and is imported only when enabled, so nothing here needs it.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from fly_driver.training.config import WandbConfig

__all__ = [
    "EPISODES_FILENAME",
    "REWARD_COLUMN_PREFIX",
    "UPDATES_FILENAME",
    "CsvLogger",
    "RunLogger",
    "TrainingEpisode",
    "TrainingLogger",
    "UpdateRecord",
    "WandbLogger",
]

EPISODES_FILENAME = "episodes.csv"
UPDATES_FILENAME = "updates.csv"

#: Every reward term becomes a column ``reward_<term>`` in ``episodes.csv``.
REWARD_COLUMN_PREFIX = "reward_"


@dataclass(frozen=True)
class TrainingEpisode:
    """One finished training episode.

    Args:
        seed: The run's seed.
        episode: Index of the episode within the run, from 0.
        update: The PPO update whose rollout the episode ended in, from 1.
        env_steps: Env steps taken by the run when the episode ended.
        steps: Length of the episode.
        total_return: Undiscounted sum of rewards, as the env reported them.
        lap_complete: Whether a lap was completed during the episode.
        lap_time_s: The completed lap's time, or ``None``.
        terminated: Ended by the env (off track, lap done, diverged).
        truncated: Ended by the step cap.
        reward_terms: Per-term sums over the episode; they add up to ``total_return``.
    """

    seed: int
    episode: int
    update: int
    env_steps: int
    steps: int
    total_return: float
    lap_complete: bool
    lap_time_s: float | None
    terminated: bool
    truncated: bool
    reward_terms: Mapping[str, float]

    def to_row(self) -> dict[str, Any]:
        """Flat CSV row: the fixed columns, then one ``reward_<term>`` column per term."""
        row = asdict(self)
        terms = row.pop("reward_terms")
        for name in sorted(terms):
            row[f"{REWARD_COLUMN_PREFIX}{name}"] = float(terms[name])
        return row


@dataclass(frozen=True)
class UpdateRecord:
    """One PPO update.

    Args:
        seed: The run's seed.
        update: Update index, from 1.
        env_steps: Env steps taken by the run after this update's rollout.
        episodes_completed: Episodes that finished during the rollout.
        mean_episode_return: Their mean return, or ``None`` if none finished.
        mean_episode_steps: Their mean length, or ``None``.
        policy_loss: Clipped surrogate loss.
        value_loss: Value loss.
        entropy: Mean policy entropy.
        approx_kl: Approximate KL between the old and new policies.
        clip_fraction: Fraction of samples whose ratio was clipped.
        explained_variance: How much of the return variance the critic explains.
        learning_rate: Step size used for the update.
        steps_per_second: Env steps per wall-clock second over collection plus update.
        elapsed_s: Wall-clock seconds since the run started.
    """

    seed: int
    update: int
    env_steps: int
    episodes_completed: int
    mean_episode_return: float | None
    mean_episode_steps: float | None
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_fraction: float
    explained_variance: float
    learning_rate: float
    steps_per_second: float
    elapsed_s: float

    def to_row(self) -> dict[str, Any]:
        """Flat CSV row."""
        return asdict(self)


class TrainingLogger(Protocol):
    """What the loop writes to."""

    def log_episode(self, record: TrainingEpisode) -> None:
        """Record a finished episode."""

    def log_update(self, record: UpdateRecord) -> None:
        """Record a PPO update."""

    def close(self) -> None:
        """Flush and release."""


class _CsvTable:
    """One CSV file whose header is fixed by its first row."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: Any = None
        self._writer: csv.DictWriter[str] | None = None
        self._columns: list[str] | None = None

    def write(self, row: Mapping[str, Any]) -> None:
        if self._writer is None or self._handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", newline="", encoding="utf-8")
            self._columns = list(row)
            self._writer = csv.DictWriter(self._handle, fieldnames=self._columns)
            self._writer.writeheader()
        elif list(row) != self._columns:
            raise ValueError(
                f"{self.path.name} columns changed mid-run: header {self._columns}, "
                f"row {list(row)}. Reward terms must be the same every episode."
            )
        self._writer.writerow(dict(row))
        # Flushed per row so an interrupted run's numbers are on disk, not in a buffer.
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None


class CsvLogger:
    """``episodes.csv`` and ``updates.csv`` under ``run_dir``, written row by row.

    Args:
        run_dir: One seed's run directory; created on the first row.
    """

    def __init__(self, run_dir: str | os.PathLike[str]) -> None:
        self.run_dir = Path(run_dir)
        self._episodes = _CsvTable(self.run_dir / EPISODES_FILENAME)
        self._updates = _CsvTable(self.run_dir / UPDATES_FILENAME)

    @property
    def episodes_path(self) -> Path:
        """Where the episode rows go."""
        return self._episodes.path

    @property
    def updates_path(self) -> Path:
        """Where the update rows go."""
        return self._updates.path

    def log_episode(self, record: TrainingEpisode) -> None:
        """Append one episode row."""
        self._episodes.write(record.to_row())

    def log_update(self, record: UpdateRecord) -> None:
        """Append one update row."""
        self._updates.write(record.to_row())

    def close(self) -> None:
        """Close both files. Safe to call more than once."""
        self._episodes.close()
        self._updates.close()


class WandbLogger:
    """Mirror the CSV rows to a Weights & Biases run. Off by default; see :class:`WandbConfig`.

    Args:
        config: The W&B settings; ``enabled`` must be true.
        run_name: The W&B run name, normally ``<condition>/seed_<k>``.
        group: The W&B group, normally the condition name, so seeds sit together.
        run_config: The resolved training config, stored on the run.
        run_dir: Where W&B keeps its local files.

    Raises:
        ImportError: If ``wandb`` is not installed. Nothing imports it otherwise.
        ValueError: If the config has W&B disabled.
    """

    def __init__(
        self,
        config: WandbConfig,
        *,
        run_name: str,
        group: str,
        run_config: Mapping[str, Any],
        run_dir: str | os.PathLike[str],
    ) -> None:
        if not config.enabled:
            raise ValueError("WandbLogger built from a config with wandb disabled")
        try:
            import wandb
        except ImportError as error:
            raise ImportError(
                "logging.wandb.enabled is true but wandb is not installed: "
                "`pip install wandb`, or set it back to false (CSV is always written)."
            ) from error
        self._run = wandb.init(
            project=config.project,
            entity=config.entity,
            mode=config.mode,
            tags=list(config.tags),
            name=run_name,
            group=group,
            config=dict(run_config),
            dir=str(run_dir),
            reinit=True,
        )

    def log_episode(self, record: TrainingEpisode) -> None:
        """Log an episode row under ``episode/``, stepped by env steps."""
        row = {f"episode/{key}": value for key, value in record.to_row().items()}
        self._run.log(row, step=record.env_steps)

    def log_update(self, record: UpdateRecord) -> None:
        """Log an update row under ``update/``, stepped by env steps."""
        row = {f"update/{key}": value for key, value in record.to_row().items()}
        self._run.log(row, step=record.env_steps)

    def close(self) -> None:
        """Finish the W&B run."""
        self._run.finish()


class RunLogger:
    """Fan one event out to several loggers.

    Args:
        loggers: Loggers to write to, in order. The CSV logger goes first so its row is on
            disk before an optional network logger gets a chance to fail.
    """

    def __init__(self, loggers: Sequence[TrainingLogger]) -> None:
        self.loggers = tuple(loggers)

    def log_episode(self, record: TrainingEpisode) -> None:
        """Forward to every logger."""
        for logger in self.loggers:
            logger.log_episode(record)

    def log_update(self, record: UpdateRecord) -> None:
        """Forward to every logger."""
        for logger in self.loggers:
            logger.log_update(record)

    def close(self) -> None:
        """Close every logger, even if an earlier one raises."""
        errors: list[Exception] = []
        for logger in self.loggers:
            try:
                logger.close()
            except Exception as error:  # noqa: BLE001 - closing must reach every logger
                errors.append(error)
        if errors:
            raise errors[0]
