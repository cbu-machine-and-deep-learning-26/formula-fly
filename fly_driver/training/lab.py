"""What the lab PC supervisor decides, apart from the processes it runs (GH-27).

``scripts/lab_train.py`` is the command. This module holds its window scoring, checkpoint
choice and restart decision as plain functions, and keeps ``weights/best/`` up to date.
"""

from __future__ import annotations

import bisect
import csv
import json
import os
import re
import shutil
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

BEST_POLICY = "policy_best.pt"  # warm_start finds brain_best.pt beside it
BEST_BRAIN = "brain_best.pt"
BEST_RECORD = "best.json"

MAX_TRAINERS = 3  # the team's limit per GPU: more brings on NVIDIA driver resets
WINDOW_DRIVES = 50
WINDOW_STEP = 5
WARMUP_UPDATES = 150
RESTART_FRACTION = 0.7

# python must be the executable: a shell or an editor with train.py on its line is no trainer.
# The character before train.py keeps lab_train.py out.
_TRAINER_COMMAND = re.compile(
    r"""^("[^"]*python[^"]*"|\S*python\S*)\s.*[\s\\/"']train\.py\b""", re.IGNORECASE
)


class Drive(NamedTuple):
    """One row of a run's ``episodes.csv``, as far as this script reads it."""

    update: int
    steps: int
    distance_m: float
    lap_complete: bool
    start_focused: bool = False


class Window(NamedTuple):
    """A 50-drive window that can be kept: its mean distance and the checkpoint behind it."""

    score: float
    last_update: int
    checkpoint: int


class Process(NamedTuple):
    """A running process, from the machine's process table."""

    pid: int
    command: str


@dataclass
class State:
    """What ``status`` and ``stop`` need from another terminal, and a restart must not lose.

    Args:
        generation: The ``g<N>`` of the newest run; only ever goes up.
        supervisor_pid: The supervisor's process id, or ``None`` after a clean stop.
        seeds: The seeds of the latest ``start``.
        runs: Every ``[run, seed]`` this script has started, oldest first.
        last_restart_check: When the last restart-from-best decision was made.
    """

    generation: int = 0
    supervisor_pid: int | None = None
    seeds: list[int] = field(default_factory=list)
    runs: list[tuple[str, int]] = field(default_factory=list)
    last_restart_check: float = 0.0


def check_seeds(seeds: Sequence[int]) -> None:
    """Refuse more seeds than one GPU may train, or the same seed twice.

    Raises:
        ValueError: If there are more than ``MAX_TRAINERS`` seeds, or duplicates.
    """
    if len(seeds) > MAX_TRAINERS:
        raise ValueError(
            f"{len(seeds)} seeds asked for, the limit is {MAX_TRAINERS} per GPU: more "
            "brings on NVIDIA driver resets. Pick some with --seeds."
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct, got {list(seeds)}")


def nearest_checkpoint(update: int, checkpoints: Sequence[int]) -> int | None:
    """The newest of the sorted ``checkpoints`` at or before ``update``, or ``None``."""
    index = bisect.bisect_right(checkpoints, update)
    return checkpoints[index - 1] if index else None


def scored_drives(drives: Sequence[Drive]) -> list[Drive]:
    """The drives a score may count: the uniform-start ones, without any focused starts.

    A focused drive starts near the corner it is there to practise, so it is short by design.
    Scoring it would hold the last-50 average under ``RESTART_FRACTION`` of the record, and a
    seed would be sent back to the same checkpoint every cycle while nothing could beat it.
    The uniform share is the distribution every earlier run was scored on, so the record means
    the same thing either side of the change.
    """
    return [drive for drive in drives if not drive.start_focused]


def best_window(drives: Sequence[Drive], checkpoints: Sequence[int]) -> Window | None:
    """The best 50-drive window that has a checkpoint, scanned in steps of 5 rows.

    Only :func:`scored_drives` count, so a window can span more rows than 50 under a focus.

    A window ending before the first checkpoint is skipped, not returned, because its weights
    were never saved. A warm-started run often peaks that early, and returning it would hide
    every later window of the run that could still beat the record.

    Args:
        drives: A run's drives, in order.
        checkpoints: The run's saved checkpoint updates, sorted.
    """
    best: Window | None = None
    scored = scored_drives(drives)
    for start in range(0, len(scored) - WINDOW_DRIVES + 1, WINDOW_STEP):
        window = scored[start : start + WINDOW_DRIVES]
        checkpoint = nearest_checkpoint(window[-1].update, checkpoints)
        if checkpoint is None:
            continue
        score = sum(drive.distance_m for drive in window) / WINDOW_DRIVES
        if best is None or score > best.score:
            best = Window(score, window[-1].update, checkpoint)
    return best


def recent_score(drives: Sequence[Drive]) -> float | None:
    """Mean distance of the last 50 :func:`scored_drives`, or ``None`` before there are 50."""
    scored = scored_drives(drives)
    if len(scored) < WINDOW_DRIVES:
        return None
    return sum(drive.distance_m for drive in scored[-WINDOW_DRIVES:]) / WINDOW_DRIVES


def seed_to_restart(
    scores: Mapping[int, float],
    last_updates: Mapping[int, int],
    best_score: float,
    restart_fraction: float = RESTART_FRACTION,
) -> int | None:
    """The seed to restart from best this cycle, or ``None``. Only the worst seed can go.

    Args:
        scores: Each seed's last-50 mean distance. A seed with fewer drives is left out.
        last_updates: Each seed's last PPO update in its current run.
        best_score: The recorded best window.
        restart_fraction: Restart only below this share of ``best_score``. 0 never restarts.
    """
    if not scores:
        return None
    worst = min(scores, key=lambda seed: scores[seed])
    is_warmed_up = last_updates[worst] > WARMUP_UPDATES
    return worst if is_warmed_up and scores[worst] < restart_fraction * best_score else None


def trainer_processes(processes: Sequence[Process]) -> list[Process]:
    """The processes running ``train.py``, this checkout's or anyone else's."""
    return [process for process in processes if _TRAINER_COMMAND.search(process.command)]


def one_per_trainer(processes: Sequence[Process]) -> list[Process]:
    """One process per trainer, for people to read: a Windows venv's python.exe has a child."""
    trainers: dict[str, Process] = {}
    for process in processes:
        match = _TRAINER_COMMAND.match(process.command)
        arguments = process.command[match.end(1) :].strip() if match else process.command
        trainers.setdefault(arguments, process)
    return list(trainers.values())


def next_run_name(state: State, today: datetime) -> str:
    """Take the next generation: ``lab_<YYYYMMDD>_g<N>``. Bumps ``state.generation``."""
    state.generation += 1
    return f"lab_{today:%Y%m%d}_g{state.generation}"


def read_drives(run_dir: Path) -> list[Drive]:
    """A run's drives, in order; empty if it has none. Skips a row torn by a killed trainer."""
    path = run_dir / "episodes.csv"
    if not path.exists():
        return []
    drives = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                drives.append(
                    Drive(
                        int(row["update"]),
                        int(row["steps"]),
                        float(row["distance_m"]),
                        row["lap_complete"] == "True",
                        # Runs from before the column read as unfocused, which they were.
                        row.get("start_focused") == "True",
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
    return drives


def saved_checkpoints(run_dir: Path) -> list[int]:
    """Sorted updates with a whole policy and brain. A killed trainer can leave a torn file."""
    updates = []
    for policy in run_dir.glob("policy_update_*.pt"):
        brain = policy.with_name(policy.name.replace("policy_", "brain_", 1))
        # torch.save writes a zip, whose directory comes last, so a torn file fails this.
        if brain.exists() and zipfile.is_zipfile(policy) and zipfile.is_zipfile(brain):
            updates.append(int(policy.stem.rsplit("_", 1)[1]))
    return sorted(updates)


def read_best(best_dir: Path) -> dict[str, Any]:
    """``best.json``'s record: score, run, seed and update.

    Raises:
        FileNotFoundError: If the policy, the brain or the record is missing, with what to do.
        ValueError: If the record has no score. Without one any drive would replace the best.
    """
    names = (BEST_POLICY, BEST_BRAIN, BEST_RECORD)
    missing = [name for name in names if not (best_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{best_dir} is missing {', '.join(missing)}. Every run warm-starts from it and "
            "never from scratch: copy weights/best/ from the other PC (AGENTS.md, Lab PCs)."
        )
    record: dict[str, Any] = json.loads((best_dir / BEST_RECORD).read_text(encoding="utf-8"))
    if "score" not in record:
        raise ValueError(f"{best_dir / BEST_RECORD} has no score")
    return record


def write_json(path: Path, data: object) -> None:
    """Replace a JSON file in one step, so another terminal never reads half of it."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def clear_record(best_dir: Path, note: str) -> dict[str, Any]:
    """Keep ``best_dir``'s checkpoint but forget its score, for a run not comparable with it.

    A score only means something on the car and rewards it was earned on. Carried over to a new
    car, it holds every seed under the restart rule and stops any new best being recorded. At
    score 0 the first window with a checkpoint becomes the record, and restarts wait for it.

    Args:
        best_dir: The folder's ``weights/best``.
        note: Where the checkpoint came from, kept in the record's ``run``.

    Returns:
        The new record.

    Raises:
        FileNotFoundError: If the checkpoint pair or record is missing.
    """
    read_best(best_dir)
    record = {"score": 0.0, "run": note, "seed": None, "update": None}
    write_json(best_dir / BEST_RECORD, record)
    return record


def promote_best(runs: Sequence[tuple[str, int]], runs_dir: Path, best_dir: Path) -> str | None:
    """Copy the best checkpoint of ``runs`` into ``best_dir`` if it beats the record.

    Returns:
        A line saying what changed, or ``None`` if the record stands.
    """
    best_score = float(read_best(best_dir)["score"])
    winner: tuple[Window, str, int] | None = None
    for run, seed in runs:
        run_dir = runs_dir / run / f"seed_{seed}"
        window = best_window(read_drives(run_dir), saved_checkpoints(run_dir))
        if window is not None and window.score > best_score:
            best_score = window.score
            winner = (window, run, seed)
    if winner is None:
        return None
    window, run, seed = winner
    source = runs_dir / run / f"seed_{seed}"
    pairs = (
        (source / f"policy_update_{window.checkpoint:05d}.pt", best_dir / BEST_POLICY),
        (source / f"brain_update_{window.checkpoint:05d}.pt", best_dir / BEST_BRAIN),
    )
    # A trainer that is starting up reads these files: copy aside, then swap them together.
    for origin, target in pairs:
        shutil.copyfile(origin, target.with_name(target.name + ".tmp"))
    for _, target in pairs:
        os.replace(target.with_name(target.name + ".tmp"), target)
    record = {"score": window.score, "run": run, "seed": seed, "update": window.checkpoint}
    write_json(best_dir / BEST_RECORD, record)
    return f"new best {window.score:.0f} m: {run} seed {seed}, update {window.checkpoint}"
