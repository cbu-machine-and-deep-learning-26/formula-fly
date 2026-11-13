"""Train in segments with Claude reviewing the reward between them (GH-27, coached run).

One long GPU job, cut into segments:

1. Train a segment of ``num_envs`` tracks for a fixed wall-clock budget, warm-started from the
   best checkpoint so far (from scratch for the first segment when there is none).
2. Keep the segment's best 50-drive window (:func:`fly_driver.training.lab.best_window`, the
   lab supervisor's score: mean metres over uniform-start drives) if it beats the record.
3. Ask the reward coach (:func:`fly_driver.coach.review.review_run`) to review the segment.
   The first proposal that passed the guardrails becomes the next segment's recipe; with
   none accepted the recipe stays as it was.

The score is distance driven, not reward, so it means the same thing whichever reward a
segment trained on, and segments trained on different rewards can be compared.

Claude never scores steps or drives here either: it only changes the reward's knobs, within
:mod:`fly_driver.coach.guardrails`, between segments.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from fly_driver.training.lab import (
    BEST_BRAIN,
    BEST_POLICY,
    best_window,
    read_drives,
    saved_checkpoints,
    write_json,
)

__all__ = ["OVERRIDE_FILE", "SegmentPlan", "promote_segment_best", "segment_config", "run_coached"]

# Keys the loop owns: a coach proposal may not change where a run goes or how big it is.
_LOOP_KEYS = ("name", "output_dir", "init_from", "seeds", "num_envs", "total_env_steps")
_STOP_GRACE_S = 60.0
_MAX_FAILURES = 2
#: Drop a recipe here (``<output>/<prefix>/override.yaml``) to change the next segment's.
OVERRIDE_FILE = "override.yaml"


@dataclass(frozen=True)
class SegmentPlan:
    """How one coached job is cut up.

    Attributes:
        prefix: Run name prefix; segment ``k`` is ``<prefix>_s<k>``.
        output_dir: Where runs go.
        num_envs: Tracks per trainer.
        steps_per_env: Rollout length per track; ``ppo.rollout_steps`` is ``num_envs`` times it.
        first_segment_s: Wall-clock budget of the first segment.
        segment_s: Wall-clock budget of every later segment.
        deadline: ``time.time()`` past which no segment may run.
        min_segment_s: A segment shorter than this is not started.
    """

    prefix: str
    output_dir: Path
    num_envs: int
    steps_per_env: int
    first_segment_s: float
    segment_s: float
    deadline: float
    min_segment_s: float = 1200.0


def segment_config(
    recipe: Mapping[str, Any], plan: SegmentPlan, index: int, init_from: Path | None
) -> dict[str, Any]:
    """The full run config for segment ``index``: the recipe with the loop's own keys.

    Args:
        recipe: The current recipe (a run config, e.g. a coach proposal).
        plan: The job's plan.
        index: Segment number, from 0.
        init_from: The best policy checkpoint so far, or ``None`` to start from scratch.
    """
    config = copy.deepcopy(dict(recipe))
    config.update(
        name=f"{plan.prefix}_s{index}",
        output_dir=str(plan.output_dir),
        init_from=str(init_from) if init_from is not None else None,
        seeds=[0],
        num_envs=plan.num_envs,
        # The wall-clock budget ends the segment; this only has to be out of reach.
        total_env_steps=10**10,
    )
    config["ppo"] = {**config.get("ppo", {}), "rollout_steps": plan.num_envs * plan.steps_per_env}
    return config


def promote_segment_best(seed_dir: Path, best_dir: Path, record: Mapping[str, Any]) -> dict | None:
    """Copy ``seed_dir``'s best window's checkpoint to ``best_dir`` if it beats ``record``.

    Returns:
        The new record, or ``None`` when the old one stands.
    """
    window = best_window(read_drives(seed_dir), saved_checkpoints(seed_dir))
    if window is None or window.score <= float(record.get("score", 0.0)):
        return None
    best_dir.mkdir(parents=True, exist_ok=True)
    for stage, target in (("policy", BEST_POLICY), ("brain", BEST_BRAIN)):
        source = seed_dir / f"{stage}_update_{window.checkpoint:05d}.pt"
        temporary = best_dir / (target + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, best_dir / target)
    new_record = {
        "score": window.score,
        "run": seed_dir.parent.name,
        "seed": int(seed_dir.name.removeprefix("seed_")),
        "update": window.checkpoint,
        "window_end_update": window.last_update,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    write_json(best_dir / "best.json", new_record)
    return new_record


def _log(log_path: Path, **event: Any) -> None:
    event = {"time": datetime.now().isoformat(timespec="seconds"), **event}
    line = json.dumps(event, default=str)
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _train_segment(config_path: Path, budget_s: float, log_file: Path) -> int | None:
    """Run ``scripts/train.py`` for ``budget_s`` seconds, then stop it and its workers.

    SIGINT first, as Ctrl-C would, then SIGKILL after a grace period. A torn checkpoint or
    CSV row is harmless: :func:`saved_checkpoints` and :func:`read_drives` skip them.
    """
    script = Path(__file__).resolve().parents[2] / "scripts" / "train.py"
    with log_file.open("a", encoding="utf-8") as output:
        process = subprocess.Popen(
            [sys.executable, "-u", str(script), "--config", str(config_path), "--overwrite"],
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=budget_s)
        except subprocess.TimeoutExpired:
            pass
        for sig, wait_s in ((signal.SIGINT, _STOP_GRACE_S), (signal.SIGKILL, 10.0)):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.wait(timeout=wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
        return None


def run_coached(
    recipe: Mapping[str, Any],
    plan: SegmentPlan,
    *,
    best_dir: Path,
    review: bool = True,
    note: str | None = None,
    coach_dir: Path | None = None,
) -> None:
    """Train segments until the deadline, with a coach review after each one.

    Args:
        recipe: The first segment's recipe.
        plan: Segment sizes and the deadline.
        best_dir: Where the best checkpoint and ``best.json`` are kept.
        review: Ask Claude between segments; ``False`` keeps the first recipe throughout.
        note: Context passed to every review.
        coach_dir: Where reviews and their history go; default ``<output>/<prefix>/coach``.
    """
    from fly_driver.coach.review import review_run  # imports anthropic only when used

    root = plan.output_dir / plan.prefix
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "coach_loop.jsonl"
    coach_dir = coach_dir or root / "coach"
    record_path = best_dir / "best.json"
    record: dict[str, Any] = (
        json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    )
    recipe = dict(recipe)
    index = 0
    failures = 0
    while True:
        budget = plan.first_segment_s if index == 0 else plan.segment_s
        remaining = plan.deadline - time.time()
        budget = min(budget, remaining)
        if budget < plan.min_segment_s:
            _log(log_path, event="done", reason="deadline", remaining_s=round(remaining))
            return
        override = root / OVERRIDE_FILE
        if override.exists():
            # A person steering a running job: their recipe wins over the coach's for this
            # segment on, and the file is kept under a new name as the record of it.
            data = yaml.safe_load(override.read_text(encoding="utf-8")) or {}
            recipe = {k: v for k, v in data.items() if k not in _LOOP_KEYS}
            override.rename(root / f"override_applied_s{index}.yaml")
            _log(log_path, event="recipe_overridden", segment=index)
        init_from = best_dir / BEST_POLICY if record else None
        config = segment_config(recipe, plan, index, init_from)
        config_path = root / f"segment_{index}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        _log(
            log_path,
            event="segment_start",
            segment=index,
            budget_s=round(budget),
            init_from=config["init_from"],
            reward=config.get("env", {}).get("params", {}).get("reward"),
        )
        exit_code = _train_segment(config_path, budget, root / f"segment_{index}.log")
        seed_dir = plan.output_dir / config["name"] / "seed_0"
        drives = read_drives(seed_dir)
        policy_drives = [drive for drive in drives if not drive.teacher_driven]
        new_record = promote_segment_best(seed_dir, best_dir, record)
        if new_record is not None:
            record = new_record
        _log(
            log_path,
            event="segment_end",
            segment=index,
            exit_code=exit_code,
            # The policy's drives only: a teacher-driven drive (env.params.teacher_drive) laps
            # because the teacher can, and must never read as the fly's first lap.
            drives=len(policy_drives),
            laps=sum(drive.lap_complete for drive in policy_drives),
            longest_m=round(max((drive.distance_m for drive in policy_drives), default=0.0)),
            teacher_drives=len(drives) - len(policy_drives),
            new_best=new_record,
            best_score=record.get("score"),
        )
        # A trainer that exits on its own before its budget crashed, unless it drove: two in a
        # row is a broken recipe or environment, and retrying until the deadline wastes the GPU.
        failures = failures + 1 if exit_code not in (None, 0) and len(drives) < 50 else 0
        if failures >= _MAX_FAILURES:
            _log(log_path, event="done", reason="segments failing", log=f"segment_{index}.log")
            return
        if review and len(drives) >= 50 and plan.deadline - time.time() > plan.min_segment_s:
            try:
                # One coach folder for the whole job, so each review sees the ones before it.
                result = review_run(
                    [plan.output_dir / config["name"]], coach_dir=coach_dir, note=note
                )
                _log(
                    log_path,
                    event="review",
                    segment=index,
                    directory=result.directory,
                    verdict=result.review.get("verdict"),
                    accepted=list(result.accepted),
                    rejected=result.rejected,
                    usage=result.usage,
                )
                # Claude lists proposals best first; take the first one that reads back.
                for title, path in result.accepted.items():
                    try:
                        proposal = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
                    except yaml.YAMLError as error:
                        _log(log_path, event="proposal_unreadable", title=title, error=repr(error))
                        continue
                    recipe = {k: v for k, v in proposal.items() if k not in _LOOP_KEYS}
                    _log(log_path, event="recipe_changed", segment=index + 1, title=title)
                    break
            except Exception as error:  # noqa: BLE001  a failed review must not end the job
                _log(log_path, event="review_failed", segment=index, error=repr(error))
        index += 1
