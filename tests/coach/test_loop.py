"""Tests for the coached segment loop's pure parts: configs and keeping the best."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from fly_driver.coach.loop import SegmentPlan, promote_segment_best, segment_config  # noqa: E402


def _plan(tmp_path: Path) -> SegmentPlan:
    return SegmentPlan(
        prefix="job",
        output_dir=tmp_path,
        num_envs=8,
        steps_per_env=512,
        first_segment_s=10.0,
        segment_s=10.0,
        deadline=time.time() + 60,
    )


def _write_run(seed_dir: Path, distances: list[float], checkpoints: list[int]) -> None:
    seed_dir.mkdir(parents=True)
    with (seed_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["update", "steps", "distance_m", "lap_complete", "start_focused"])
        for index, distance in enumerate(distances):
            writer.writerow([index, 100, distance, False, False])
    for update in checkpoints:
        torch.save({"update": update}, seed_dir / f"policy_update_{update:05d}.pt")
        torch.save({"update": update}, seed_dir / f"brain_update_{update:05d}.pt")


def test_segment_config_keeps_the_loop_keys_over_a_proposal(tmp_path: Path) -> None:
    proposal = {
        "name": "coach_proposal",
        "seeds": [0, 1, 2],
        "num_envs": 16,
        "init_from": "weights/best/policy_best.pt",
        "env": {"params": {"reward": {"crash_penalty_m": 12.0}}},
        "ppo": {"rollout_steps": 2048, "gamma": 0.997},
    }
    config = segment_config(proposal, _plan(tmp_path), 3, tmp_path / "best" / "policy_best.pt")
    assert config["name"] == "job_s3"
    assert config["seeds"] == [0]
    assert config["num_envs"] == 8
    assert config["ppo"] == {"rollout_steps": 8 * 512, "gamma": 0.997}
    assert config["init_from"] == str(tmp_path / "best" / "policy_best.pt")
    assert config["env"]["params"]["reward"]["crash_penalty_m"] == 12.0
    assert proposal["ppo"]["rollout_steps"] == 2048  # the proposal itself is untouched


def test_first_segment_starts_from_scratch(tmp_path: Path) -> None:
    assert segment_config({}, _plan(tmp_path), 0, None)["init_from"] is None


def test_promotes_only_a_better_window(tmp_path: Path) -> None:
    seed_dir = tmp_path / "job_s0" / "seed_0"
    # 100 drives: the second 50 average 300 m; the checkpoint at update 90 is behind them.
    _write_run(seed_dir, [100.0] * 50 + [300.0] * 50, checkpoints=[10, 90])
    best_dir = tmp_path / "best"

    record = promote_segment_best(seed_dir, best_dir, {})
    assert record is not None
    assert record["score"] == 300.0
    assert record["update"] == 90
    assert torch.load(best_dir / "policy_best.pt")["update"] == 90
    assert json.loads((best_dir / "best.json").read_text())["score"] == 300.0

    assert promote_segment_best(seed_dir, best_dir, {"score": 301.0}) is None


def test_an_override_file_replaces_the_recipe_for_the_next_segment(tmp_path: Path) -> None:
    import yaml

    from fly_driver.coach import loop

    plan = SegmentPlan(
        prefix="job",
        output_dir=tmp_path,
        num_envs=2,
        steps_per_env=8,
        first_segment_s=1.0,
        segment_s=1.0,
        deadline=time.time() + 5,
        min_segment_s=0.5,
    )
    root = tmp_path / "job"
    root.mkdir()
    (root / loop.OVERRIDE_FILE).write_text(
        yaml.safe_dump({"name": "ignored", "ppo": {"teacher_coef": 0.5}}), encoding="utf-8"
    )
    seen: list[dict] = []

    def fake_train(config_path: Path, budget_s: float, log_file: Path) -> int:
        seen.append(yaml.safe_load(config_path.read_text(encoding="utf-8")))
        time.sleep(budget_s)
        return 0

    original = loop._train_segment
    loop._train_segment = fake_train  # type: ignore[assignment]
    try:
        loop.run_coached({"ppo": {}}, plan, best_dir=tmp_path / "best", review=False)
    finally:
        loop._train_segment = original  # type: ignore[assignment]
    assert seen[0]["ppo"]["teacher_coef"] == 0.5
    assert seen[0]["name"] == "job_s0"
    assert (root / "override_applied_s0.yaml").exists()
    assert not (root / loop.OVERRIDE_FILE).exists()


def test_read_drives_flags_teacher_driven_rows(tmp_path: Path) -> None:
    from fly_driver.training.lab import read_drives

    seed_dir = tmp_path / "seed_0"
    seed_dir.mkdir()
    (seed_dir / "episodes.csv").write_text(
        "update,steps,distance_m,lap_complete,start_focused,teacher_driven\n"
        "1,100,50.0,False,False,False\n"
        "1,15000,11000.0,True,True,True\n",
        encoding="utf-8",
    )
    drives = read_drives(seed_dir)
    assert [drive.teacher_driven for drive in drives] == [False, True]


def test_the_digest_leaves_teacher_drives_out(tmp_path: Path) -> None:
    from fly_driver.coach.digest import build_digest, format_digest

    seed_dir = tmp_path / "run" / "seed_0"
    seed_dir.mkdir(parents=True)
    (seed_dir / "episodes.csv").write_text(
        "update,steps,distance_m,lap_complete,start_focused,teacher_driven,start_m\n"
        "1,100,50.0,False,False,False,0\n"
        "1,15000,11000.0,True,True,True,0\n",
        encoding="utf-8",
    )
    (seed_dir / "updates.csv").write_text("update,env_steps\n1,4096\n", encoding="utf-8")
    (seed_dir / "config.yaml").write_text("name: run\n", encoding="utf-8")
    digest = build_digest([tmp_path / "run"])
    assert len(digest.drives) == 1
    assert digest.seeds[0]["laps"] == 0
    assert digest.seeds[0]["teacher_drives"] == 1
    assert "Teacher drives" in format_digest(digest)
