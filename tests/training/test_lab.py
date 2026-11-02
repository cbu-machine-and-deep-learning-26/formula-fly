"""The lab PC supervisor's decisions: best window, checkpoint, which seed restarts (GH-27)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from fly_driver.training.config import TrainConfig
from fly_driver.training.lab import (
    Drive,
    Process,
    best_window,
    check_seeds,
    nearest_checkpoint,
    one_per_trainer,
    promote_best,
    read_drives,
    seed_to_restart,
    trainer_processes,
)

CONFIGS = Path(__file__).parents[2] / "fly_driver" / "configs"


def _drives(distances: list[float], drives_per_update: int = 5) -> list[Drive]:
    return [
        Drive(index // drives_per_update + 1, 1000, distance, False)
        for index, distance in enumerate(distances)
    ]


def test_best_window_finds_the_peak_and_the_checkpoint_at_or_before_its_end() -> None:
    # 50 drives at 100 m, 50 at 500 m, 50 at 100 m; 5 drives per update, so the peak's drives
    # are rows 50-99 and end in update 20.
    drives = _drives([100.0] * 50 + [500.0] * 50 + [100.0] * 50)
    window = best_window(drives, [10, 20, 30])
    assert window is not None
    assert window.score == 500.0
    assert (window.last_update, window.checkpoint) == (20, 20)
    assert nearest_checkpoint(29, [10, 20, 30]) == 20


def test_a_window_ending_before_the_first_checkpoint_gives_way_to_one_that_was_saved() -> None:
    # The best window is the first, ending in update 10, but the first checkpoint is at 20.
    drives = _drives([900.0] * 50 + [200.0] * 100)
    assert nearest_checkpoint(10, [20, 30]) is None
    window = best_window(drives, [20, 30])
    assert window is not None
    assert window.checkpoint >= 20
    assert window.score < 900.0
    assert best_window(_drives([900.0] * 50), [20, 30]) is None


def test_best_window_needs_fifty_drives() -> None:
    assert best_window(_drives([900.0] * 49), [1]) is None


@pytest.mark.parametrize(
    "scores, last_updates, expected",
    [
        pytest.param({0: 100.0, 1: 800.0}, {0: 150, 1: 900}, None, id="worst-still-warming-up"),
        pytest.param({0: 700.0, 1: 800.0}, {0: 900, 1: 900}, None, id="worst-at-70-percent"),
        pytest.param({0: 699.0, 1: 300.0}, {0: 900, 1: 151}, 1, id="worst-below-70-percent"),
        pytest.param({}, {}, None, id="no-seed-has-50-drives"),
    ],
)
def test_which_seed_restarts(
    scores: dict[int, float], last_updates: dict[int, int], expected: int | None
) -> None:
    assert seed_to_restart(scores, last_updates, best_score=1000.0) == expected


def test_only_the_worst_seed_is_considered() -> None:
    # Seed 1 is low and warmed up, but the worst seed is warming up, so nobody restarts.
    assert seed_to_restart({0: 100.0, 1: 300.0}, {0: 10, 1: 900}, best_score=1000.0) is None


def test_more_than_three_seeds_is_refused() -> None:
    check_seeds([0, 1, 2])
    with pytest.raises(ValueError, match="limit is 3"):
        check_seeds([0, 1, 2, 3])


def test_trainer_processes_names_train_py_but_not_the_supervisor_or_an_editor() -> None:
    trainer = Process(1, r"D:\x\.venv-flyvis\Scripts\python.exe D:\x\scripts\train.py --seed 0")
    processes = [
        trainer,
        Process(2, r".venv-flyvis\Scripts\python.exe scripts\lab_train.py start"),
        Process(3, "notepad.exe scripts\\train.py"),
        Process(4, 'powershell -Command "python scripts\\train.py --seed 0"'),
    ]
    assert trainer_processes(processes) == [trainer]


def test_a_venv_launcher_and_its_child_count_as_one_trainer() -> None:
    arguments = r"D:\x\scripts\train.py --config D:\x\logs\g1.yaml --seed 0"
    launcher = Process(1, rf"D:\x\.venv-flyvis\Scripts\python.exe {arguments}")
    child = Process(2, rf'"C:\Program Files\Python311\python.exe" {arguments}')
    other_seed = Process(3, rf"D:\x\.venv-flyvis\Scripts\python.exe {arguments[:-1]}1")
    assert one_per_trainer([launcher, child, other_seed]) == [launcher, other_seed]


def _write_run(run_dir: Path, distances: list[float], checkpoints: list[int]) -> None:
    run_dir.mkdir(parents=True)
    rows = ["update,steps,distance_m,lap_complete"]
    rows += [f"{d.update},{d.steps},{d.distance_m},False" for d in _drives(distances)]
    # A trainer killed mid-write leaves half a row, which must not count as a drive.
    (run_dir / "episodes.csv").write_text("\n".join(rows) + "\n12,", encoding="utf-8")
    for update in checkpoints:
        for stage in ("policy", "brain"):
            with zipfile.ZipFile(run_dir / f"{stage}_update_{update:05d}.pt", "w") as archive:
                archive.writestr("update", f"{stage} {update}")


def test_promote_best_copies_the_pair_only_when_the_record_is_beaten(tmp_path: Path) -> None:
    runs_dir, best_dir = tmp_path / "runs", tmp_path / "best"
    best_dir.mkdir()
    (best_dir / "policy_best.pt").write_text("old")
    (best_dir / "brain_best.pt").write_text("old")
    (best_dir / "best.json").write_text(json.dumps({"score": 400.0}))
    run_dir = runs_dir / "lab_20261102_g1" / "seed_2"
    _write_run(run_dir, [500.0] * 100, [10, 20])
    # Torn: this is the newest checkpoint, and it must not be chosen.
    (run_dir / "policy_update_00030.pt").write_bytes(b"PK torn")
    (run_dir / "brain_update_00030.pt").write_bytes(b"PK torn")

    assert len(read_drives(run_dir)) == 100
    assert promote_best([("lab_20261102_g1", 2)], runs_dir, best_dir) is not None
    record = json.loads((best_dir / "best.json").read_text())
    assert record == {"score": 500.0, "run": "lab_20261102_g1", "seed": 2, "update": 10}
    with zipfile.ZipFile(best_dir / "brain_best.pt") as archive:
        assert archive.read("update") == b"brain 10"

    assert promote_best([("lab_20261102_g1", 2)], runs_dir, best_dir) is None


def test_train_lab_yaml_is_the_recipe_and_warm_starts_from_weights_best() -> None:
    pytest.importorskip("yaml")
    config = TrainConfig.from_yaml(CONFIGS / "train_lab.yaml")
    assert config.seeds == (0, 1, 2)
    assert (config.ppo.learning_rate, config.ppo.target_kl) == (5.0e-5, 0.0075)
    assert config.init_from == "weights/best/policy_best.pt"
