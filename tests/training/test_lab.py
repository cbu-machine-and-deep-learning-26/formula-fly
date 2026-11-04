"""The lab PC supervisor's decisions: best window, checkpoint, which seed restarts (GH-27)."""

from __future__ import annotations

import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from fly_driver.training.config import TrainConfig
from fly_driver.training.lab import (
    WINDOW_DRIVES,
    Drive,
    Process,
    best_window,
    check_seeds,
    clear_record,
    nearest_checkpoint,
    one_per_trainer,
    promote_best,
    read_drives,
    recent_score,
    scored_drives,
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


def test_the_restart_fraction_sets_the_bar() -> None:
    """A real collapse still restarts at 0.3, and an ordinary dip below the lucky best doesn't."""
    assert seed_to_restart({0: 500.0}, {0: 900}, best_score=1359.0) == 0, "the default bar is 951 m"
    assert seed_to_restart({0: 500.0}, {0: 900}, 1359.0, restart_fraction=0.3) is None
    assert seed_to_restart({0: 200.0}, {0: 900}, 1359.0, restart_fraction=0.3) == 0
    assert seed_to_restart({0: 1.0}, {0: 900}, 1359.0, restart_fraction=0.0) is None


def test_clearing_the_record_keeps_the_weights_and_lets_a_new_car_earn_its_own(
    tmp_path: Path,
) -> None:
    runs_dir, best_dir = tmp_path / "runs", tmp_path / "best"
    best_dir.mkdir()
    (best_dir / "policy_best.pt").write_text("rescaled")
    (best_dir / "brain_best.pt").write_text("rescaled")
    (best_dir / "best.json").write_text(json.dumps({"score": 1359.0}))

    record = clear_record(best_dir, "1,359 m checkpoint, rescaled for the stiff car")
    assert json.loads((best_dir / "best.json").read_text()) == record
    assert record["score"] == 0.0
    assert (best_dir / "policy_best.pt").read_text() == "rescaled"
    assert seed_to_restart({0: 1.0}, {0: 900}, record["score"]) is None, "nothing to fall short of"

    run_dir = runs_dir / "lab_20261105_g1" / "seed_0"
    _write_run(run_dir, [300.0] * 60, [1])
    assert promote_best([("lab_20261105_g1", 0)], runs_dir, best_dir) is not None
    assert json.loads((best_dir / "best.json").read_text())["score"] == 300.0


def test_clearing_a_folder_with_no_checkpoint_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="missing"):
        clear_record(tmp_path, "nothing here")


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


def _alternating(uniform_m: float, focused_m: float, count: int) -> list[Drive]:
    """``count`` of each kind in turn, five rows per update, as a focused run writes them."""
    rows = []
    for index in range(count * 2):
        is_focused = index % 2 == 1
        distance = focused_m if is_focused else uniform_m
        rows.append(Drive(index // 5 + 1, 1000, distance, False, is_focused))
    return rows


class TestFocusedStarts:
    """A focused drive is short by design, and must not look like a collapse (GH-103)."""

    def test_scores_read_only_the_uniform_drives(self) -> None:
        drives = _alternating(uniform_m=1200.0, focused_m=300.0, count=60)
        assert len(scored_drives(drives)) == 60
        assert recent_score(drives) == 1200.0
        window = best_window(drives, [1])
        assert window is not None
        assert window.score == 1200.0

    def test_a_focused_run_at_the_old_pace_is_not_restarted(self) -> None:
        """The trap: scored naively this run averages 750 m and would restart every cycle."""
        drives = _alternating(uniform_m=1200.0, focused_m=300.0, count=60)
        last_50 = drives[-WINDOW_DRIVES:]
        naive = sum(drive.distance_m for drive in last_50) / WINDOW_DRIVES
        assert naive == pytest.approx(750.0)
        assert seed_to_restart({0: naive}, {0: 900}, best_score=1359.0) == 0, "the trap is real"

        score = recent_score(drives)
        assert score is not None
        assert seed_to_restart({0: score}, {0: 900}, best_score=1359.0) is None

    def test_a_uniform_collapse_is_still_restarted_under_a_focus(self) -> None:
        drives = _alternating(uniform_m=400.0, focused_m=300.0, count=60)
        score = recent_score(drives)
        assert score == 400.0
        assert seed_to_restart({0: score}, {0: 900}, best_score=1359.0) == 0

    def test_focused_drives_cannot_set_a_record(self) -> None:
        drives = _alternating(uniform_m=500.0, focused_m=4000.0, count=60)
        window = best_window(drives, [1])
        assert window is not None
        assert window.score == 500.0

    def test_a_score_waits_for_fifty_uniform_drives_however_many_focused_there_are(self) -> None:
        drives = _alternating(uniform_m=500.0, focused_m=300.0, count=49)
        assert len(drives) == 98
        assert recent_score(drives) is None
        assert best_window(drives, [1]) is None

    def test_a_lap_from_a_focused_start_still_counts_as_a_lap(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "seed_0"
        run_dir.mkdir()
        (run_dir / "episodes.csv").write_text(
            "update,steps,distance_m,lap_complete,start_focused\n"
            "1,1000,300.0,False,True\n"
            "1,9000,7000.0,True,True\n"
            "2,1000,900.0,False,False\n",
            encoding="utf-8",
        )
        drives = read_drives(run_dir)
        assert [drive.start_focused for drive in drives] == [True, True, False]
        assert sum(drive.lap_complete for drive in drives) == 1
        assert [drive.distance_m for drive in scored_drives(drives)] == [900.0]

    def test_a_run_from_before_the_column_reads_exactly_as_it_did(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "seed_0"
        _write_run(run_dir, [500.0] * 60, [1])
        drives = read_drives(run_dir)
        assert len(drives) == 60
        assert not any(drive.start_focused for drive in drives)
        assert recent_score(drives) == 500.0


def test_train_lab_yaml_is_the_recipe_and_warm_starts_from_weights_best() -> None:
    pytest.importorskip("yaml")
    config = TrainConfig.from_yaml(CONFIGS / "train_lab.yaml")
    assert config.seeds == (0, 1, 2)
    assert (config.ppo.learning_rate, config.ppo.target_kl) == (5.0e-5, 0.0075)
    assert config.init_from == "weights/best/policy_best.pt"


def test_the_vale_recipe_is_train_lab_plus_its_two_changes_and_nothing_else() -> None:
    """GH-103's comparison against the running recipe only means something if this holds."""
    pytest.importorskip("yaml")
    recipe = TrainConfig.from_yaml(CONFIGS / "train_lab.yaml")
    vale = TrainConfig.from_yaml(CONFIGS / "train_lab_vale.yaml")

    assert vale.env.params["start_focus"] == {
        "share": 0.5,
        "zones": [[470.0, 620.0], [1290.0, 1700.0], [4800.0, 5200.0]],
    }
    assert vale.ppo.entropy_coef == 0.002
    assert recipe.ppo.entropy_coef == 0.005

    params = {key: value for key, value in vale.env.params.items() if key != "start_focus"}
    undone = replace(
        vale,
        name=recipe.name,
        env=replace(vale.env, params=params),
        ppo=replace(vale.ppo, entropy_coef=recipe.ppo.entropy_coef),
    )
    assert undone == recipe
