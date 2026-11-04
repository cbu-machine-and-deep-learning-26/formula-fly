"""The lab comparison (GH-103): hand-checked counts, and what a real folder's files give it."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

pytest.importorskip("yaml")

from fly_driver.analysis.lab_compare import (  # noqa: E402
    LAP_M,
    VALE_FROM_M,
    VALE_TO_M,
    config_differences,
    corner_survival,
    format_comparison,
    summarize_run,
)


class TestCornerSurvival:
    """Vale survival is the number GH-103 is judged on, so every case is worked out by hand."""

    def test_short_of_the_corner_is_no_arrival(self) -> None:
        assert corner_survival([(5000.0, 400.0)]) == (0, 0)

    def test_ending_in_the_corner_is_an_arrival_and_no_pass(self) -> None:
        assert corner_survival([(5000.0, 515.0)]) == (1, 0)

    def test_leaving_the_corner_is_a_pass(self) -> None:
        assert corner_survival([(5000.0, 900.0)]) == (1, 1)

    def test_starting_inside_the_corner_has_not_arrived_at_it(self) -> None:
        """A start past the corner's entry must not count as surviving it."""
        assert corner_survival([(5550.0, 200.0)]) == (0, 0)

    def test_a_start_after_the_corner_meets_it_on_the_next_lap(self) -> None:
        reaches_it = LAP_M - 5700.0 + VALE_FROM_M + 10.0
        assert corner_survival([(5700.0, reaches_it)]) == (1, 0)
        assert corner_survival([(5700.0, reaches_it - 20.0)]) == (0, 0)

    def test_going_round_again_meets_it_twice(self) -> None:
        once_round_and_through = LAP_M + (VALE_TO_M - VALE_FROM_M) + 500.0
        assert corner_survival([(5000.0, once_round_and_through)]) == (2, 2)

    def test_the_counts_add_up_over_many_drives(self) -> None:
        assert corner_survival([(5000.0, 515.0), (5000.0, 900.0), (100.0, 50.0)]) == (2, 1)


class TestConfigDifferences:
    """This is what marks the test's changes, so a change must not be able to hide."""

    BASE = {
        "name": "lab_g1",
        "seeds": [0],
        "ppo": {"entropy_coef": 0.005, "gamma": 0.997},
        "env": {"params": {"random_start": True}},
    }

    def test_the_vale_recipe_shows_both_its_changes_and_nothing_else(self) -> None:
        test = {
            "name": "lab_g9",
            "seeds": [1, 2],
            "ppo": {"entropy_coef": 0.002, "gamma": 0.997},
            "env": {"params": {"random_start": True, "start_focus": {"share": 0.5}}},
        }
        assert config_differences(self.BASE, test) == {
            "env.params.start_focus.share": (None, 0.5),
            "ppo.entropy_coef": (0.005, 0.002),
        }

    def test_identical_recipes_differ_in_nothing_but_their_names(self) -> None:
        assert config_differences(self.BASE, {**self.BASE, "name": "lab_g2"}) == {}

    def test_a_key_removed_from_the_test_shows_too(self) -> None:
        test = {"ppo": {"entropy_coef": 0.005}, "env": {"params": {"random_start": True}}}
        assert config_differences(self.BASE, test) == {"ppo.gamma": (0.997, None)}


def _write_run(
    folder: Path, run: str, seed: int, entropy: float, drives: list[tuple[float, float, bool]]
) -> Path:
    """A run as ``train.py`` lays it out, from ``(start_m, distance_m, is_focused)`` drives."""
    run_dir = folder / "runs" / run / f"seed_{seed}"
    (run_dir / "eval" / "step_000002048").mkdir(parents=True)
    rows = ["seed,episode,update,env_steps,steps,lap_complete,start_m,distance_m,start_focused"]
    rows += [
        f"{seed},{index},{index // 5 + 1},{index},100,False,{start},{distance},{is_focused}"
        for index, (start, distance, is_focused) in enumerate(drives)
    ]
    (run_dir / "episodes.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (run_dir / "updates.csv").write_text(
        f"update,env_steps,entropy,mean_abs_steer\n1,2048,9.9,0.1\n2,4096,{entropy},0.85\n",
        encoding="utf-8",
    )
    (run_dir / "eval" / "step_000002048" / "episodes.csv").write_text(
        "progress_m,lap_complete\n1183.0,False\n", encoding="utf-8"
    )
    for stage in ("policy", "brain"):
        with zipfile.ZipFile(run_dir / f"{stage}_update_00001.pt", "w") as archive:
            archive.writestr("update", stage)
    (run_dir / "config.yaml").write_text(
        f"name: {run}\nseeds: [{seed}]\nppo:\n  entropy_coef: {entropy}\n", encoding="utf-8"
    )
    return run_dir


def test_a_run_reads_the_supervisors_score_and_counts_every_drive_for_vale(tmp_path: Path) -> None:
    uniform = [(100.0, 1000.0, False)] * 50
    focused_through_vale = [(5000.0, 900.0, True)] * 50
    run_dir = _write_run(tmp_path, "lab_20261104_g1", 2, 2.1, uniform + focused_through_vale)
    summary = summarize_run(run_dir)
    assert summary.seed == 2
    assert summary.drives == 100
    assert summary.last_50_m == 1000.0, "the score is the uniform-start drives'"
    assert summary.best_window_m == 1000.0
    assert (summary.vale_arrivals, summary.vale_passes) == (50, 50)
    assert summary.env_steps == 4096
    assert summary.entropy == 2.1, "the last update's, not the first's"
    assert summary.mean_abs_steer == 0.85
    assert summary.best_eval_m == 1183.0


def test_a_run_from_before_the_new_columns_still_reads(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "lab_20261103_g1" / "seed_0"
    run_dir.mkdir(parents=True)
    rows = ["update,steps,distance_m,lap_complete"] + ["1,100,500.0,False"] * 60
    (run_dir / "episodes.csv").write_text("\n".join(rows) + "\n12,", encoding="utf-8")
    (run_dir / "updates.csv").write_text("update,env_steps,entropy\n1,2048,2.0\n", encoding="utf-8")
    summary = summarize_run(run_dir)
    assert summary.drives == 60
    assert summary.last_50_m == 500.0
    assert summary.mean_abs_steer is None
    assert (summary.vale_arrivals, summary.best_eval_m) == (0, None)


def test_a_run_with_nothing_yet_does_not_crash(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "lab_20261104_g1" / "seed_0"
    run_dir.mkdir(parents=True)
    summary = summarize_run(run_dir)
    assert (summary.drives, summary.env_steps, summary.last_50_m) == (0, 0, None)


def test_the_comparison_marks_the_difference_and_prints_both_folders(tmp_path: Path) -> None:
    base, test = tmp_path / "formula-fly", tmp_path / "formula-fly-vale"
    _write_run(base, "lab_20261104_g1", 0, 0.005, [(5000.0, 515.0, False)] * 60)
    _write_run(test, "lab_20261104_g1", 0, 0.002, [(5000.0, 900.0, False)] * 60)
    text = format_comparison(base, test)
    assert "ppo.entropy_coef: 0.005 -> 0.002" in text
    assert "through Vale        0.0%  (0 of 60 arrivals)" in text
    assert "through Vale        100.0%  (60 of 60 arrivals)" in text
    assert str(base) in text
    assert str(test) in text


def test_the_run_prefix_picks_the_runs(tmp_path: Path) -> None:
    base, test = tmp_path / "base", tmp_path / "test"
    _write_run(base, "lab_20261103_g1", 0, 0.005, [(0.0, 1.0, False)])
    _write_run(base, "lab_20261104_g1", 0, 0.005, [(0.0, 1.0, False)])
    _write_run(test, "lab_20261104_g1", 0, 0.005, [(0.0, 1.0, False)])
    text = format_comparison(base, test, run_prefix="lab_20261104")
    assert "lab_20261103_g1" not in text


def test_a_folder_with_no_runs_is_an_error_saying_which(tmp_path: Path) -> None:
    base, test = tmp_path / "base", tmp_path / "test"
    _write_run(base, "lab_20261104_g1", 0, 0.005, [(0.0, 1.0, False)])
    with pytest.raises(FileNotFoundError, match="test"):
        format_comparison(base, test)
