"""Choosing starts and reporting steering (GH-104), on hand-made drives."""

from __future__ import annotations

import csv
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from fly_driver.analysis.steering_chatter import DriveTrace, trace_stats
from fly_driver.analysis.steering_report import (
    ConditionSpec,
    DriveResult,
    StartSet,
    exit_speed_rows,
    measure,
    print_summary,
    summary_rows,
    write_results,
)
from fly_driver.envs.segments import Segment
from fly_driver.interface import ControlVector

SEGMENTS = (
    Segment(0, "S0", False, None, 0.0, 100.0, math.inf),
    Segment(1, "T1", True, True, 100.0, 200.0, 50.0),
)


class _SeededEnv:
    """Starts at ``seed`` metres, or 0 for no seed, and ends after one step."""

    frame_rate_hz = 50.0

    def __init__(self) -> None:
        self.position = 0.0
        self.reset_seeds: list[int | None] = []

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.reset_seeds.append(seed)
        self.position = float(seed or 0)
        return np.zeros((2, 2, 3), dtype=np.uint8), {"arclength_m": self.position}

    def step(self, action: ControlVector) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        del action
        info = {"progress_m": 1.0, "speed_mps": 10.0, "arclength_m": self.position, "crashed": True}
        return np.zeros((2, 2, 3), dtype=np.uint8), 0.0, True, False, info


class _StraightAgent:
    def reset(self, seed: int | None = None) -> None:
        del seed

    def act(self, frame: np.ndarray) -> ControlVector:
        del frame
        return ControlVector(steer=0.5, throttle=1.0, brake=0.0)


def _result(condition: str, kind: str, steer: list[float], arclength: float = 50.0) -> DriveResult:
    size = len(steer)
    trace = DriveTrace(
        seed=0,
        start_m=arclength,
        distance_m=100.0,
        outcome="crashed",
        steer=np.array(steer),
        throttle=np.ones(size),
        brake=np.zeros(size),
        kingpin_rad=np.zeros(size),
        speed_mps=np.full(size, 10.0),
        arclength_m=np.full(size, arclength),
        lateral_m=np.zeros(size),
    )
    return DriveResult(condition, kind, trace, trace_stats(trace, 50.0, SEGMENTS), {})


def test_zone_seeds_are_read_from_real_resets_and_skip_starts_outside_the_zone() -> None:
    starts = StartSet.choose(_SeededEnv(), uniform=3, zone_count=2, zone=(1005.0, 1007.0))
    assert starts.uniform == (0, 1, 2)
    assert starts.zone == (1005, 1006)


def test_zone_seeds_never_overlap_the_uniform_seeds() -> None:
    starts = StartSet.choose(_SeededEnv(), uniform=20, zone_count=1, zone=(0.0, 1e9))
    assert not set(starts.uniform) & set(starts.zone)


def test_an_impossible_zone_fails_loudly() -> None:
    with pytest.raises(ValueError, match="found 0 of 1"):
        StartSet.choose(_SeededEnv(), uniform=0, zone_count=1, zone=(-2.0, -1.0))


def test_a_negative_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        StartSet.choose(_SeededEnv(), uniform=-1, zone_count=0, zone=(0.0, 1.0))


def test_measure_drives_the_grid_then_every_seed_on_the_right_env() -> None:
    grid_env, random_env = _SeededEnv(), _SeededEnv()
    results = measure(
        "noise_off",
        _StraightAgent(),
        grid_env=grid_env,
        random_env=random_env,
        starts=StartSet(uniform=(0, 1), zone=(1005,)),
        segments=SEGMENTS,
    )
    assert [result.start_kind for result in results] == ["grid", "uniform", "uniform", "zone"]
    assert grid_env.reset_seeds == [0]
    assert random_env.reset_seeds == [0, 1, 1005]
    assert {result.condition for result in results} == {"noise_off"}


def test_summary_pools_counts_over_total_time() -> None:
    calm = _result("noise_off", "uniform", [0.5] * 100)  # 2 s, no flips
    chatter = _result("noise_off", "zone", [1.0, -1.0] * 50)  # 2 s, 99 flips
    rows = summary_rows([calm, chatter])
    pooled = next(row for row in rows if row["start_kind"] == "all" and row["region"] == "straight")
    assert pooled["drives"] == 2
    assert pooled["flips"] == 99
    assert pooled["flips_per_s"] == pytest.approx(99.0 / 4.0)
    only_calm = next(
        row for row in rows if row["start_kind"] == "uniform" and row["region"] == "straight"
    )
    assert only_calm["flips"] == 0


def test_summary_reports_the_standard_error_of_mean_distance() -> None:
    near = _result("noise_off", "uniform", [0.0])
    far = _result("noise_off", "uniform", [0.0])
    near = replace(near, trace=replace(near.trace, distance_m=100.0))
    far = replace(far, trace=replace(far.trace, distance_m=300.0))
    pooled = next(row for row in summary_rows([near, far]) if row["start_kind"] == "all")
    # Sample std of 100 and 300 is sqrt(20000), over sqrt(2) gives 100.
    assert pooled["mean_distance_m"] == pytest.approx(200.0)
    assert pooled["distance_se_m"] == pytest.approx(100.0)


def test_one_drive_has_no_standard_error() -> None:
    row = summary_rows([_result("noise_off", "grid", [0.0])])[0]
    assert math.isnan(row["distance_se_m"])


def test_summary_keeps_conditions_apart() -> None:
    rows = summary_rows([_result("noise_off", "grid", [1.0]), _result("noise_on", "grid", [1.0])])
    assert {row["condition"] for row in rows} == {"noise_off", "noise_on"}
    assert all(row["drives"] == 1 for row in rows)


def test_write_results_writes_both_tables_and_a_trace_per_drive(tmp_path: Path) -> None:
    write_results([_result("noise_off", "grid", [1.0, -1.0])], tmp_path / "out")
    with (tmp_path / "out" / "episodes.csv").open(encoding="utf-8") as handle:
        episodes = list(csv.DictReader(handle))
    assert len(episodes) == 1
    assert episodes[0]["condition"] == "noise_off"
    assert float(episodes[0]["straight_flips_per_s"]) == pytest.approx(25.0)
    assert (tmp_path / "out" / "summary.csv").exists()
    trace = np.load(tmp_path / "out" / "noise_off_grid_0.npz")
    assert trace["steer"].tolist() == [1.0, -1.0]


def test_write_results_with_no_drives_writes_nothing(tmp_path: Path) -> None:
    write_results([], tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_print_summary_prints_a_row_per_group(capsys: pytest.CaptureFixture[str]) -> None:
    print_summary([_result("noise_off", "grid", [1.0, -1.0])])
    output = capsys.readouterr().out
    assert "flips/s" in output
    assert output.count("noise_off") == 6  # grid and all, three regions each


@pytest.mark.parametrize(
    ("label", "noise_on", "blend", "gain"),
    [
        ("noise_off", False, None, None),
        ("noise_on", True, None, None),
        ("smooth_0.2", False, 0.2, None),
        ("noise_off_smooth_0.5", False, 0.5, None),
        ("noise_on_smooth_0.5", True, 0.5, None),
        ("gain_0.5", False, None, 0.5),
        ("noise_off_gain_0.5", False, None, 0.5),
        ("gain_1", False, None, 1.0),
        ("gain_0.5_smooth_0.2", False, 0.2, 0.5),
    ],
)
def test_condition_labels_parse(
    label: str, noise_on: bool, blend: float | None, gain: float | None
) -> None:
    assert ConditionSpec.parse(label) == ConditionSpec(noise_on, blend, gain)


@pytest.mark.parametrize(
    "label",
    ["", "noise", "noise_onward", "smooth", "smooth_fast", "gain", "gain_x", "gain_0.5_noise_on"],
)
def test_an_unknown_condition_label_is_refused(label: str) -> None:
    with pytest.raises(ValueError, match="unknown condition"):
        ConditionSpec.parse(label)


def test_a_gain_with_noise_on_is_refused() -> None:
    with pytest.raises(ValueError, match="gain with noise on"):
        ConditionSpec.parse("noise_on_gain_0.5")


def test_exit_speed_rows_pool_by_condition_and_straight_in_kmh() -> None:
    first = _result("noise_off", "grid", [0.0])
    second = _result("noise_off", "zone", [0.0])
    other = _result("noise_on", "grid", [0.0])
    first.straight_exits["S0"] = [10.0]
    second.straight_exits["S0"] = [20.0, 30.0]
    other.straight_exits["S0"] = [5.0]
    rows = exit_speed_rows([first, second, other])
    assert rows == [
        {"condition": "noise_off", "straight": "S0", "exits": 3, "mean_exit_kmh": 72.0},
        {"condition": "noise_on", "straight": "S0", "exits": 1, "mean_exit_kmh": 18.0},
    ]


def test_write_results_records_vale_and_straight_exits(tmp_path: Path) -> None:
    result = _result("noise_off", "grid", [0.0, 0.0], arclength=5500.0)
    result = replace(result, trace=replace(result.trace, start_m=5000.0))  # a real run-up
    result.straight_exits["S0"] = [10.0]
    write_results([result], tmp_path)
    with (tmp_path / "episodes.csv").open(encoding="utf-8") as handle:
        episode = next(csv.DictReader(handle))
    assert episode["vale"] == "ended_in_zone"
    with (tmp_path / "straight_exits.csv").open(encoding="utf-8") as handle:
        exit_row = next(csv.DictReader(handle))
    assert exit_row["straight"] == "S0"
    assert float(exit_row["mean_exit_kmh"]) == pytest.approx(36.0)
