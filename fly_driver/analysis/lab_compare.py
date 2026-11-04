"""Did a lab recipe help? The same numbers for two lab folders, side by side (GH-103).

``train status`` looks at one folder and one moment. A test recipe in its own checkout needs
comparing with the recipe it changed, so this reads each folder's runs and prints what differs
between their configs, then the numbers that decide it.

Scores come from :mod:`fly_driver.training.lab`, so they are the supervisor's own: uniform-start
drives only. Vale survival counts every drive, because that is what a focused start practises.
Nothing here writes anywhere, so it is safe to run while a folder is training.
"""

from __future__ import annotations

import csv
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fly_driver.training.lab import (
    best_window,
    read_drives,
    recent_score,
    saved_checkpoints,
)

__all__ = [
    "RunSummary",
    "config_differences",
    "corner_survival",
    "format_comparison",
    "summarize_run",
]

#: Vale on the practice track, by where drives ended on 2026-11-03: 95% of the drives that
#: reached this stretch ended in it.
VALE_FROM_M = 5500.0
VALE_TO_M = 5600.0
LAP_M = 5886.8

#: Config keys that name a run rather than describe its recipe.
_NOT_THE_RECIPE = frozenset({"name", "seeds", "output_dir", "init_from"})


@dataclass(frozen=True)
class RunSummary:
    """One seed of one run.

    Args:
        run: The run's name, ``lab_<YYYYMMDD>_g<N>``.
        seed: The seed.
        env_steps: Env steps it has taken.
        drives: Training drives finished, of every kind.
        last_50_m: The supervisor's score now, or ``None`` before 50 uniform-start drives.
        best_window_m: The best 50-drive window with a saved checkpoint, or ``None``.
        longest_m: The longest single drive, of every kind.
        laps: Completed laps, of every kind.
        vale_arrivals: Times a drive reached Vale, not counting one that started inside it.
        vale_passes: How many of those got through.
        best_eval_m: The furthest deterministic drive from the grid, or ``None``.
        eval_laps: Laps completed by those deterministic drives.
        entropy: Policy entropy at the last update, or ``None``.
        mean_abs_steer: Mean ``|steer|`` over the last rollout, or ``None`` for an older run.
    """

    run: str
    seed: int
    env_steps: int
    drives: int
    last_50_m: float | None
    best_window_m: float | None
    longest_m: float
    laps: int
    vale_arrivals: int
    vale_passes: int
    best_eval_m: float | None
    eval_laps: int
    entropy: float | None
    mean_abs_steer: float | None


def config_differences(
    base: Mapping[str, Any], other: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Every recipe key that differs between two run configs, as ``{dotted.key: (base, other)}``.

    A key one config lacks reads as ``None``, so a setting only the test has still shows up.
    """

    def flatten(config: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
        flat: dict[str, Any] = {}
        for key, value in config.items():
            if not prefix and key in _NOT_THE_RECIPE:
                continue
            if isinstance(value, Mapping) and value:
                flat.update(flatten(value, f"{prefix}{key}."))
            else:
                flat[f"{prefix}{key}"] = value
        return flat

    flat_base, flat_other = flatten(base), flatten(other)
    return {
        key: (flat_base.get(key), flat_other.get(key))
        for key in sorted(flat_base.keys() | flat_other.keys())
        if flat_base.get(key) != flat_other.get(key)
    }


def corner_survival(
    drives: Iterable[tuple[float, float]],
    *,
    from_m: float = VALE_FROM_M,
    to_m: float = VALE_TO_M,
    lap_m: float = LAP_M,
) -> tuple[int, int]:
    """How often drives reached a stretch of track, and how often they got through it.

    A drive that goes round the lap and meets the stretch twice counts twice. A drive that
    started inside the stretch has not arrived at it, so that pass does not count.

    Args:
        drives: ``(start_m, distance_m)`` per drive.
        from_m: Where the stretch starts.
        to_m: Where it ends.
        lap_m: The lap's length.

    Returns:
        ``(arrivals, passes)``.
    """
    arrivals, passes = 0, 0
    for start_m, distance_m in drives:
        end_m = start_m + distance_m
        gate_m = from_m if start_m < from_m else from_m + lap_m
        while gate_m <= end_m:
            arrivals += 1
            passes += end_m >= gate_m + (to_m - from_m)
            gate_m += lap_m
    return arrivals, passes


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, str | None], key: str) -> float | None:
    """A float from a CSV row, or ``None`` if the column is missing, empty or torn."""
    try:
        return float(row[key])  # type: ignore[arg-type]
    except (KeyError, TypeError, ValueError):
        return None


def summarize_run(run_dir: Path) -> RunSummary:
    """Read one ``runs/<run>/seed_<k>`` directory.

    Args:
        run_dir: The seed's directory.

    Returns:
        Its numbers. A run with no drives yet gives zeros and ``None``.
    """
    drives = read_drives(run_dir)
    updates = _read_rows(run_dir / "updates.csv")
    window = best_window(drives, saved_checkpoints(run_dir))

    positions = []
    for row in _read_rows(run_dir / "episodes.csv"):
        start_m, distance_m = _number(row, "start_m"), _number(row, "distance_m")
        if start_m is not None and distance_m is not None:
            positions.append((start_m, distance_m))
    arrivals, passes = corner_survival(positions)

    eval_rows = [
        row for path in sorted(run_dir.glob("eval/step_*/episodes.csv")) for row in _read_rows(path)
    ]
    eval_distances = [
        distance for row in eval_rows if (distance := _number(row, "progress_m")) is not None
    ]
    last_update = updates[-1] if updates else {}
    return RunSummary(
        run=run_dir.parent.name,
        seed=int(run_dir.name.removeprefix("seed_")),
        env_steps=int(_number(last_update, "env_steps") or 0),
        drives=len(drives),
        last_50_m=recent_score(drives),
        best_window_m=None if window is None else window.score,
        longest_m=max((drive.distance_m for drive in drives), default=0.0),
        laps=sum(drive.lap_complete for drive in drives),
        vale_arrivals=arrivals,
        vale_passes=passes,
        best_eval_m=max(eval_distances, default=None),
        eval_laps=sum(row.get("lap_complete") == "True" for row in eval_rows),
        entropy=_number(last_update, "entropy"),
        mean_abs_steer=_number(last_update, "mean_abs_steer"),
    )


def _run_dirs(folder: Path, run_prefix: str) -> list[Path]:
    return sorted(path for path in (folder / "runs").glob(f"{run_prefix}*/seed_*") if path.is_dir())


def _latest_config(run_dirs: list[Path]) -> dict[str, Any]:
    paths = [run_dir / "config.yaml" for run_dir in run_dirs]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return {}
    newest = max(existing, key=lambda path: path.stat().st_mtime)
    return yaml.safe_load(newest.read_text(encoding="utf-8")) or {}


def _shown(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def _folder_lines(folder: Path, runs: list[RunSummary]) -> list[str]:
    arrivals = sum(run.vale_arrivals for run in runs)
    passes = sum(run.vale_passes for run in runs)
    survival = f"{100.0 * passes / arrivals:.1f}%" if arrivals else "-"
    best = [run.best_window_m for run in runs if run.best_window_m is not None]
    evals = [run.best_eval_m for run in runs if run.best_eval_m is not None]
    entropies = [run.entropy for run in runs if run.entropy is not None]
    steers = [run.mean_abs_steer for run in runs if run.mean_abs_steer is not None]
    lines = [
        f"{folder}",
        f"  env steps           {sum(run.env_steps for run in runs):,}",
        f"  drives              {sum(run.drives for run in runs):,}",
        f"  laps                {sum(run.laps for run in runs)}",
        f"  best 50-drive score {_shown(max(best, default=None))} m  (uniform-start drives)",
        f"  longest drive       {_shown(max((run.longest_m for run in runs), default=None))} m",
        f"  through Vale        {survival}  ({passes:,} of {arrivals:,} arrivals)",
        f"  best grid eval      {_shown(max(evals, default=None))} m, "
        f"{sum(run.eval_laps for run in runs)} laps (no noise)",
        f"  entropy, |steer|    "
        f"{_shown(statistics.mean(entropies) if entropies else None, 2)}, "
        f"{_shown(statistics.mean(steers) if steers else None, 2)}  (each run's last update)",
        f"  {'run':<24}{'steps':>10}{'drives':>8}{'last 50':>9}{'best 50':>9}{'longest':>9}"
        f"{'Vale':>12}{'laps':>6}",
    ]
    for run in runs:
        vale = f"{run.vale_passes}/{run.vale_arrivals}"
        name = f"{run.run} s{run.seed}"
        lines.append(
            f"  {name:<24}{run.env_steps:>10,}{run.drives:>8,}{_shown(run.last_50_m):>9}"
            f"{_shown(run.best_window_m):>9}{_shown(run.longest_m):>9}{vale:>12}{run.laps:>6}"
        )
    return lines


def format_comparison(base: Path, other: Path, *, run_prefix: str = "lab_") -> str:
    """The comparison as text: what differs between the recipes, then each folder's numbers.

    Args:
        base: The folder running the recipe being compared against.
        other: The folder running the test recipe.
        run_prefix: Which runs count. A date, such as ``lab_20261103``, picks one day's.

    Returns:
        Lines of text, for printing.

    Raises:
        FileNotFoundError: If a folder has no runs with that prefix.
    """
    run_dirs = {folder: _run_dirs(folder, run_prefix) for folder in (base, other)}
    for folder, found in run_dirs.items():
        if not found:
            raise FileNotFoundError(f"no {folder / 'runs' / run_prefix}*/seed_* runs")

    differences = config_differences(
        _latest_config(run_dirs[base]), _latest_config(run_dirs[other])
    )
    lines = ["What differs between the recipes, as run (base -> test):"]
    lines += [f"  {key}: {old} -> {new}" for key, (old, new) in differences.items()]
    if not differences:
        lines.append("  nothing")
    for folder in (base, other):
        lines.append("")
        lines += _folder_lines(folder, [summarize_run(run_dir) for run_dir in run_dirs[folder]])
    scored_note = "Scores count uniform-start drives only; Vale and laps count every drive."
    return "\n".join([*lines, "", scored_note])
