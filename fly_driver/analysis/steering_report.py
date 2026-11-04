"""Running a driver from a fixed set of starts and reporting its steering (GH-104).

The starts are reset seeds, so they are the env's own: a given seed always gives the same start
on a random-start env. Three kinds of start answer different questions:

- ``grid``: the start every evaluation uses.
- ``uniform``: round the lap, the way training starts.
- ``zone``: the Stowe and Vale approach (#103), since few uniform drives ever get there.
"""

from __future__ import annotations

import csv
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fly_driver.analysis.steering_chatter import (
    DriveTrace,
    SteeringStats,
    combine_stats,
    drive,
    trace_stats,
)
from fly_driver.envs.segments import Segment

__all__ = ["DriveResult", "StartSet", "measure", "print_summary", "summary_rows", "write_results"]

#: Where zone seeds begin, so they never overlap the uniform seeds counted up from 0.
_ZONE_FIRST_SEED = 1000

#: Give up looking for zone seeds after this many tries, rather than loop on an empty zone.
_ZONE_MAX_TRIES = 2000

REGIONS = ("all", "straight", "corner")


@dataclass(frozen=True)
class StartSet:
    """The reset seeds a measurement drives from.

    Args:
        uniform: Seeds counted up from 0, which start anywhere round the lap.
        zone: Seeds whose start falls inside the chosen stretch of track.
    """

    uniform: tuple[int, ...]
    zone: tuple[int, ...]

    @classmethod
    def choose(
        cls, env: Any, *, uniform: int, zone_count: int, zone: tuple[float, float]
    ) -> StartSet:
        """Pick seeds, reading each zone start from a real reset of a random-start env.

        Args:
            env: A random-start env.
            uniform: How many uniform seeds.
            zone_count: How many zone seeds.
            zone: ``(start_m, end_m)`` the zone starts must fall in.

        Raises:
            ValueError: If a count is negative, or the zone yields too few starts.
        """
        if uniform < 0 or zone_count < 0:
            raise ValueError(f"counts must be non-negative, got {uniform} and {zone_count}")
        low, high = zone
        found: list[int] = []
        for seed in range(_ZONE_FIRST_SEED, _ZONE_FIRST_SEED + _ZONE_MAX_TRIES):
            if len(found) == zone_count:
                break
            _, info = env.reset(seed=seed)
            if low <= float(info["arclength_m"]) <= high:
                found.append(seed)
        if len(found) < zone_count:
            raise ValueError(f"found {len(found)} of {zone_count} starts in {low}-{high} m")
        return cls(uniform=tuple(range(uniform)), zone=tuple(found))


@dataclass(frozen=True)
class DriveResult:
    """One measured drive.

    Args:
        condition: What was being tested, e.g. ``noise_off``.
        start_kind: ``grid``, ``uniform`` or ``zone``.
        trace: Every frame of the drive.
        stats: Its steering, keyed by region in :data:`REGIONS`.
    """

    condition: str
    start_kind: str
    trace: DriveTrace
    stats: dict[str, SteeringStats]


def measure(
    condition: str,
    driver: Any,
    *,
    grid_env: Any,
    random_env: Any,
    starts: StartSet,
    segments: Sequence[Segment],
    max_steps: int | None = None,
) -> list[DriveResult]:
    """Drive one condition from the grid and from every start in ``starts``.

    Args:
        condition: Label for the rows.
        driver: Anything with ``reset(seed=...)`` and ``act(frame)``.
        grid_env: An env that starts on the grid.
        random_env: An env that starts from its reset seed.
        starts: The seeds.
        segments: The whole lap, to split straights from corners.
        max_steps: Frame limit per drive.

    Returns:
        One result per drive, grid first.
    """
    plan: list[tuple[str, Any, int]] = [("grid", grid_env, 0)]
    plan += [("uniform", random_env, seed) for seed in starts.uniform]
    plan += [("zone", random_env, seed) for seed in starts.zone]
    results = []
    for number, (start_kind, env, seed) in enumerate(plan, start=1):
        began = time.perf_counter()
        trace = drive(env, driver, seed=seed, max_steps=max_steps)
        stats = trace_stats(trace, float(env.frame_rate_hz), segments)
        results.append(DriveResult(condition, start_kind, trace, stats))
        print(
            f"{condition} {number}/{len(plan)} {start_kind} seed {seed}: "
            f"{trace.start_m:.0f} m, drove {trace.distance_m:.0f} m, {trace.outcome}, "
            f"{stats['all'].flips_per_s:.1f} flips/s, "
            f"|steer| {stats['all'].mean_abs_steer:.2f}, "
            f"{trace.steer.size} frames in {time.perf_counter() - began:.0f} s",
            flush=True,
        )
    return results


def _stat_columns(stats: SteeringStats) -> dict[str, float | int]:
    return {
        "frames": stats.frames,
        "seconds": round(stats.seconds, 3),
        "flips": stats.flips,
        "flips_per_s": round(stats.flips_per_s, 3),
        "mean_abs_steer": round(stats.mean_abs_steer, 4),
        "full_lock_share": round(stats.full_lock_share, 4),
        "mean_step": round(stats.mean_step, 4),
    }


def summary_rows(results: Sequence[DriveResult]) -> list[dict[str, Any]]:
    """Pool drives by condition, start kind and region, with all start kinds pooled as ``all``.

    Flips are pooled as counts over total time, so a long calm drive outweighs a short bad one
    in proportion to how long it ran.

    Args:
        results: The measured drives.

    Returns:
        One row per ``(condition, start_kind, region)``.
    """
    rows: list[dict[str, Any]] = []
    conditions = list(dict.fromkeys(result.condition for result in results))
    for condition in conditions:
        chosen = [result for result in results if result.condition == condition]
        kinds = list(dict.fromkeys(result.start_kind for result in chosen))
        for kind in [*kinds, "all"]:
            group = [item for item in chosen if kind in ("all", item.start_kind)]
            distances = [item.trace.distance_m for item in group]
            for region in REGIONS:
                pooled = combine_stats(item.stats[region] for item in group)
                rows.append(
                    {
                        "condition": condition,
                        "start_kind": kind,
                        "region": region,
                        "drives": len(group),
                        "mean_distance_m": round(float(np.mean(distances)), 1),
                        **_stat_columns(pooled),
                    }
                )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_results(results: Sequence[DriveResult], output_dir: Path) -> None:
    """Write ``episodes.csv``, ``summary.csv`` and a trace per drive.

    Args:
        results: The measured drives. Writes nothing when empty.
        output_dir: Created if missing.
    """
    if not results:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for result in results:
        trace = result.trace
        row: dict[str, Any] = {
            "condition": result.condition,
            "start_kind": result.start_kind,
            "seed": trace.seed,
            "start_m": round(trace.start_m, 1),
            "end_m": round(trace.end_m, 1),
            "distance_m": round(trace.distance_m, 1),
            "outcome": trace.outcome,
            "top_speed_kmh": round(float(np.max(trace.speed_mps, initial=0.0)) * 3.6, 1),
        }
        for region in REGIONS:
            stats = result.stats[region]
            row[f"{region}_seconds"] = round(stats.seconds, 3)
            row[f"{region}_flips_per_s"] = round(stats.flips_per_s, 3)
            row[f"{region}_mean_abs_steer"] = round(stats.mean_abs_steer, 4)
            row[f"{region}_full_lock_share"] = round(stats.full_lock_share, 4)
        episodes.append(row)
        np.savez_compressed(
            output_dir / f"{result.condition}_{result.start_kind}_{trace.seed}.npz",
            steer=trace.steer,
            throttle=trace.throttle,
            brake=trace.brake,
            kingpin_rad=trace.kingpin_rad,
            speed_mps=trace.speed_mps,
            arclength_m=trace.arclength_m,
            lateral_m=trace.lateral_m,
        )
    _write_csv(output_dir / "episodes.csv", episodes)
    _write_csv(output_dir / "summary.csv", summary_rows(results))


def print_summary(results: Sequence[DriveResult]) -> None:
    """Print the pooled table."""
    print(
        f"{'condition':<12}{'starts':<9}{'region':<10}{'drives':>7}{'mean m':>9}{'seconds':>9}"
        f"{'flips/s':>9}{'|steer|':>9}{'lock':>7}{'step':>7}"
    )
    for row in summary_rows(results):
        print(
            f"{row['condition']:<12}{row['start_kind']:<9}{row['region']:<10}{row['drives']:>7}"
            f"{row['mean_distance_m']:>9.0f}{row['seconds']:>9.1f}{row['flips_per_s']:>9.2f}"
            f"{row['mean_abs_steer']:>9.2f}{row['full_lock_share']:>7.2f}{row['mean_step']:>7.2f}"
        )
