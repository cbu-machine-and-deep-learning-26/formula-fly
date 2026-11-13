"""What a training run did, as text a reviewer can read in one sitting (GH-27, reward coach).

The reward coach (:mod:`fly_driver.coach.review`) hands this digest to Claude. Its numbers
come from the files every run already writes -- ``episodes.csv``, ``updates.csv`` and
``config.yaml`` -- so it works on any run, old or new, local or copied off a lab PC. Nothing
here writes anywhere, so it is safe to run while a trainer is still writing those files.

It answers the questions a reward review needs, in this order:

1. **The recipe**: the reward, the slow rule, the starts and the PPO settings, as run.
2. **Progress**: drives, laps, the supervisor's 50-drive score, and the trend over time.
3. **Where drives end**: per segment of the lap, with why (crash, stall, step cap) and how
   fast the car was going, when the run logged it.
4. **Corners**: how many drives reached each corner and how many got out of it.
5. **Reward terms**: what each term paid per drive, and return per metre, which is the
   farming check `AGENTS.md` §11 asks for -- a policy that earns without progressing shows
   up as return out of line with distance.
6. **PPO health**: entropy, KL, clip fraction, explained variance, controls, steps/s.
"""

from __future__ import annotations

import csv
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from fly_driver.analysis.lab_compare import corner_survival
from fly_driver.envs.centerline import Centerline
from fly_driver.envs.segments import Segment, find_segments
from fly_driver.training.lab import WINDOW_DRIVES, best_window, read_drives, saved_checkpoints

__all__ = ["RunDigest", "build_digest", "format_digest", "lap_segments", "seed_dirs"]

#: The recipe keys a reward review reads, as dotted paths into a run's ``config.yaml``.
RECIPE_KEYS = (
    "num_envs",
    "total_env_steps",
    "init_from",
    "env.max_steps",
    "env.params.reward",
    "env.params.random_start",
    "env.params.start_focus",
    "env.params.stall_speed_mps",
    "env.params.stall_seconds",
    "env.params.track_limit",
    "env.params.racing_line",
    "ppo",
    "brain",
    "eye",
)

_TREND_BUCKETS = 8


def lap_segments() -> tuple[float, list[Segment]]:
    """The lap's length and its segments, as the env times them."""
    centerline = Centerline.load()
    return float(centerline.length), find_segments(centerline)


def seed_dirs(paths: Iterable[Path]) -> list[Path]:
    """Every ``seed_<k>`` directory under the given run or seed directories, sorted."""
    found: set[Path] = set()
    for path in paths:
        path = Path(path)
        if (path / "episodes.csv").exists():
            found.add(path)
        found.update(child.parent for child in path.glob("**/seed_*/episodes.csv"))
    return sorted(found)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _dotted(config: Mapping[str, Any], key: str) -> Any:
    value: Any = config
    for part in key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


@dataclass
class RunDigest:
    """The numbers for one or more seeds of a run, pooled.

    Args:
        name: What the digest covers.
        recipe: :data:`RECIPE_KEYS` from the newest seed's config.
        seeds: One line of headline numbers per seed.
        drives: Every drive the policy drove, pooled, in each seed's order. Teacher-driven
            drives (``env.params.teacher_drive``) are left out of every statistic: the teacher
            laps because it can, and its passes would read as the policy's.
        updates: Every PPO update row, pooled.
        lap_m: Lap length.
        segments: The lap's segments.
    """

    name: str
    recipe: dict[str, Any]
    seeds: list[dict[str, Any]]
    drives: list[dict[str, str]] = field(default_factory=list)
    updates: list[dict[str, str]] = field(default_factory=list)
    lap_m: float = 0.0
    segments: list[Segment] = field(default_factory=list)


def build_digest(paths: Sequence[Path], *, name: str | None = None) -> RunDigest:
    """Read every seed under ``paths``.

    Raises:
        FileNotFoundError: If no ``episodes.csv`` is found under any of them.
    """
    dirs = seed_dirs(paths)
    if not dirs:
        raise FileNotFoundError(f"no seed_<k>/episodes.csv under {[str(p) for p in paths]}")
    lap_m, segments = lap_segments()
    newest = max(dirs, key=lambda d: (d / "episodes.csv").stat().st_mtime)
    config: dict[str, Any] = {}
    for config_path in (newest / "config.yaml", newest.parent / "config.yaml"):
        if config_path.exists():
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            break
    recipe = {key: _dotted(config, key) for key in RECIPE_KEYS}
    digest = RunDigest(
        name=name or ", ".join(f"{d.parent.name}/{d.name}" for d in dirs),
        recipe=recipe,
        seeds=[],
        lap_m=lap_m,
        segments=segments,
    )
    for seed_dir in dirs:
        rows = _read_rows(seed_dir / "episodes.csv")
        drives = [row for row in rows if row.get("teacher_driven") != "True"]
        updates = _read_rows(seed_dir / "updates.csv")
        lab_drives = [drive for drive in read_drives(seed_dir) if not drive.teacher_driven]
        window = best_window(lab_drives, saved_checkpoints(seed_dir))
        uniform = [drive for drive in lab_drives if not drive.start_focused]
        last = uniform[-WINDOW_DRIVES:]
        digest.seeds.append(
            {
                "seed_dir": f"{seed_dir.parent.name}/{seed_dir.name}",
                "env_steps": int(_number(updates[-1], "env_steps") or 0) if updates else 0,
                "updates": len(updates),
                "drives": len(drives),
                "teacher_drives": len(rows) - len(drives),
                "laps": sum(row.get("lap_complete") == "True" for row in drives),
                "last_50_uniform_m": (
                    sum(d.distance_m for d in last) / len(last)
                    if len(last) == WINDOW_DRIVES
                    else None
                ),
                "best_window_m": None if window is None else window.score,
                "longest_m": max((d.distance_m for d in lab_drives), default=0.0),
            }
        )
        digest.drives.extend(drives)
        digest.updates.extend(updates)
    return digest


# -- formatting ---------------------------------------------------------------------------


def _fmt(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def _segment_at(arclength: float, segments: Sequence[Segment], lap_m: float) -> Segment:
    position = arclength % lap_m
    for segment in segments:
        if segment.start_m <= position < segment.end_m:
            return segment
    return segments[-1]


def _recipe_lines(digest: RunDigest) -> list[str]:
    dumped = yaml.safe_dump(digest.recipe, sort_keys=False, default_flow_style=None, width=100)
    return ["## Recipe (from the newest seed's config.yaml)", "", "```yaml", dumped.rstrip(), "```"]


def _progress_lines(digest: RunDigest) -> list[str]:
    lines = [
        "## Progress per seed",
        "",
        "Policy drives only: teacher-driven drives (env.params.teacher_drive) are counted in "
        "their own column and left out of every other number here.",
        "",
        "| Seed | Env steps | Updates | Drives | Teacher drives | Laps | Last 50 uniform (m) "
        "| Best window (m) | Longest (m) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for seed in digest.seeds:
        lines.append(
            f"| {seed['seed_dir']} | {seed['env_steps']:,} | {seed['updates']} "
            f"| {seed['drives']:,} | {seed.get('teacher_drives', 0):,} "
            f"| {seed['laps']} | {_fmt(seed['last_50_uniform_m'])} | {_fmt(seed['best_window_m'])} "
            f"| {_fmt(seed['longest_m'])} |"
        )
    distances = [d for row in digest.drives if (d := _number(row, "distance_m")) is not None]
    if len(distances) >= _TREND_BUCKETS:
        size = len(distances) // _TREND_BUCKETS
        means = [
            statistics.fmean(distances[i * size : (i + 1) * size]) for i in range(_TREND_BUCKETS)
        ]
        lines += [
            "",
            f"Mean distance per drive over time, {_TREND_BUCKETS} equal buckets of {size:,} drives "
            "(all starts, pooled seeds in file order): "
            + " -> ".join(f"{mean:,.0f}" for mean in means)
            + " m",
        ]
    return lines


def _ending_lines(digest: RunDigest) -> list[str]:
    ends: dict[str, Counter[str]] = defaultdict(Counter)
    speeds: dict[str, list[float]] = defaultdict(list)
    has_reason = False
    for row in digest.drives:
        start_m, distance_m = _number(row, "start_m"), _number(row, "distance_m")
        if start_m is None or distance_m is None:
            continue
        segment = _segment_at(start_m + distance_m, digest.segments, digest.lap_m).name
        reason = row.get("end_reason") or ""
        has_reason = has_reason or bool(reason)
        if not reason:
            reason = "truncated" if row.get("truncated") == "True" else "ended"
        ends[segment][reason] += 1
        if (speed := _number(row, "max_speed_kmh")) is not None:
            speeds[segment].append(speed)
    total = sum(sum(counter.values()) for counter in ends.values())
    if not total:
        return []
    lines = [
        "## Where drives end",
        "",
        "Segment the drive's last metre is in. "
        + (
            "Reasons are the env's: crash = past the track limit, stall = the slow rule."
            if has_reason
            else "This run predates end_reason logging; 'ended' is any termination."
        ),
        "",
        "| Segment | Kind | From (m) | Drives ending | Share | Reasons | Median top speed (km/h) |",
        "|---|---|---|---|---|---|---|",
    ]
    for segment in digest.segments:
        counter = ends.get(segment.name)
        if not counter:
            continue
        count = sum(counter.values())
        reasons = ", ".join(f"{name} {n:,}" for name, n in counter.most_common())
        median = statistics.median(speeds[segment.name]) if speeds[segment.name] else None
        lines.append(
            f"| {segment.name} | {'corner' if segment.is_corner else 'straight'} "
            f"| {segment.start_m:,.0f} | {count:,} | {count / total:.1%} | {reasons} "
            f"| {_fmt(median)} |"
        )
    return lines


def _corner_lines(digest: RunDigest) -> list[str]:
    positions = [
        (start, distance)
        for row in digest.drives
        if (start := _number(row, "start_m")) is not None
        and (distance := _number(row, "distance_m")) is not None
    ]
    if not positions:
        return []
    half = len(positions) // 2
    lines = [
        "## Corners: arrivals and passes",
        "",
        "A drive arrives when it reaches the corner's entry and passes when it leaves the exit. "
        "A drive that started inside the corner does not count as arriving. "
        "First half vs second half of the drives shows the trend.",
        "",
        "| Corner | From-to (m) | Arrivals | Passed | Pass rate | 1st half | 2nd half |",
        "|---|---|---|---|---|---|---|",
    ]
    for segment in digest.segments:
        if not segment.is_corner:
            continue
        bounds = {"from_m": segment.start_m, "to_m": segment.end_m, "lap_m": digest.lap_m}
        arrivals, passes = corner_survival(positions, **bounds)
        early = corner_survival(positions[:half], **bounds)
        late = corner_survival(positions[half:], **bounds)

        def rate(pair: tuple[int, int]) -> str:
            return f"{pair[1] / pair[0]:.1%}" if pair[0] else "-"

        lines.append(
            f"| {segment.name} | {segment.start_m:,.0f}-{segment.end_m:,.0f} | {arrivals:,} "
            f"| {passes:,} | {rate((arrivals, passes))} | {rate(early)} | {rate(late)} |"
        )
    return lines


def _reward_lines(digest: RunDigest) -> list[str]:
    term_columns = sorted(
        {key for row in digest.drives[:1] for key in row if key.startswith("reward_")}
    )
    if not term_columns:
        return []
    totals = {column: 0.0 for column in term_columns}
    returns, distances = [], []
    for row in digest.drives:
        for column in term_columns:
            totals[column] += _number(row, column) or 0.0
        total, distance = _number(row, "total_return"), _number(row, "distance_m")
        if total is not None and distance is not None:
            returns.append(total)
            distances.append(distance)
    drives = max(len(digest.drives), 1)
    absolute = sum(abs(value) for value in totals.values()) or 1.0
    lines = [
        "## Reward terms (scaled, as the policy saw them)",
        "",
        "| Term | Mean per drive | Share of absolute total |",
        "|---|---|---|",
    ]
    for column in term_columns:
        lines.append(
            f"| {column.removeprefix('reward_')} | {totals[column] / drives:+.3f} "
            f"| {abs(totals[column]) / absolute:.1%} |"
        )
    if len(returns) >= 3 and statistics.pstdev(distances) > 0 and statistics.pstdev(returns) > 0:
        correlation = statistics.correlation(returns, distances)
        per_metre = [r / d for r, d in zip(returns, distances, strict=True) if d > 50]
        lines += [
            "",
            f"Return vs distance, Pearson r = {correlation:.3f} over {len(returns):,} drives. "
            "Return per metre (drives over 50 m): "
            + (
                f"median {statistics.median(per_metre):.4f}, "
                f"p95 {sorted(per_metre)[int(0.95 * (len(per_metre) - 1))]:.4f}"
                if per_metre
                else "-"
            )
            + ". A low r, or a p95 far above the median, would mean some drives earn without "
            "driving: the farming check.",
        ]
    return lines


def _ppo_lines(digest: RunDigest) -> list[str]:
    if not digest.updates:
        return []
    columns = (
        "entropy",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
        "value_loss",
        "mean_throttle",
        "mean_brake",
        "mean_abs_steer",
        "steps_per_second",
        "learning_rate",
    )
    count = len(digest.updates)
    stretches = {
        "first 10%": digest.updates[: max(count // 10, 1)],
        "middle": digest.updates[count * 9 // 20 : max(count * 11 // 20, count * 9 // 20 + 1)],
        "last 10%": digest.updates[-max(count // 10, 1) :],
    }
    lines = [
        "## PPO health (pooled seeds, means over each stretch of updates)",
        "",
        "| Stat | " + " | ".join(stretches) + " |",
        "|---|" + "---|" * len(stretches),
    ]
    for column in columns:
        cells = []
        for rows in stretches.values():
            values = [v for row in rows if (v := _number(row, column)) is not None]
            cells.append(f"{statistics.fmean(values):.4g}" if values else "-")
        if any(cell != "-" for cell in cells):
            lines.append(f"| {column} | " + " | ".join(cells) + " |")
    return lines


def format_digest(digest: RunDigest) -> str:
    """The digest as Markdown, every section in order."""
    sections = [
        [f"# Training digest: {digest.name}", "", f"Lap length {digest.lap_m:,.0f} m."],
        _recipe_lines(digest),
        _progress_lines(digest),
        _corner_lines(digest),
        _ending_lines(digest),
        _reward_lines(digest),
        _ppo_lines(digest),
    ]
    return "\n\n".join("\n".join(section) for section in sections if section) + "\n"
