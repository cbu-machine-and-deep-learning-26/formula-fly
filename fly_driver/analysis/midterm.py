"""GH-27 midterm results: the RQ1 comparison, aggregated from run directories.

Reads what training and the robustness evaluation wrote, and nothing else::

    <runs>/<condition>/seed_<k>/config.yaml               which eye (and brain) the run used
    <runs>/<condition>/seed_<k>/eval/step_<n>/summary.json  the learning curve
    <runs>/<condition>/seed_<k>/robustness/summary.json     the final evaluation

Runs are grouped by model: the eye type, plus ``+<brain>`` when the run had one, so a
flyvis + central-complex run is not pooled with plain flyvis. The final evaluation (clean
plus every perturbation) is aggregated per model as the mean and
sample standard deviation over seeds; each seed contributes one number per metric, its mean
over the evaluation episodes. ``scripts/midterm_results.py`` runs the robustness
evaluation (:mod:`fly_driver.analysis.robustness`) where it is missing and calls
:func:`write_results`.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from fly_driver.analysis.learning_curves import learning_curve
from fly_driver.training.config import TrainConfig
from fly_driver.training.evaluation import (
    CLEAN_CONDITION,
    ConditionSummary,
    EvalConfig,
    EvalReport,
)

__all__ = [
    "MODEL_ORDER",
    "ROBUSTNESS_DIRNAME",
    "SeedRun",
    "aggregate",
    "find_seed_runs",
    "load_report",
    "markdown_table",
    "plot_learning_curves",
    "seed_metrics",
    "write_results",
]

ROBUSTNESS_DIRNAME = "robustness"

#: Table and legend order: the connectome first, then the three RQ1 controls. Any other model
#: (e.g. ``flyvis+central_complex``) follows, alphabetically.
MODEL_ORDER = ("flyvis", "shuffled", "cnn", "random_projection")

#: Per-seed metrics read off a condition summary, as ``(column, attribute)``.
_SUMMARY_METRICS = (
    ("lap_completion", "lap_completion_rate"),
    ("lap_time_s", "mean_lap_time_s"),
    ("return", "mean_return"),
    ("return_ratio", "return_ratio_to_clean"),
    ("progress_m", "mean_progress_m"),
)

_SUMMARY_FIELDS = tuple(field.name for field in fields(ConditionSummary))

#: Condition label → metric column → this seed's value (``None`` when it has none).
SeedMetrics = dict[str, dict[str, float | None]]


@dataclass(frozen=True)
class SeedRun:
    """One finished seed: where it is and which model it trained."""

    model: str
    seed: int
    run_dir: Path


def load_report(path: str | Path) -> EvalReport:
    """Rebuild an :class:`EvalReport` from its ``summary.json`` (episode rows are dropped).

    Keys the harness adds later are ignored, so older and newer runs load alike.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return EvalReport(
        config=EvalConfig.from_dict(data["config"]),
        episodes=(),
        conditions=tuple(
            ConditionSummary(**{key: item[key] for key in _SUMMARY_FIELDS})
            for item in data["conditions"]
        ),
        actions_digest=data["actions_digest"],
        video_backend=data["video_backend"],
    )


def find_seed_runs(runs_dir: str | Path) -> list[SeedRun]:
    """Every ``<condition>/seed_<k>`` under ``runs_dir`` that has a final checkpoint."""
    runs = []
    for config_path in sorted(Path(runs_dir).glob("*/seed_*/config.yaml")):
        run_dir = config_path.parent
        if not (run_dir / "policy_final.pt").exists():
            continue
        config = TrainConfig.from_yaml(config_path)
        brain = config.brain.type
        model = config.eye.type if brain == "none" else f"{config.eye.type}+{brain}"
        runs.append(SeedRun(model=model, seed=int(config.seeds[0]), run_dir=run_dir))
    return runs


def _model_order(models: Iterable[str]) -> list[str]:
    return sorted(
        set(models),
        key=lambda model: (
            MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER),
            model,
        ),
    )


def _mean_std(values: Sequence[float | None]) -> tuple[float, float, int]:
    """Mean, sample std (NaN below two values) and count of the non-missing values."""
    present = np.array([value for value in values if value is not None], dtype=np.float64)
    if present.size == 0:
        return math.nan, math.nan, 0
    std = float(present.std(ddof=1)) if present.size > 1 else math.nan
    return float(present.mean()), std, int(present.size)


def seed_metrics(report: EvalReport) -> SeedMetrics:
    """One seed's numbers per evaluation condition, read off its robustness report."""
    return {
        item.condition: {column: getattr(item, attribute) for column, attribute in _SUMMARY_METRICS}
        for item in report.conditions
    }


def aggregate(seeds: dict[str, list[SeedMetrics]]) -> list[dict[str, object]]:
    """Mean ± std over seeds, per model and per evaluation condition.

    Args:
        seeds: Model → one :func:`seed_metrics` result per seed.

    Returns:
        One row per (model, condition): ``model, condition, seeds``, then ``<metric>_mean``,
        ``<metric>_std`` and ``<metric>_n`` (seeds with a value; lap time only exists for
        seeds that finished a lap) for every metric.
    """
    rows: list[dict[str, object]] = []
    for model in _model_order(seeds):
        per_seed = seeds[model]
        for condition in per_seed[0]:
            row: dict[str, object] = {
                "model": model,
                "condition": condition,
                "seeds": len(per_seed),
            }
            for column, _ in _SUMMARY_METRICS:
                mean, std, count = _mean_std([metrics[condition][column] for metrics in per_seed])
                row[f"{column}_mean"] = mean
                row[f"{column}_std"] = std
                row[f"{column}_n"] = count
            rows.append(row)
    return rows


def _cell(row: dict[str, object], column: str, digits: int) -> str:
    mean, std = float(row[f"{column}_mean"]), float(row[f"{column}_std"])  # type: ignore[arg-type]
    if math.isnan(mean):
        return "–"
    text = f"{mean:.{digits}f}"
    if not math.isnan(std):
        text += f" ± {std:.{digits}f}"
    if row[f"{column}_n"] != row["seeds"]:
        text += f" (n={row[f'{column}_n']})"
    return text


def markdown_table(rows: list[dict[str, object]]) -> str:
    """The clean-condition table, then the robustness table, as Markdown."""
    clean = [
        "| Model | Seeds | Lap completion | Lap time (s) | Progress (m) | Return |",
        "|---|---|---|---|---|---|",
    ]
    robust = [
        "| Model | Perturbation | Return / clean | Lap completion | Progress (m) |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        if row["condition"] == CLEAN_CONDITION:
            clean.append(
                f"| {row['model']} | {row['seeds']} | {_cell(row, 'lap_completion', 2)} | "
                f"{_cell(row, 'lap_time_s', 1)} | {_cell(row, 'progress_m', 0)} | "
                f"{_cell(row, 'return', 1)} |"
            )
        else:
            robust.append(
                f"| {row['model']} | {row['condition']} | {_cell(row, 'return_ratio', 2)} | "
                f"{_cell(row, 'lap_completion', 2)} | {_cell(row, 'progress_m', 0)} |"
            )
    return "\n".join(clean) + "\n\n" + "\n".join(robust) + "\n"


def plot_learning_curves(runs: list[SeedRun], path: str | Path) -> Path:
    """Clean evaluation return, progress and lap completion against env steps, per model.

    Each line is the mean over seeds with a ± std band.

    Seeds are aligned on the evaluation steps they all share.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for model in _model_order(run.model for run in runs):
        reports_per_seed = [
            [load_report(item) for item in sorted(run.run_dir.glob("eval/step_*/summary.json"))]
            for run in runs
            if run.model == model
        ]
        for axis, metric, label in (
            (axes[0], "mean_return", "Return"),
            (axes[1], "mean_progress_m", "Progress (m)"),
            (axes[2], "lap_completion_rate", "Lap completion"),
        ):
            curves = [learning_curve(reports, metric) for reports in reports_per_seed if reports]
            if not curves:
                continue
            shared = sorted(set.intersection(*(set(steps.tolist()) for steps, _ in curves)))
            if not shared:
                continue
            values = np.array(
                [
                    [dict(zip(steps.tolist(), ys, strict=True))[step] for step in shared]
                    for steps, ys in curves
                ]
            )
            mean = np.nanmean(values, axis=0)
            std = np.nanstd(values, axis=0)
            (line,) = axis.plot(shared, mean, label=f"{model} (n={len(curves)})")
            axis.fill_between(shared, mean - std, mean + std, color=line.get_color(), alpha=0.2)
            axis.set_ylabel(label)
    for axis in axes:
        axis.set_xlabel("Env steps")
        axis.grid(alpha=0.3)
    axes[0].legend()
    figure.suptitle("Deterministic evaluation during training (mean ± std over seeds)")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def write_results(runs_dir: str | Path, results_dir: str | Path, figure_path: str | Path) -> str:
    """Aggregate every finished seed under ``runs_dir`` into CSVs, a table and a figure.

    Writes ``midterm_results.csv`` (per model and evaluation condition), ``midterm_per_seed.csv``
    (per seed, the inputs to the first) and ``midterm_table.md`` into ``results_dir``, and the
    learning-curve figure to ``figure_path``.

    Returns:
        The Markdown table.

    Raises:
        FileNotFoundError: If no finished seed has a robustness report.
    """
    runs = find_seed_runs(runs_dir)
    seeds: dict[str, list[SeedMetrics]] = {}
    per_seed: list[dict[str, object]] = []
    for run in runs:
        robustness = run.run_dir / ROBUSTNESS_DIRNAME
        if not (robustness / "summary.json").exists():
            continue
        report = load_report(robustness / "summary.json")
        metrics = seed_metrics(report)
        seeds.setdefault(run.model, []).append(metrics)
        for condition, values in metrics.items():
            per_seed.append(
                {
                    "model": run.model,
                    "seed": run.seed,
                    "env_steps": report.config.env_steps,
                    "condition": condition,
                    "episodes": report.config.episodes,
                    **values,
                }
            )
    if not seeds:
        raise FileNotFoundError(f"no seed under {runs_dir} has {ROBUSTNESS_DIRNAME}/summary.json")
    rows = aggregate(seeds)
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)
    _write_csv(results / "midterm_results.csv", rows)
    _write_csv(results / "midterm_per_seed.csv", per_seed)
    table = markdown_table(rows)
    (results / "midterm_table.md").write_text(table, encoding="utf-8")
    plot_learning_curves(runs, figure_path)
    return table


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
