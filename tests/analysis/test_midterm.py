"""Mean ± std over seeds for the GH-27 table (``fly_driver.analysis.midterm.aggregate``)."""

from __future__ import annotations

import math

import pytest

from fly_driver.analysis.midterm import SeedMetrics, aggregate, markdown_table, seed_metrics
from fly_driver.training.evaluation import ConditionSummary, EvalConfig, EvalReport
from fly_driver.training.perturbations import PerturbationSpec


def _seed(clean_return: float, lap_time: float | None, noisy_return: float) -> SeedMetrics:
    def summary(condition: str, mean_return: float, ratio: float | None) -> ConditionSummary:
        return ConditionSummary(
            condition=condition,
            episodes=3,
            mean_return=mean_return,
            std_return=0.0,
            lap_completion_rate=0.0 if lap_time is None else 1.0,
            mean_lap_time_s=lap_time,
            mean_steps=100.0,
            return_ratio_to_clean=ratio,
        )

    config = EvalConfig(
        perturbations=(PerturbationSpec.from_dict({"name": "brightness", "scale": 0.5}),)
    )
    noisy_label = config.condition_labels[1]
    report = EvalReport(
        config=config,
        episodes=(),
        conditions=(
            summary("clean", clean_return, None),
            summary(noisy_label, noisy_return, noisy_return / clean_return),
        ),
        actions_digest="",
        video_backend=None,
    )
    # Progress mirrors the return; the noisy condition has no progress file entry.
    return seed_metrics(report, {"clean": [clean_return, clean_return]})


def test_mean_and_sample_std_over_seeds_per_model_and_condition() -> None:
    rows = aggregate(
        {
            "cnn": [
                _seed(10.0, 90.0, 5.0),
                _seed(20.0, None, 20.0),
                _seed(30.0, 100.0, 15.0),
            ],
            "flyvis": [_seed(4.0, None, 2.0)],
            "flyvis+central_complex": [_seed(4.0, None, 2.0)],
        }
    )
    # flyvis sorts first (MODEL_ORDER), an unknown model last; each has two rows.
    models = [row["model"] for row in rows]
    assert models == ["flyvis"] * 2 + ["cnn"] * 2 + ["flyvis+central_complex"] * 2
    cnn_clean = next(r for r in rows if r["model"] == "cnn" and r["condition"] == "clean")
    assert cnn_clean["seeds"] == 3
    assert cnn_clean["return_mean"] == pytest.approx(20.0)
    assert cnn_clean["return_std"] == pytest.approx(10.0)  # ddof=1
    assert cnn_clean["lap_completion_mean"] == pytest.approx(2 / 3)
    # Lap time averages only the seeds that finished a lap.
    assert cnn_clean["lap_time_s_mean"] == pytest.approx(95.0)
    assert cnn_clean["lap_time_s_n"] == 2
    cnn_noisy = next(r for r in rows if r["model"] == "cnn" and r["condition"] != "clean")
    assert cnn_noisy["return_ratio_mean"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)
    flyvis_clean = rows[0]
    assert math.isnan(flyvis_clean["return_std"])  # one seed: no spread to report
    assert math.isnan(flyvis_clean["lap_time_s_mean"])
    table = markdown_table(rows)
    assert "| cnn | 3 | 0.67 ± 0.58 | 95.0 ± 7.1 (n=2) |" in table
    assert "| flyvis | 1 | 0.00 | – | 4 | 4.0 |" in table
    assert cnn_clean["progress_m_mean"] == pytest.approx(20.0)
    assert math.isnan(cnn_noisy["progress_m_mean"]) and cnn_noisy["progress_m_n"] == 0
