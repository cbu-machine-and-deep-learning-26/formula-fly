"""YAML experiment config loader."""

from __future__ import annotations

from pathlib import Path

from fly_driver.training.config import default_config_path, load_experiment
from fly_driver.training.logging import MetricsLogger


def test_default_yaml_condition_fields() -> None:
    cfg = load_experiment(default_config_path())
    assert cfg.condition.eye_type == "random_projection"
    assert cfg.condition.brain == "identity"
    assert cfg.condition.eye_mode == "frozen"
    assert cfg.frozen is True
    assert cfg.condition.seed == 0
    assert cfg.eye.output_size == 256
    assert cfg.logging.wandb is False


def test_named_configs_exist() -> None:
    root = Path(default_config_path()).parent
    for name in ("cnn.yaml", "random_projection.yaml", "shuffled.yaml", "flyvis_frozen.yaml"):
        cfg = load_experiment(root / name)
        assert cfg.condition.seed in {0, 1, 2} or cfg.condition.seed == 0
        assert cfg.condition.eye_mode in {"frozen", "finetuned"}


def test_csv_logger_writes_header_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    with MetricsLogger(path) as log:
        log.log(0, {"reward/progress": 0.1, "reward/on_track": 0.01})
        log.log(1, {"reward/progress": 0.2, "reward/on_track": 0.01})
    text = path.read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    assert lines[0].startswith("step,")
    assert "reward/progress" in lines[0]
    assert len(lines) == 3
