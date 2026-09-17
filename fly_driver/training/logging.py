"""CSV metrics logger with optional Weights & Biases."""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Always writes CSV. W&B is used only when enabled and installed."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        wandb_enabled: bool = False,
        wandb_project: str = "formula-fly",
        wandb_mode: str = "offline",
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None
        self._file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter[str] | None = None
        self._wandb = None
        if wandb_enabled:
            try:
                import wandb
            except ImportError as exc:
                raise ImportError(
                    "wandb is optional. Install with: pip install 'fly-driver[logging]' "
                    "or set logging.wandb: false."
                ) from exc
            self._wandb = wandb.init(
                project=wandb_project,
                mode=wandb_mode,
                config=dict(run_config or {}),
            )

    def log(self, step: int, metrics: Mapping[str, Any]) -> None:
        row = {"step": int(step), **{k: _as_csv(v) for k, v in metrics.items()}}
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        assert self._writer is not None
        self._writer.writerow(row)
        self._file.flush()
        if self._wandb is not None:
            self._wandb.log(dict(metrics), step=int(step))

    def close(self) -> None:
        self._file.close()
        if self._wandb is not None:
            self._wandb.finish()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _as_csv(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return str(value)
    return value
