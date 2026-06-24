"""Checkpoint and artifact helpers for behavioral LM finetuning."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

from transformers import TrainerCallback


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_step(path: str | Path) -> int | None:
    """Return the global step encoded in a Transformers checkpoint directory."""

    match = CHECKPOINT_RE.match(Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Find the newest checkpoint directory in a Trainer output directory."""

    root = Path(output_dir)
    if not root.exists():
        return None
    checkpoints = [
        (step, path)
        for path in root.iterdir()
        if path.is_dir()
        for step in [checkpoint_step(path)]
        if step is not None
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: item[0])[1]


@dataclass
class TrainingArtifactCallback(TrainerCallback):
    """Trainer callback that writes jsonl/csv metrics and periodically refreshes plots."""

    output_dir: str | Path
    plot_steps: int = 50
    log_dir_name: str = "logs"
    plot_dir_name: str = "plots"
    records: list[dict[str, Any]] = field(default_factory=list)
    _last_plot_step: int = 0

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.log_dir = self.output_dir / self.log_dir_name
        self.plot_dir = self.output_dir / self.plot_dir_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "trainer_metrics.jsonl"
        self.csv_path = self.log_dir / "trainer_metrics.csv"
        self.records = _load_jsonl_records(self.jsonl_path)
        if self.records:
            self._last_plot_step = int(self.records[-1].get("step", 0) or 0)

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not getattr(state, "is_world_process_zero", True):
            return control

        logs = dict(logs or {})
        step = int(getattr(state, "global_step", 0) or logs.get("step", 0) or 0)
        record: dict[str, Any] = {
            "time": datetime.now(timezone.utc).isoformat(),
            "step": step,
            "epoch": getattr(state, "epoch", None),
        }
        record.update(_json_safe_mapping(logs))
        self.records.append(record)
        _append_jsonl(self.jsonl_path, record)
        _write_csv(self.csv_path, self.records)

        if step == 0 or step - self._last_plot_step >= self.plot_steps:
            plot_metric_history(self.records, self.plot_dir)
            self._last_plot_step = step
        return control


def plot_metric_history(records: list[dict[str, Any]], output_dir: str | Path) -> None:
    """Plot loss, perplexity, and learning-rate curves from Trainer log records."""

    if not records:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    if "step" not in df.columns:
        return

    for column in ("loss", "eval_loss", "learning_rate"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["step"] = pd.to_numeric(df["step"], errors="coerce")

    loss_columns = [column for column in ("loss", "eval_loss") if column in df.columns]
    if loss_columns:
        plt.figure()
        for column in loss_columns:
            plot_df = df[df[column].notna()]
            if not plot_df.empty:
                plt.plot(plot_df["step"], plot_df[column], label=column)
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Training and Evaluation Loss")
        plt.legend()
        plt.savefig(output / "loss.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        for column in loss_columns:
            plot_df = df[df[column].notna()].copy()
            if plot_df.empty:
                continue
            ppl = plot_df[column].apply(lambda value: math.exp(value) if value < 20 else float("inf"))
            plt.plot(plot_df["step"], ppl, label=column.replace("loss", "perplexity"))
        plt.xlabel("Step")
        plt.ylabel("Perplexity")
        plt.title("Training and Evaluation Perplexity")
        plt.legend()
        plt.savefig(output / "perplexity.png", dpi=300, bbox_inches="tight")
        plt.close()

    if "learning_rate" in df.columns:
        lr_df = df[df["learning_rate"].notna()]
        if not lr_df.empty:
            plt.figure()
            plt.plot(lr_df["step"], lr_df["learning_rate"], label="learning_rate")
            plt.xlabel("Step")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.legend()
            plt.savefig(output / "learning_rate.png", dpi=300, bbox_inches="tight")
            plt.close()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for record in records for key in record})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _json_safe_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: _json_safe(value) for key, value in mapping.items()}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return str(value)
    return str(value)
