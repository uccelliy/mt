"""Shared helpers for timestamped machine-readable reports."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


DEFAULT_REPORT_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def report_timestamp(now: datetime | None = None) -> str:
    """Return the timestamp format used for report artifacts."""

    return (now or datetime.now()).strftime(DEFAULT_REPORT_TIMESTAMP_FORMAT)


def timestamped_report_path(
    output_dir: str | Path,
    stem: str,
    suffix: str,
    *,
    timestamp: str | None = None,
) -> Path:
    """Build a timestamped report path without writing it."""

    suffix = suffix if suffix.startswith(".") else f".{suffix}"
    return Path(output_dir) / f"{stem}_{timestamp or report_timestamp()}{suffix}"


def save_json_report(
    payload: Any,
    output_dir: str | Path,
    stem: str,
    *,
    timestamp: str | None = None,
    indent: int = 2,
) -> Path:
    """Write a JSON report using the shared timestamped report naming convention."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = timestamped_report_path(output_dir, stem, ".json", timestamp=timestamp)
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    return path


def save_text_report(
    content: str,
    output_dir: str | Path,
    stem: str,
    *,
    timestamp: str | None = None,
    suffix: str = ".log",
) -> Path:
    """Write a text report using the shared timestamped report naming convention."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = timestamped_report_path(output_dir, stem, suffix, timestamp=timestamp)
    path.write_text(content, encoding="utf-8")
    return path
