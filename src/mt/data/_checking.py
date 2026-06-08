"""Contract checking utilities with machine-readable reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from mt.data._contracts import (
    DataContract,
    missing_required_columns,
    standard_behavior_contract,
)


LOGGER = logging.getLogger(__name__)
DEFAULT_CONTRACT_LOG_DIR = Path("logs") / "data_contract"


@dataclass
class SkippedDataSource:
    """One data source skipped during contract checking."""

    path: str
    missing_columns: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "missing_columns": list(self.missing_columns),
            "error": self.error,
        }


@dataclass
class ContractCheckReport:
    """Machine-readable result from checking a set of files against a contract."""

    contract: str
    valid: tuple[str, ...] = ()
    skipped: tuple[SkippedDataSource, ...] = ()
    log_path: Path | None = None
    report_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def valid_count(self) -> int:
        return len(self.valid)

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "valid": list(self.valid),
            "skipped": [source.to_dict() for source in self.skipped],
            "valid_count": self.valid_count,
            "skipped_count": self.skipped_count,
            "log_path": str(self.log_path) if self.log_path is not None else None,
            "report_path": str(self.report_path) if self.report_path is not None else None,
            "metadata": self.metadata,
        }


def check_data_contract(
    data_root: str | Path,
    contract: DataContract | None = None,
    *,
    pattern: str = "*.csv",
    log_dir: str | Path = DEFAULT_CONTRACT_LOG_DIR,
    save: bool = True,
    logger: logging.Logger | None = None,
) -> ContractCheckReport:
    """Check files against a contract, log skips, and optionally save a report."""

    contract = contract or standard_behavior_contract()
    data_root = Path(data_root)
    log = logger or LOGGER
    valid: list[str] = []
    skipped: list[SkippedDataSource] = []
    log_lines: list[str] = []

    for path in sorted(data_root.glob(pattern)):
        try:
            columns = pd.read_csv(path, nrows=0).columns
            missing = missing_required_columns(columns, contract)
        except (OSError, ValueError) as exc:
            skipped_source = SkippedDataSource(path=str(path), error=str(exc))
            skipped.append(skipped_source)
            line = f"Skipping {path.name}: {exc}"
            log.error(line)
            log_lines.append(line)
            continue

        if missing:
            skipped_source = SkippedDataSource(
                path=str(path),
                missing_columns=tuple(missing),
            )
            skipped.append(skipped_source)
            line = f"Skipping {path.name}: missing columns {missing}"
            log.error(line)
            log_lines.append(line)
            continue

        valid.append(str(path))

    report = ContractCheckReport(
        contract=contract.name,
        valid=tuple(valid),
        skipped=tuple(skipped),
        metadata={
            "data_root": str(data_root),
            "pattern": pattern,
        },
    )

    if save:
        _save_contract_report(report, log_lines, Path(log_dir))

    return report


def _save_contract_report(
    report: ContractCheckReport,
    log_lines: list[str],
    log_dir: Path,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{report.contract}_{timestamp}"
    report.log_path = log_dir / f"{stem}.log"
    report.report_path = log_dir / f"{stem}.json"

    summary = f"valid={report.valid_count} skipped={report.skipped_count}"
    report.log_path.write_text(
        "\n".join([summary, *log_lines, ""]),
        encoding="utf-8",
    )
    report.report_path.write_text(
        json.dumps(report.to_dict(), indent=2),
        encoding="utf-8",
    )
