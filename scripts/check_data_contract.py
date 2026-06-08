"""Check CSV files against the default behavioral data contract."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt.data import check_data_contract


def main() -> None:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s:%(message)s")
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "mt_data" / "data"
    report = check_data_contract(data_root)
    print(f"valid={report.valid_count} skipped={report.skipped_count}")
    print(f"log={report.log_path}")
    print(f"report={report.report_path}")


if __name__ == "__main__":
    main()
