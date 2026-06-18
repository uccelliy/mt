from __future__ import annotations

import json
from datetime import datetime

from mt.data import report_timestamp, save_json_report, save_text_report, timestamped_report_path


def test_report_timestamp_uses_shared_format() -> None:
    timestamp = report_timestamp(datetime(2026, 6, 18, 9, 0, 13))

    assert timestamp == "20260618_090013"


def test_timestamped_report_path_normalizes_suffix(tmp_path) -> None:
    path = timestamped_report_path(tmp_path, "run", "json", timestamp="20260618_090013")

    assert path == tmp_path / "run_20260618_090013.json"


def test_save_json_report_writes_timestamped_json(tmp_path) -> None:
    path = save_json_report({"finite": True}, tmp_path, "run", timestamp="20260618_090013")

    assert path == tmp_path / "run_20260618_090013.json"
    assert json.loads(path.read_text(encoding="utf-8")) == {"finite": True}


def test_save_text_report_writes_timestamped_text(tmp_path) -> None:
    path = save_text_report("valid=1 skipped=0\n", tmp_path, "contract", timestamp="20260618_090013")

    assert path == tmp_path / "contract_20260618_090013.log"
    assert path.read_text(encoding="utf-8") == "valid=1 skipped=0\n"
