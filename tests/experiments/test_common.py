"""Tests for shared experiment-runner plumbing."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"
                       / "experiments"))

from _common import (  # noqa: E402
    filter_by_max_chars,
    parse_shard,
    report_skips,
    skip_log_for,
)

def make_rows():
    return [{'experiment': "a", 'participant': "1", 'text': "x" * 10},
            {'experiment': "a", 'participant': "2", 'text': "x" * 100},
            {'experiment': "b", 'participant': "3", 'text': "x" * 200}]

def test_filter_by_max_chars_partitions_on_length():
    kept, skipped = filter_by_max_chars(make_rows(), 100)
    assert [r['participant'] for r in kept] == ["1", "2"]
    assert [r['participant'] for r in skipped] == ["3"]

def test_filter_by_max_chars_keeps_boundary_length():
    kept, skipped = filter_by_max_chars(make_rows(), 10)
    assert len(kept) == 1 and len(skipped) == 2

def test_report_skips_writes_log_csv(tmp_path, capsys):
    rows = make_rows()
    _, skipped = filter_by_max_chars(rows, 50)
    log = tmp_path / "out.skipped.csv"
    report_skips(rows, skipped, 50, skip_log=log)
    written = pd.read_csv(log, dtype=str)
    assert set(written['participant']) == {"2", "3"}
    assert "ENTIRE TASK DROPPED" in capsys.readouterr().out

def test_report_skips_silent_when_nothing_skipped(tmp_path, capsys):
    log = tmp_path / "out.skipped.csv"
    report_skips(make_rows(), [], 1000, skip_log=log)
    assert not log.exists()
    assert capsys.readouterr().out == ""

def test_skip_log_for_derives_sibling_path():
    assert skip_log_for("a/b/run.csv") == Path("a/b/run.skipped.csv")

def test_parse_shard_roundtrip_and_validation():
    assert parse_shard("1/4") == (1, 4)
    assert parse_shard(None) is None
    with pytest.raises(SystemExit):
        parse_shard("4/4")
