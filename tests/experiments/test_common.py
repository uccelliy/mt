"""Tests for shared experiment-runner plumbing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"
                       / "experiments"))

from _common import (  # noqa: E402
    append_records,
    completed_sessions,
    filter_by_max_chars,
    is_device_out_of_memory,
    load_sessions,
    parse_shard,
    report_skips,
    resolve_dtype,
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

def test_device_out_of_memory_rejects_unrelated_runtime_errors():
    assert is_device_out_of_memory(RuntimeError("CUDA out of memory"))
    assert is_device_out_of_memory(RuntimeError("MPS backend out of memory"))
    assert not is_device_out_of_memory(RuntimeError("kernel launch failed"))

def test_append_records_ignores_empty_batches(tmp_path):
    output = tmp_path / "out.csv"
    append_records(output, [])
    assert not output.exists()

def test_completed_sessions_accepts_empty_files(tmp_path):
    output = tmp_path / "out.csv"
    output.touch()
    assert completed_sessions(output) == set()

def test_load_sessions_filters_one_exact_participant(tmp_path):
    data = tmp_path / "sessions.jsonl"
    data.write_text("\n".join(json.dumps(row) for row in make_rows()))
    rows = load_sessions(data, experiment="a", participant="2")
    assert [row['participant'] for row in rows] == ["2"]

def test_load_sessions_reads_utf8_independently_of_windows_code_page(tmp_path):
    data = tmp_path / "sessions.jsonl"
    data.write_text(json.dumps({"experiment": "a", "participant": "1",
                                "text": "it’s valid UTF-8"},
                               ensure_ascii=False), encoding="utf-8")
    assert load_sessions(data)[0]["text"] == "it’s valid UTF-8"

def test_resolve_dtype_accepts_indexed_device_names():
    import torch

    assert resolve_dtype("auto", "cuda:0") is torch.float16
    assert resolve_dtype("auto", "mps:0") is torch.bfloat16
