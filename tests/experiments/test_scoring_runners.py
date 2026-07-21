"""Tests for experiment-runner failure isolation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"
                       / "experiments"))

import run_transcript_scoring as transcript_runner  # noqa: E402
import run_window_scoring as window_runner  # noqa: E402

ROW = {'experiment': "task", 'participant': "1",
       'text': "Instructions.\nTrial: <<A>>\n"}

def raise_runtime_error(*args, **kwargs):
    raise RuntimeError("kernel launch failed")

def raise_oom(*args, **kwargs):
    raise RuntimeError("CUDA out of memory")

def test_transcript_runner_reraises_non_oom_runtime_errors(monkeypatch,
                                                           tmp_path):
    monkeypatch.setattr(transcript_runner, "score_session_rows",
                        raise_runtime_error)
    failures = tmp_path / "failed.csv"
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        transcript_runner.score_chunk(None, None, [ROW], "cuda", 1024,
                                      failures)
    assert not failures.exists()

def test_window_runner_reraises_non_oom_runtime_errors(monkeypatch,
                                                       tmp_path):
    monkeypatch.setattr(window_runner, "score_window_grid",
                        raise_runtime_error)
    failures = tmp_path / "failed.csv"
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        window_runner.score_session_windows(None, None, ROW, [0], 1,
                                            "cuda", 1024, failures)
    assert not failures.exists()

def test_transcript_runner_logs_real_oom(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript_runner, "score_session_rows", raise_oom)
    monkeypatch.setattr(transcript_runner, "empty_device_cache",
                        lambda device: None)
    failures = tmp_path / "failed.csv"
    assert transcript_runner.score_chunk(None, None, [ROW], "cuda", 1024,
                                         failures) == []
    frame = pd.read_csv(failures)
    assert frame.loc[0, "participant"] == 1
    assert "out of memory" in frame.loc[0, "error"].lower()

def test_window_runner_logs_real_oom(monkeypatch, tmp_path):
    monkeypatch.setattr(window_runner, "score_window_grid", raise_oom)
    monkeypatch.setattr(window_runner, "empty_device_cache",
                        lambda device: None)
    failures = tmp_path / "failed.csv"
    assert window_runner.score_session_windows(
        None, None, ROW, [0], 1, "cuda", 1024, failures) == []
    frame = pd.read_csv(failures)
    assert frame.loc[0, "participant"] == 1
    assert "out of memory" in frame.loc[0, "error"].lower()

def test_transcript_summary_separates_paper_and_macro_metrics():
    frame = pd.DataFrame([
        {'experiment': 'a', 'participant': '1', 'nll': 2.0,
         'num_tokens': 1},
        {'experiment': 'a', 'participant': '1', 'nll': 4.0,
         'num_tokens': 2},
        {'experiment': 'a', 'participant': '2', 'nll': 8.0,
         'num_tokens': 1},
    ])
    summary = transcript_runner.summarize_scores(frame).loc['a']
    assert summary['paper_token_nll'] == pytest.approx(3.5)
    assert summary['macro_choice_nll'] == pytest.approx(5.5)
    assert summary['choices'] == 3
    assert summary['choice_tokens'] == 4
    assert summary['participants'] == 2
