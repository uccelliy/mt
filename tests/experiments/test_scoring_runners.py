"""Tests for experiment-runner failure isolation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))

import run_transcript_scoring as transcript_runner  # noqa: E402
import run_window_scoring as window_runner  # noqa: E402

from mt.evaluation.transcript_scoring import ContextLengthError  # noqa: E402

ROW = {"experiment": "task", "participant": "1", "text": "Instructions.\nTrial: <<A>>\n"}
THREE_TRIAL_ROW = {
    "experiment": "task",
    "participant": "1",
    "text": ("Instructions.\nTrial 1: <<A>>\nTrial 2: <<B>>\nTrial 3: <<C>>\n"),
}


def raise_runtime_error(*args, **kwargs):
    raise RuntimeError("kernel launch failed")


def raise_oom(*args, **kwargs):
    raise RuntimeError("CUDA out of memory")


def raise_context_overflow(*args, **kwargs):
    raise ContextLengthError("Transcript has 35038 tokens, exceeding the model context of 32768.")


def test_transcript_runner_reraises_non_oom_runtime_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript_runner, "score_session_rows", raise_runtime_error)
    failures = tmp_path / "failed.csv"
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        transcript_runner.score_chunk(None, None, [ROW], "cuda", 1024, failures)
    assert not failures.exists()


def test_window_runner_reraises_non_oom_runtime_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(window_runner, "score_window_grid", raise_runtime_error)
    failures = tmp_path / "failed.csv"
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        window_runner.score_session_windows(None, None, ROW, [0], 1, "cuda", 1024, failures)
    assert not failures.exists()


def test_transcript_runner_logs_real_oom(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript_runner, "score_session_rows", raise_oom)
    monkeypatch.setattr(transcript_runner, "empty_device_cache", lambda device: None)
    failures = tmp_path / "failed.csv"
    assert transcript_runner.score_chunk(None, None, [ROW], "cuda", 1024, failures) == []
    frame = pd.read_csv(failures)
    assert frame.loc[0, "participant"] == 1
    assert "out of memory" in frame.loc[0, "error"].lower()


def test_window_runner_logs_real_oom(monkeypatch, tmp_path):
    monkeypatch.setattr(window_runner, "score_window_grid", raise_oom)
    monkeypatch.setattr(window_runner, "empty_device_cache", lambda device: None)
    failures = tmp_path / "failed.csv"
    assert (
        window_runner.score_session_windows(None, None, ROW, [0], 1, "cuda", 1024, failures) == []
    )
    frame = pd.read_csv(failures)
    assert frame.loc[0, "participant"] == 1
    assert "out of memory" in frame.loc[0, "error"].lower()


def test_transcript_runner_logs_context_overflow(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript_runner, "score_session_rows", raise_context_overflow)
    monkeypatch.setattr(transcript_runner, "empty_device_cache", lambda device: None)
    failures = tmp_path / "failed.csv"
    assert transcript_runner.score_chunk(None, None, [ROW], "cuda", 1024, failures) == []
    frame = pd.read_csv(failures)
    assert frame.loc[0, "participant"] == 1
    assert "exceeding the model context" in frame.loc[0, "error"]


def test_window_runner_logs_context_overflow(monkeypatch, tmp_path):
    monkeypatch.setattr(window_runner, "score_window_grid", raise_context_overflow)
    monkeypatch.setattr(window_runner, "empty_device_cache", lambda device: None)
    failures = tmp_path / "failed.csv"
    assert (
        window_runner.score_session_windows(None, None, ROW, [0], 1, "cuda", 1024, failures) == []
    )
    frame = pd.read_csv(failures)
    assert frame.loc[0, "participant"] == 1
    assert "exceeding the model context" in frame.loc[0, "error"]


def test_window_runner_deduplicates_equivalent_prompts(monkeypatch, tmp_path):
    captured = {}

    def fake_grid(model, tokenizer, segmented, cells, **kwargs):
        captured["cells"] = cells
        return [
            [{"target_index": target, "choice_index": target, "nll": 1.0, "num_tokens": 1}]
            for target, _ in cells
        ]

    monkeypatch.setattr(window_runner, "score_window_grid", fake_grid)
    records = window_runner.score_session_windows(
        None,
        None,
        THREE_TRIAL_ROW,
        [0, 1, 2, "full"],
        5,
        "cpu",
        1024,
        tmp_path / "failed.csv",
        "e0-informed",
    )

    assert captured["cells"] == [
        (0, 0),
        (1, 0),
        (2, 0),
        (1, 1),
        (2, 1),
        (2, 2),
    ]
    assert len(records) == 12
    assert {record["window"] for record in records} == {
        0,
        1,
        2,
        "full",
    }
    assert {record["position_grid"] for record in records} == {
        "e0-informed",
    }
    assert {record["target_fraction"] for record in records} == {
        0.0,
        0.5,
        1.0,
    }


def test_transcript_summary_separates_paper_and_macro_metrics():
    frame = pd.DataFrame(
        [
            {"experiment": "a", "participant": "1", "nll": 2.0, "num_tokens": 1},
            {"experiment": "a", "participant": "1", "nll": 4.0, "num_tokens": 2},
            {"experiment": "a", "participant": "2", "nll": 8.0, "num_tokens": 1},
        ]
    )
    summary = transcript_runner.summarize_scores(frame).loc["a"]
    assert summary["paper_token_nll"] == pytest.approx(3.5)
    assert summary["macro_choice_nll"] == pytest.approx(5.5)
    assert summary["choices"] == 3
    assert summary["choice_tokens"] == 4
    assert summary["participants"] == 2
