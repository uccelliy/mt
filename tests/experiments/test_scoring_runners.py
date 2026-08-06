"""Tests for experiment-runner failure isolation."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))

import _common as common  # noqa: E402
import run_transcript_scoring as transcript_runner  # noqa: E402
import run_window_scoring as window_runner  # noqa: E402

from mt.evaluation.transcript_scoring import (  # noqa: E402
    ChoiceScore,
    ContextLengthError,
    OptionScore,
    TokenCandidate,
)

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


def make_choice(choice_index, human_choice="A"):
    return ChoiceScore(
        choice_index=choice_index,
        nll=1.0,
        num_tokens=1,
        human_choice=human_choice,
        pred_choice="Z",
        pred_token_id=90,
        raw_generation="Z",
        format_ok=False,
        k_options=3,
        topk=(
            TokenCandidate(
                token_index=0,
                rank=0,
                token_id=90,
                token="Z",
                logprob=-0.1,
            ),
        ),
        options=(
            OptionScore(
                option=human_choice,
                logprob=-1.0,
                n_tokens=1,
                is_human=True,
            ),
        ),
        options_status="scored",
    )


def test_transcript_runner_reraises_non_oom_runtime_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript_runner, "score_session_rows", raise_runtime_error)
    failures = tmp_path / "failed.csv"
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        transcript_runner.score_chunk(None, None, [ROW], "cuda", 1024, 0, 0, failures)
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
    assert transcript_runner.score_chunk(None, None, [ROW], "cuda", 1024, 0, 0, failures) == []
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
    assert transcript_runner.score_chunk(None, None, [ROW], "cuda", 1024, 0, 0, failures) == []
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
    captured = {"full_calls": 0}

    def fake_full(model, tokenizer, text, **kwargs):
        captured["full_calls"] += 1
        return [make_choice(index, choice) for index, choice in enumerate("ABC")]

    def fake_grid(model, tokenizer, segmented, cells, **kwargs):
        captured["cells"] = cells
        captured["grid_kwargs"] = kwargs
        return [[make_choice(target, "ABC"[target])] for target, _ in cells]

    monkeypatch.setattr(window_runner, "score_marked_text", fake_full)
    monkeypatch.setattr(window_runner, "score_window_grid", fake_grid)
    observations = window_runner.score_session_windows(
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
    assert captured["full_calls"] == 1
    assert captured["grid_kwargs"]["top_k"] == 20
    assert captured["grid_kwargs"]["max_options"] == 256
    assert len(observations) == 12
    metadata = [meta for meta, _ in observations]
    assert {record["window"] for record in metadata} == {
        0,
        1,
        2,
        "full",
    }
    assert {record["condition"] for record in metadata} == {
        "full",
        "e3:w=0",
        "e3:w=1",
        "e3:w=2",
    }
    assert {record["position_grid"] for record in metadata} == {
        "e0-informed",
    }
    assert {record["target_fraction"] for record in metadata} == {
        0.0,
        0.5,
        1.0,
    }
    full = [(meta, score) for meta, score in observations if meta["condition"] == "full"]
    assert [score.choice_index for _, score in full] == [0, 1, 2]
    assert all(meta["is_anchor"] for meta, _ in full)


def test_window_runner_full_only_never_enters_window_grid(monkeypatch, tmp_path):
    calls = {"full": 0}

    def fake_full(*args, **kwargs):
        calls["full"] += 1
        return [make_choice(index, choice) for index, choice in enumerate("ABC")]

    def unexpected_grid(*args, **kwargs):
        raise AssertionError("full must not be represented as a window-grid cell")

    monkeypatch.setattr(window_runner, "score_marked_text", fake_full)
    monkeypatch.setattr(window_runner, "score_window_grid", unexpected_grid)
    observations = window_runner.score_session_windows(
        None,
        None,
        THREE_TRIAL_ROW,
        ["full"],
        5,
        "cpu",
        1024,
        tmp_path / "failed.csv",
        "e0-informed",
    )

    assert calls["full"] == 1
    assert len(observations) == 3
    assert {meta["condition"] for meta, _ in observations} == {"full"}


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


def test_transcript_table_paths_share_one_run_directory(tmp_path):
    plain = transcript_runner.table_paths(tmp_path, None)
    assert plain == {
        "predictions": tmp_path / "predictions.csv",
        "pred_topk": tmp_path / "pred_topk.csv",
        "pred_options": tmp_path / "pred_options.csv",
    }
    sharded = transcript_runner.table_paths(tmp_path, (2, 4))
    assert sharded == {
        "predictions": tmp_path / "predictions_shard2.csv",
        "pred_topk": tmp_path / "pred_topk_shard2.csv",
        "pred_options": tmp_path / "pred_options_shard2.csv",
    }


def test_transcript_runner_rejects_legacy_output_abbreviation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_transcript_scoring.py",
            "--model",
            "fake",
            "--data",
            "fake",
            "--output",
            "legacy.csv",
        ],
    )
    with pytest.raises(SystemExit) as error:
        transcript_runner.main()
    assert error.value.code == 2


def test_transcript_manifest_records_explicit_choice_readout(monkeypatch, tmp_path):
    monkeypatch.setattr(transcript_runner, "git_commit", lambda: "abc123")
    args = SimpleNamespace(
        adapter=None,
        data="prompts.jsonl",
        load="4bit",
        dtype="fp16",
        batch_tokens=8192,
        choice_readout="greedy-unconstrained-1token",
        top_k=20,
        max_options=32,
        shard=None,
    )
    run = {"model": "model", "dataset": "dataset", "condition": "full"}

    transcript_runner.write_manifest(tmp_path, args, run, None)

    manifest = json.loads((tmp_path / "run.json").read_text())
    assert manifest["choice_readout"] == "greedy-unconstrained-1token"


def test_transcript_child_tables_carry_the_complete_parent_key():
    run = {"model": "model", "dataset": "dataset", "condition": "full"}
    meta = {"experiment": "task", "participant": "1"}
    score = ChoiceScore(
        choice_index=0,
        nll=1.0,
        num_tokens=1,
        human_choice="A",
        pred_choice="Z",
        pred_token_id=90,
        raw_generation="Z",
        format_ok=False,
        k_options=2,
        topk=(TokenCandidate(token_index=0, rank=0, token_id=66, token="B", logprob=-0.1),),
        options=(OptionScore(option="A", logprob=-1.0, n_tokens=1, is_human=True),),
        options_status="scored",
    )
    predictions, topk, options = transcript_runner.shape_session(meta, [score], run)
    key_columns = ("model", "dataset", "condition", "experiment", "participant", "choice_index")
    expected = tuple(predictions[0][name] for name in key_columns)
    assert tuple(topk[0][name] for name in key_columns) == expected
    assert tuple(options[0][name] for name in key_columns) == expected
    assert predictions[0]["pred_choice"] == "Z"
    assert predictions[0]["pred_token_id"] == 90
    assert predictions[0]["raw_generation"] == "Z"
    assert predictions[0]["format_ok"] is False
    assert max(score.options, key=lambda option: option.logprob).option == "A"
    assert topk[0]["token_id"] == 66


def test_window_table_paths_share_one_run_directory(tmp_path):
    plain = window_runner.table_paths(tmp_path, None)
    assert plain == {
        "predictions": tmp_path / "predictions.csv",
        "pred_topk": tmp_path / "pred_topk.csv",
        "pred_options": tmp_path / "pred_options.csv",
    }
    sharded = window_runner.table_paths(tmp_path, (3, 4))
    assert sharded == {
        "predictions": tmp_path / "predictions_shard3.csv",
        "pred_topk": tmp_path / "pred_topk_shard3.csv",
        "pred_options": tmp_path / "pred_options_shard3.csv",
    }


@pytest.mark.parametrize(
    "partial",
    [b'2,"unterminated\nfield', b"2,unfinished"],
)
def test_window_resume_repairs_only_the_crash_truncated_csv_tail(tmp_path, partial):
    path = tmp_path / "predictions_shard0.csv"
    complete = b'a,b\n1,"line ""one""\nline two"\n'
    path.write_bytes(complete + partial)

    removed = window_runner.repair_incomplete_csv_tail(path, chunk_size=3)

    assert removed == len(partial)
    assert path.read_bytes() == complete
    assert pd.read_csv(path).to_dict("records") == [{"a": 1, "b": 'line "one"\nline two'}]


def test_window_resume_removes_an_incomplete_first_csv_write(tmp_path):
    path = tmp_path / "predictions_shard0.csv"
    path.write_bytes(b'model,dataset,"unfinished')

    assert window_runner.repair_incomplete_csv_tail(path) > 0
    assert not path.exists()


def test_window_resume_state_uses_only_atomic_session_commits(tmp_path):
    tables = window_runner.table_paths(tmp_path, (2, 4))
    commits = window_runner.session_commit_dir(tmp_path, (2, 4))
    # A valid-looking partial prediction append must not make the session done.
    pd.DataFrame([{"experiment": "task", "participant": "1"}]).to_csv(
        tables["predictions"], index=False
    )
    assert window_runner.read_session_commits(commits) == {}

    path = window_runner.publish_session_commit(
        commits,
        {"experiment": "task", "participant": "1"},
        [{"condition": "full"}, {"condition": "e3:w=0"}],
    )

    assert path.parent == commits
    assert not list(commits.glob(".*.tmp"))
    payload = window_runner.read_session_commits(commits)[("task", "1")]
    assert payload["status"] == "success"
    assert payload["prediction_rows"] == 2
    assert payload["conditions"] == {"e3:w=0": 1, "full": 1}


def test_window_write_session_publishes_commit_after_all_three_tables(tmp_path):
    tables = window_runner.table_paths(tmp_path, None)
    commits = window_runner.session_commit_dir(tmp_path, None)
    session = {"experiment": "task", "participant": "1"}
    run = {"model": "model", "dataset": "dataset"}
    common = {
        "target_index": 0,
        "effective_window": 0,
        "position_grid": "e0-informed",
        "n_segments": 1,
        "target_fraction": 0.0,
        "is_anchor": True,
    }
    observations = [
        (({"condition": "full", "window": "full"} | common), make_choice(0)),
        (({"condition": "e3:w=0", "window": 0} | common), make_choice(0)),
    ]

    window_runner.write_session(tables, session, observations, run, commits=commits)

    assert all(path.exists() for path in tables.values())
    marker = window_runner.read_session_commits(commits)[("task", "1")]
    assert marker["prediction_rows"] == 2
    assert marker["conditions"] == {"e3:w=0": 1, "full": 1}


def test_window_write_session_fsyncs_tables_before_publishing_commit(monkeypatch, tmp_path):
    tables = window_runner.table_paths(tmp_path, None)
    commits = window_runner.session_commit_dir(tmp_path, None)
    session = {"experiment": "task", "participant": "1"}
    run = {"model": "model", "dataset": "dataset"}
    metadata = {
        "condition": "full",
        "window": "full",
        "target_index": 0,
        "effective_window": 0,
        "position_grid": "e0-informed",
        "n_segments": 1,
        "target_fraction": 0.0,
        "is_anchor": True,
    }
    events = []

    monkeypatch.setattr(
        window_runner,
        "append_records",
        lambda path, records: events.append(("append", Path(path).name)),
    )
    monkeypatch.setattr(
        window_runner,
        "fsync_files",
        lambda paths: events.append(("fsync", tuple(Path(path).name for path in paths))),
    )
    monkeypatch.setattr(
        window_runner,
        "publish_session_commit",
        lambda directory, meta, predictions: events.append(("commit", len(predictions))),
    )

    window_runner.write_session(
        tables,
        session,
        [(metadata, make_choice(0))],
        run,
        commits=commits,
    )

    assert events == [
        ("append", "pred_topk.csv"),
        ("append", "pred_options.csv"),
        ("append", "predictions.csv"),
        ("fsync", ("predictions.csv", "pred_topk.csv", "pred_options.csv")),
        ("commit", 1),
    ]


@pytest.mark.skipif(os.name == "nt", reason="directory fsync is POSIX-only")
def test_fsync_files_persists_file_before_parent_directory(monkeypatch, tmp_path):
    path = tmp_path / "predictions.csv"
    path.write_text("header\nrow\n", encoding="utf-8")
    events = []

    def record_fsync(descriptor):
        mode = os.fstat(descriptor).st_mode
        events.append("directory" if stat.S_ISDIR(mode) else "file")

    monkeypatch.setattr(common.os, "fsync", record_fsync)

    common.fsync_files([path, tmp_path / "absent.csv"])

    assert events == ["file", "directory"]


def test_window_commit_fsyncs_directory_after_atomic_replace(monkeypatch, tmp_path):
    commits = window_runner.session_commit_dir(tmp_path, None)
    events = []
    real_replace = window_runner.os.replace

    def record_replace(source, destination):
        events.append(("replace", Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(window_runner.os, "replace", record_replace)
    monkeypatch.setattr(
        window_runner,
        "fsync_directory",
        lambda path: events.append(("fsync_directory", Path(path))),
    )

    marker = window_runner.publish_session_commit(
        commits,
        {"experiment": "task", "participant": "1"},
        [{"condition": "full"}],
    )

    replace_index = events.index(("replace", marker))
    assert events[replace_index + 1] == ("fsync_directory", commits)


def test_window_child_tables_use_unique_standard_parent_keys():
    run = {"model": "model", "dataset": "dataset"}
    session = {"experiment": "task", "participant": "1"}
    score = make_choice(0)
    common = {
        "target_index": 0,
        "effective_window": 0,
        "position_grid": "e0-informed",
        "n_segments": 1,
        "target_fraction": 0.0,
        "is_anchor": True,
    }
    observations = [
        (({"condition": "full", "window": "full"} | common), score),
        (({"condition": "e3:w=0", "window": 0} | common), score),
    ]

    predictions, topk, options = window_runner.shape_session(session, observations, run)

    parent_key = (
        "model",
        "dataset",
        "condition",
        "experiment",
        "participant",
        "choice_index",
    )
    prediction_keys = [tuple(row[column] for column in parent_key) for row in predictions]
    assert len(prediction_keys) == len(set(prediction_keys)) == 2
    assert [tuple(row[column] for column in parent_key) for row in topk] == prediction_keys
    assert [tuple(row[column] for column in parent_key) for row in options] == prediction_keys
    for child in [*topk, *options]:
        assert child["target_index"] == 0
        assert child["position_grid"] == "e0-informed"
        assert child["is_anchor"] is True
    assert predictions[0]["pred_choice"] == "Z"
    assert predictions[0]["pred_token_id"] == 90
    assert predictions[0]["raw_generation"] == "Z"


def test_window_manifest_records_full_three_table_protocol(monkeypatch, tmp_path):
    monkeypatch.setattr(window_runner, "git_commit", lambda: "abc123")
    args = SimpleNamespace(
        adapter="adapter",
        data="prompts.jsonl",
        condition_prefix="e3",
        windows=[0, 1, "full"],
        num_positions=5,
        position_grid="e0-informed",
        load="4bit",
        dtype="fp16",
        batch_tokens=8192,
        choice_readout="greedy-unconstrained-1token",
        top_k=20,
        max_options=256,
        shard="2/4",
    )
    run = {"model": "base+adapter", "dataset": "dataset"}

    window_runner.write_manifest(tmp_path, args, run, (2, 4))

    manifest = json.loads((tmp_path / "run_shard2.json").read_text())
    assert manifest["condition"] == "e3"
    assert manifest["full_condition"] == "full"
    assert manifest["windows"] == [0, 1, "full"]
    assert manifest["segmentation_protocol"] == "last-pre-marker-blank-v1"
    assert manifest["window_unit"] == "marked-choice-segment"
    assert manifest["choice_readout"] == "greedy-unconstrained-1token"
    assert manifest["top_k"] == 20
    assert manifest["max_options"] == 256
    assert manifest["commit"] == "abc123"


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("0,0", "duplicate window"),
        ("-1,full", "non-negative"),
        ("0,,full", "empty item"),
        ("short,full", "non-negative integer"),
    ],
)
def test_window_list_rejects_ambiguous_protocols(value, message):
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        window_runner.parse_windows(value)
