"""Tests for streaming, integrity-checked Track P shard merging."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))

from merge_run_shards import MergeError, _session_commit_name, merge_run  # noqa: E402

PARENT_KEY = {
    "model": "model",
    "dataset": "dataset",
    "condition": "full",
    "experiment": "task",
}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prediction(participant):
    return {
        **PARENT_KEY,
        "participant": participant,
        "choice_index": "0",
        "pred_choice": "A",
        "pred_token_id": "65",
        "raw_generation": "A",
        "format_ok": "True",
        "human_choice": "A",
        "num_tokens": "1",
        "k_options": "1",
        "options_status": "scored",
    }


def topk(participant, rank, token=None):
    return {
        **PARENT_KEY,
        "participant": participant,
        "choice_index": "0",
        "token_index": "0",
        "rank": str(rank),
        "token_id": str(65 + rank),
        "token": token or chr(ord("A") + rank),
        "logprob": "-0.1",
    }


def option(participant, value="A", *, is_human=None):
    if is_human is None:
        is_human = value == "A"
    return {
        **PARENT_KEY,
        "participant": participant,
        "choice_index": "0",
        "option": value,
        "logprob": "-0.2",
        "n_tokens": "1",
        "is_human": str(is_human),
        "is_correct": "",
    }


def window_metadata(target, window, *, n_segments=1, is_anchor=True):
    effective = target if window == "full" else min(target, int(window))
    return {
        "window": str(window),
        "target_index": str(target),
        "effective_window": str(effective),
        "position_grid": "e0-informed",
        "n_segments": str(n_segments),
        "target_fraction": str(target / max(n_segments - 1, 1)),
        "is_anchor": str(is_anchor),
    }


def write_commit(run_dir, shard, participant, conditions, *, experiment="task"):
    directory = run_dir / f"session_commits_shard{shard}"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "experiment": experiment,
        "participant": str(participant),
        "status": "success",
        "prediction_rows": sum(conditions.values()),
        "conditions": conditions,
    }
    path = directory / _session_commit_name(experiment, participant)
    path.write_text(json.dumps(payload), encoding="utf-8")


def complete_shards(run_dir, duplicate_parent=False):
    (run_dir / "launcher.json").write_text(json.dumps({"top_k": 2, "num_shards": 2}))
    participants = ("1", "1" if duplicate_parent else "2")
    for shard, participant in enumerate(participants):
        write_csv(run_dir / f"predictions_shard{shard}.csv", [prediction(participant)])
        write_csv(
            run_dir / f"pred_topk_shard{shard}.csv", [topk(participant, 0), topk(participant, 1)]
        )
        write_csv(run_dir / f"pred_options_shard{shard}.csv", [option(participant)])


def complete_e3_shard(run_dir, *, finite_choices=(0, 1, 2)):
    (run_dir / "launcher.json").write_text(
        json.dumps(
            {
                "top_k": 2,
                "num_shards": 1,
                "suite": "e3",
                "condition": "e3",
                "windows": [0, "full"],
                "num_positions": 5,
                "position_grid": "e0-informed",
                "full_condition": "full",
            }
        )
    )
    predictions = []
    topk_rows = []
    option_rows = []
    for condition, window, choices in (
        ("full", "full", range(3)),
        ("e3:w=0", "0", finite_choices),
    ):
        for choice_index in choices:
            extra = {
                "condition": condition,
                "choice_index": str(choice_index),
                **window_metadata(choice_index, window, n_segments=3),
            }
            predictions.append({**prediction("1"), **extra})
            topk_rows.extend({**topk("1", rank), **extra} for rank in range(2))
            option_rows.append({**option("1"), **extra})
    write_csv(run_dir / "predictions_shard0.csv", predictions)
    write_csv(run_dir / "pred_topk_shard0.csv", topk_rows)
    write_csv(run_dir / "pred_options_shard0.csv", option_rows)
    write_commit(
        run_dir,
        0,
        "1",
        {"full": 3, "e3:w=0": len(tuple(finite_choices))},
    )


def test_merge_run_streams_three_tables_and_cleans_child_replays(tmp_path):
    complete_shards(tmp_path)
    shard = tmp_path / "pred_topk_shard0.csv"
    rows = pd.read_csv(shard, dtype=str).to_dict("records")
    rows += [topk("1", 0), topk("1", 1), topk("orphan", 0)]
    write_csv(shard, rows)

    stats = merge_run(tmp_path, 2)

    assert len(pd.read_csv(tmp_path / "predictions.csv")) == 2
    assert len(pd.read_csv(tmp_path / "pred_topk.csv")) == 4
    assert len(pd.read_csv(tmp_path / "pred_options.csv")) == 2
    assert stats["pred_topk"] == {
        "rows": 4,
        "orphan_rows_dropped": 1,
        "duplicate_rows_dropped": 2,
        "incomplete_rows_dropped": 0,
    }


def test_merge_run_treats_e3_conditions_as_distinct_parent_rows(tmp_path):
    (tmp_path / "launcher.json").write_text(
        json.dumps(
            {
                "top_k": 2,
                "num_shards": 1,
                "suite": "e3",
                "condition": "e3",
                "windows": [0, "full"],
                "num_positions": 5,
                "position_grid": "e0-informed",
                "full_condition": "full",
            }
        )
    )
    predictions = []
    topk_rows = []
    option_rows = []
    for condition, window in (("full", "full"), ("e3:w=0", "0")):
        extra = {"condition": condition, **window_metadata(0, window)}
        predictions.append({**prediction("1"), **extra})
        topk_rows.extend({**topk("1", rank), **extra} for rank in range(2))
        option_rows.append({**option("1"), **extra})
    write_csv(tmp_path / "predictions_shard0.csv", predictions)
    write_csv(tmp_path / "pred_topk_shard0.csv", topk_rows)
    write_csv(tmp_path / "pred_options_shard0.csv", option_rows)
    write_commit(tmp_path, 0, "1", {"full": 1, "e3:w=0": 1})

    merge_run(tmp_path, 1)

    merged = pd.read_csv(tmp_path / "predictions.csv", dtype=str)
    assert merged["condition"].tolist() == ["full", "e3:w=0"]
    assert len(pd.read_csv(tmp_path / "pred_topk.csv")) == 4
    assert len(pd.read_csv(tmp_path / "pred_options.csv")) == 2


def test_merge_run_rejects_incomplete_e3_anchor_coverage(tmp_path):
    complete_e3_shard(tmp_path, finite_choices=(0, 1))

    with pytest.raises(MergeError, match="incomplete e3:w=0 anchor coverage"):
        merge_run(tmp_path, 1)


def test_merge_run_rejects_e3_predictions_without_atomic_commit(tmp_path):
    complete_e3_shard(tmp_path)
    marker = next((tmp_path / "session_commits_shard0").glob("*.json"))
    marker.unlink()

    with pytest.raises(MergeError, match="session commits disagree with predictions"):
        merge_run(tmp_path, 1)


def test_merge_run_rejects_child_e3_metadata_mismatch(tmp_path):
    complete_e3_shard(tmp_path)
    path = tmp_path / "pred_topk_shard0.csv"
    rows = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
    rows[0]["target_fraction"] = "0.25"
    write_csv(path, rows)

    with pytest.raises(MergeError, match="child E3 metadata differs"):
        merge_run(tmp_path, 1)


def test_merge_run_recovers_e3_partial_append_then_deterministic_replay(tmp_path):
    complete_e3_shard(tmp_path)
    prediction_path = tmp_path / "predictions_shard0.csv"
    topk_path = tmp_path / "pred_topk_shard0.csv"
    option_path = tmp_path / "pred_options_shard0.csv"
    predictions = pd.read_csv(prediction_path, dtype=str, keep_default_na=False).to_dict("records")
    topk_rows = pd.read_csv(topk_path, dtype=str, keep_default_na=False).to_dict("records")
    option_rows = pd.read_csv(option_path, dtype=str, keep_default_na=False).to_dict("records")
    # Each prefix is a possible append interrupted before the atomic marker;
    # the following complete occurrence is the deterministic resume replay.
    write_csv(prediction_path, predictions[:4] + predictions)
    write_csv(topk_path, topk_rows[:1] + topk_rows)
    write_csv(option_path, option_rows[:1] + option_rows)

    stats = merge_run(tmp_path, 1)

    assert len(pd.read_csv(tmp_path / "predictions.csv")) == 6
    assert len(pd.read_csv(tmp_path / "pred_topk.csv")) == 12
    assert len(pd.read_csv(tmp_path / "pred_options.csv")) == 6
    assert stats["predictions"]["duplicate_rows_dropped"] == 4
    assert stats["pred_topk"]["incomplete_rows_dropped"] == 1


def test_merge_run_keeps_complete_replay_after_partial_child_group(tmp_path):
    complete_shards(tmp_path)
    shard = tmp_path / "pred_topk_shard0.csv"
    # Simulate a crash during the first child append, followed by a resume
    # that appends a complete occurrence for the same prediction.
    write_csv(shard, [topk("1", 0), topk("1", 0), topk("1", 1)])

    stats = merge_run(tmp_path, 2)

    merged = pd.read_csv(tmp_path / "pred_topk.csv", dtype=str)
    participant = merged[merged["participant"] == "1"]
    assert participant["rank"].tolist() == ["0", "1"]
    assert stats["pred_topk"]["incomplete_rows_dropped"] == 1
    assert stats["pred_topk"]["duplicate_rows_dropped"] == 0


def test_merge_run_rejects_topk_count_with_out_of_range_suffix(tmp_path):
    complete_shards(tmp_path)
    write_csv(
        tmp_path / "pred_topk_shard0.csv",
        [topk("1", 0), topk("1", 2)],
    )

    with pytest.raises(MergeError, match="no complete child occurrence"):
        merge_run(tmp_path, 2)


def test_merge_run_keeps_complete_options_replay_after_partial_group(tmp_path):
    complete_shards(tmp_path)
    prediction_path = tmp_path / "predictions_shard0.csv"
    predictions = pd.read_csv(prediction_path, dtype=str).to_dict("records")
    predictions[0]["k_options"] = "2"
    write_csv(prediction_path, predictions)
    write_csv(
        tmp_path / "pred_options_shard0.csv",
        [option("1", "A"), option("1", "A"), option("1", "B")],
    )

    stats = merge_run(tmp_path, 2)

    merged = pd.read_csv(tmp_path / "pred_options.csv", dtype=str)
    participant = merged[merged["participant"] == "1"]
    assert participant["option"].tolist() == ["A", "B"]
    assert stats["pred_options"]["incomplete_rows_dropped"] == 1


def test_merge_run_rejects_parent_with_only_partial_topk(tmp_path):
    complete_shards(tmp_path)
    write_csv(tmp_path / "pred_topk_shard0.csv", [topk("1", 0)])

    with pytest.raises(MergeError, match="no complete child occurrence"):
        merge_run(tmp_path, 2)

    assert not (tmp_path / "predictions.csv").exists()
    assert not (tmp_path / "pred_topk.csv").exists()


def test_merge_run_requires_zero_options_when_not_scored(tmp_path):
    complete_shards(tmp_path)
    path = tmp_path / "predictions_shard0.csv"
    rows = pd.read_csv(path, dtype=str).to_dict("records")
    rows[0]["options_status"] = "multi_token_option"
    write_csv(path, rows)

    with pytest.raises(MergeError, match="expects exactly 0 rows"):
        merge_run(tmp_path, 2)


def test_merge_run_accepts_multi_token_skip_with_no_option_rows(tmp_path):
    complete_shards(tmp_path)
    prediction_path = tmp_path / "predictions_shard0.csv"
    predictions = pd.read_csv(prediction_path, dtype=str).to_dict("records")
    predictions[0]["options_status"] = "multi_token_option"
    write_csv(prediction_path, predictions)

    option_path = tmp_path / "pred_options_shard0.csv"
    columns = pd.read_csv(option_path, dtype=str).columns
    pd.DataFrame(columns=columns).to_csv(option_path, index=False)

    merge_run(tmp_path, 2)

    merged = pd.read_csv(tmp_path / "pred_options.csv", dtype=str)
    assert merged["participant"].tolist() == ["2"]


def test_merge_run_rejects_unknown_options_status(tmp_path):
    complete_shards(tmp_path)
    path = tmp_path / "predictions_shard0.csv"
    rows = pd.read_csv(path, dtype=str).to_dict("records")
    rows[0]["options_status"] = "multi_token"
    write_csv(path, rows)

    with pytest.raises(MergeError, match="unknown options_status.*multi_token"):
        merge_run(tmp_path, 2)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"n_tokens": "2"}, "must have n_tokens=1"),
        ({"is_human": "False"}, "must mark exactly human_choice"),
        ({"option": "B"}, "must mark exactly human_choice"),
        ({"logprob": "nan"}, "logprob must be finite and non-positive"),
        ({"logprob": "0.1"}, "logprob must be finite and non-positive"),
    ],
)
def test_merge_run_rejects_invalid_scored_option_payload(tmp_path, changes, message):
    complete_shards(tmp_path)
    path = tmp_path / "pred_options_shard0.csv"
    rows = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
    rows[0].update(changes)
    write_csv(path, rows)

    with pytest.raises(MergeError, match=message):
        merge_run(tmp_path, 2)


def test_merge_run_rejects_inconsistent_topk_metadata(tmp_path):
    complete_shards(tmp_path)
    (tmp_path / "run_shard0.json").write_text(json.dumps({"top_k": 2}))
    (tmp_path / "run_shard1.json").write_text(json.dumps({"top_k": 3}))

    with pytest.raises(MergeError, match="inconsistent top_k metadata"):
        merge_run(tmp_path, 2)


@pytest.mark.parametrize(
    ("table", "column"),
    [
        ("predictions", "pred_choice"),
        ("predictions", "pred_token_id"),
        ("predictions", "human_choice"),
        ("predictions", "raw_generation"),
        ("predictions", "format_ok"),
        ("pred_topk", "token_id"),
        ("pred_topk", "token"),
        ("pred_topk", "logprob"),
        ("pred_options", "option"),
        ("pred_options", "logprob"),
        ("pred_options", "n_tokens"),
        ("pred_options", "is_human"),
        ("pred_options", "is_correct"),
    ],
)
def test_merge_run_rejects_legacy_shard_missing_raw_schema(tmp_path, table, column):
    complete_shards(tmp_path)
    path = tmp_path / f"{table}_shard0.csv"
    rows = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
    for row in rows:
        row.pop(column)
    write_csv(path, rows)

    with pytest.raises(MergeError, match=rf"missing columns.*{column}"):
        merge_run(tmp_path, 2)

    assert not (tmp_path / "predictions.csv").exists()
    assert not (tmp_path / "pred_topk.csv").exists()
    assert not (tmp_path / "pred_options.csv").exists()


def test_merge_run_rejects_missing_table_shard_before_publish(tmp_path):
    complete_shards(tmp_path)
    (tmp_path / "pred_options_shard1.csv").unlink()

    with pytest.raises(MergeError, match="missing shards"):
        merge_run(tmp_path, 2)
    assert not (tmp_path / "predictions.csv").exists()
    assert not (tmp_path / "pred_topk.csv").exists()
    assert not (tmp_path / "pred_options.csv").exists()


def test_merge_run_rejects_duplicate_parent_keys(tmp_path):
    complete_shards(tmp_path, duplicate_parent=True)

    with pytest.raises(MergeError, match="duplicate prediction key"):
        merge_run(tmp_path, 2)
    assert not (tmp_path / "predictions.csv").exists()


def test_merge_run_deduplicates_identical_prediction_replay_within_shard(tmp_path):
    complete_shards(tmp_path)
    path = tmp_path / "predictions_shard0.csv"
    rows = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
    write_csv(path, rows + rows)

    stats = merge_run(tmp_path, 2)

    assert len(pd.read_csv(tmp_path / "predictions.csv")) == 2
    assert stats["predictions"]["duplicate_rows_dropped"] == 1


def test_merge_run_rejects_nonidentical_prediction_replay_within_shard(tmp_path):
    complete_shards(tmp_path)
    path = tmp_path / "predictions_shard0.csv"
    rows = pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")
    changed = {**rows[0], "pred_choice": "B"}
    write_csv(path, rows + [changed])

    with pytest.raises(MergeError, match="non-identical replay"):
        merge_run(tmp_path, 2)


def test_merge_run_rejects_nonempty_failure_log(tmp_path):
    complete_shards(tmp_path)
    write_csv(
        tmp_path / "predictions_shard0.failed.csv",
        [{"experiment": "task", "participant": "1", "error": "OOM"}],
    )

    with pytest.raises(MergeError, match="failed sessions"):
        merge_run(tmp_path, 2)
