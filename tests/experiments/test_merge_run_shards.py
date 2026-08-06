"""Tests for streaming, integrity-checked Track P shard merging."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))

from merge_run_shards import MergeError, merge_run  # noqa: E402

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
    return {**PARENT_KEY, "participant": participant, "choice_index": "0",
            "pred_choice": "A", "human_choice": "A"}


def topk(participant, rank, token=None):
    return {**PARENT_KEY, "participant": participant, "choice_index": "0",
            "token_index": "0", "rank": str(rank),
            "token": token or chr(ord("A") + rank), "logprob": "-0.1"}


def option(participant):
    return {**PARENT_KEY, "participant": participant, "choice_index": "0",
            "option": "A", "logprob": "-0.2", "n_tokens": "1",
            "is_human": "True", "is_correct": ""}


def complete_shards(run_dir, duplicate_parent=False):
    participants = ("1", "1" if duplicate_parent else "2")
    for shard, participant in enumerate(participants):
        write_csv(run_dir / f"predictions_shard{shard}.csv",
                  [prediction(participant)])
        write_csv(run_dir / f"pred_topk_shard{shard}.csv",
                  [topk(participant, 0), topk(participant, 1)])
        write_csv(run_dir / f"pred_options_shard{shard}.csv",
                  [option(participant)])


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
    }


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


def test_merge_run_rejects_nonempty_failure_log(tmp_path):
    complete_shards(tmp_path)
    write_csv(tmp_path / "predictions_shard0.failed.csv",
              [{"experiment": "task", "participant": "1",
                "error": "OOM"}])

    with pytest.raises(MergeError, match="failed sessions"):
        merge_run(tmp_path, 2)
