"""Tests for paper-protocol NLL cache conversion."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"
                       / "experiments"))

from build_paper_style_nll import (  # noqa: E402
    choice_token_counts,
    paper_task,
    select_expected_scores,
    summarize_paper_scores,
)


class CharTokenizer:
    """Character tokenizer suitable for cutoff bookkeeping tests."""

    def __call__(self, text, add_special_tokens, truncation,
                 return_offsets_mapping, max_length=None):
        ids = list(range(len(text)))
        offsets = [(index, index + 1) for index in range(len(text))]
        if add_special_tokens:
            ids = [999] + ids
            offsets = [(0, 0)] + offsets
        if truncation:
            ids = ids[:max_length]
            offsets = offsets[:max_length]
        return {"input_ids": ids, "offset_mapping": offsets}


def test_paper_task_uses_official_prefixes():
    assert paper_task("xiong2023neural/exp1.csv") == "xiong2023neural"
    assert paper_task("not-in-paper/exp.csv") is None


def test_choice_counts_head_truncation_drops_tail_answers():
    full, truncated = choice_token_counts("<<a>> x <<b>>", CharTokenizer(),
                                           max_tokens=8)
    assert full == [1, 1]
    assert truncated == [1, 0]


def test_select_expected_scores_and_session_mean_summary():
    expected = pd.DataFrame([
        {"paper_task": "task", "experiment": "task/exp.csv",
         "participant": "1", "choice_index": 0,
         "expected_num_tokens": 1},
        {"paper_task": "task", "experiment": "task/exp.csv",
         "participant": "1", "choice_index": 1,
         "expected_num_tokens": 1},
        {"paper_task": "task", "experiment": "task/exp.csv",
         "participant": "2", "choice_index": 0,
         "expected_num_tokens": 4},
    ])
    scores = pd.DataFrame([
        {"experiment": "task/exp.csv", "participant": "1",
         "choice_index": 0, "nll": 2.0, "num_tokens": 1},
        {"experiment": "task/exp.csv", "participant": "1",
         "choice_index": 1, "nll": 4.0, "num_tokens": 1},
        {"experiment": "task/exp.csv", "participant": "2",
         "choice_index": 0, "nll": 16.0, "num_tokens": 4},
    ])
    result = select_expected_scores(scores, expected)
    summary = summarize_paper_scores(result).loc["task"]
    assert summary["official_eval_loss"] == pytest.approx(3.5)
    assert summary["token_micro_nll"] == pytest.approx(22 / 6)
    assert summary["n_sessions"] == 2
