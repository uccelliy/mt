"""Tests for the score-CSV comparison tool."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "experiments"))

import compare_scoring  # noqa: E402

E0_ROWS = [
    {"experiment": "task", "participant": 1, "choice_index": 0, "nll": 0.5},
    {"experiment": "task", "participant": 1, "choice_index": 1, "nll": 1.5},
    {"experiment": "task", "participant": 2, "choice_index": 0, "nll": 2.5},
]


def write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def compare(tmp_path, reference_rows, candidate_rows):
    reference = compare_scoring.load_scores(
        write_csv(tmp_path / "ref.csv", reference_rows))
    candidate = compare_scoring.load_scores(
        write_csv(tmp_path / "cand.csv", candidate_rows))
    keys = compare_scoring.align_keys(reference, candidate)
    merged = reference.merge(candidate, on=list(keys), how="inner",
                             suffixes=("_ref", "_cand"))
    return compare_scoring.summarize(reference, candidate, merged, keys)


def test_identical_files_have_zero_delta(tmp_path):
    report = compare(tmp_path, E0_ROWS, E0_ROWS)
    assert report['matched_rows'] == 3
    assert report['max_delta'] == 0.0
    assert compare_scoring.judge(report, 1e-3, 0.25, 0.99, strict=True) == []


def test_delta_beyond_tolerance_fails(tmp_path):
    shifted = [dict(row, nll=row['nll'] + 0.01) for row in E0_ROWS]
    report = compare(tmp_path, E0_ROWS, shifted)
    assert report['max_delta'] == pytest.approx(0.01)
    assert compare_scoring.judge(report, 1e-3, 0.25, 0.99, strict=False)
    # a uniform shift is a bias, and shows up in the signed mean
    assert report['signed_mean'] == pytest.approx(0.01)
    assert compare_scoring.judge(report, 0.02, 0.25, 0.99, strict=False) == []


def test_worst_row_names_the_offending_key(tmp_path):
    candidate = [dict(row) for row in E0_ROWS]
    candidate[2]['nll'] = 9.0
    report = compare(tmp_path, E0_ROWS, candidate)
    assert report['worst']['participant'] == "2"


def test_subset_candidate_passes_unless_strict(tmp_path):
    subset = [row for row in E0_ROWS if row['participant'] == 1]
    report = compare(tmp_path, E0_ROWS, subset)
    assert report['matched_rows'] == 2
    assert len(report['only_reference']) == 1
    assert compare_scoring.judge(report, 1e-3, 0.25, 0.99, strict=False) == []
    assert compare_scoring.judge(report, 1e-3, 0.25, 0.99, strict=True)


def test_disjoint_sessions_fail_with_no_matches(tmp_path):
    other = [dict(row, experiment="other") for row in E0_ROWS]
    report = compare(tmp_path, E0_ROWS, other)
    assert report['matched_rows'] == 0
    assert compare_scoring.judge(report, 1e-3, 0.25, 0.99, strict=False)


def test_window_column_joins_e3_output(tmp_path):
    rows = [
        {"experiment": "task", "participant": 1, "window": w,
         "target_index": 0, "choice_index": 0, "nll": 0.5 + w}
        for w in (0, 1, 2)
    ]
    report = compare(tmp_path, rows, rows)
    assert "window" in report['keys']
    assert report['matched_rows'] == 3


def test_constant_series_reports_undefined_correlation(tmp_path):
    flat = [dict(row, nll=1.0) for row in E0_ROWS]
    report = compare(tmp_path, flat, flat)
    assert report['correlation'] is None
    # an undefined correlation must not by itself fail the comparison
    assert compare_scoring.judge(report, 1e-3, 0.25, 0.99, strict=False) == []


def test_zero_mean_row_noise_passes_on_the_aggregate(tmp_path):
    """The real L1 case: different GPU kernels, same science."""

    rng = random.Random(0)
    reference = [{"experiment": "task", "participant": i // 40,
                  "choice_index": i % 40, "nll": 0.5 + rng.random()}
                 for i in range(400)]
    # fp16 attention on Volta vs Blackwell disagrees per row by ~0.02 nat,
    # with no preferred sign, so experiment means still agree closely
    noisy = [dict(row, nll=row['nll'] + rng.gauss(0, 0.02))
             for row in reference]
    report = compare(tmp_path, reference, noisy)
    assert report['mean_delta'] > 1e-3
    assert report['experiment_delta']['max'] < 1e-2
    assert abs(report['signed_mean']) < 5e-3
    assert compare_scoring.judge(report, 1e-2, 0.25, 0.99,
                                 strict=False) == []


def test_wrong_model_fails_on_the_aggregate(tmp_path):
    """A checkpoint mix-up shifts every row the same way."""

    shifted = [dict(row, nll=row['nll'] + 0.17) for row in E0_ROWS]
    report = compare(tmp_path, E0_ROWS, shifted)
    assert report['experiment_delta']['max'] == pytest.approx(0.17)
    assert report['signed_mean'] == pytest.approx(0.17)
    assert compare_scoring.judge(report, 1e-2, 0.25, 0.99, strict=False)


def test_token_count_mismatch_fails(tmp_path):
    reference = [dict(row, num_tokens=1) for row in E0_ROWS]
    retokenized = [dict(row, num_tokens=2) for row in E0_ROWS]
    report = compare(tmp_path, reference, retokenized)
    assert report['token_mismatches'] == 3
    assert compare_scoring.judge(report, 1e-2, 0.25, 0.99, strict=False)


def test_participant_dtype_mismatch_still_joins(tmp_path):
    as_strings = [dict(row, participant=str(row['participant']))
                  for row in E0_ROWS]
    report = compare(tmp_path, E0_ROWS, as_strings)
    assert report['matched_rows'] == 3
