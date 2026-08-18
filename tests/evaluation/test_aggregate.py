from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mt.evaluation.aggregate import (
    BENCHMARK_EXCLUDED_EXPERIMENTS,
    drop_excluded,
    matched_probability_ratio,
    probability_ratio,
    replace_sessions,
    task_nll,
    task_probability,
    task_table,
)

def frame(rows):
    return pd.DataFrame(rows, columns=['experiment', 'participant',
                                       'choice_index', 'nll'])

def baseline_frame(rows):
    return pd.DataFrame(rows, columns=['experiment', 'participant',
                                       'choice_index', 'uniform'])

def test_task_nll_weights_participants_equally():
    # one participant with 1 choice, one with 3: the 3-choice participant
    # must not dominate just by making more choices
    scores = frame([
        ('a', 'p1', 0, 1.0),
        ('a', 'p2', 0, 0.0), ('a', 'p2', 1, 0.0), ('a', 'p2', 2, 0.0),
    ])
    assert task_nll(scores)['a'] == pytest.approx(0.5)

def test_task_nll_weights_experiments_separately():
    scores = frame([('a', 'p1', 0, 1.0), ('b', 'p1', 0, 3.0)])
    result = task_nll(scores)
    assert result['a'] == pytest.approx(1.0)
    assert result['b'] == pytest.approx(3.0)

def test_probability_is_exp_of_negative_nll():
    scores = frame([('a', 'p1', 0, 0.5), ('a', 'p1', 1, 1.5)])
    assert task_probability(scores)['a'] == pytest.approx(np.exp(-1.0))

def test_ratio_equals_exponentiated_nll_difference():
    model = frame([('a', 'p1', 0, 0.5)])
    baseline = frame([('a', 'p1', 0, 1.2)])
    assert probability_ratio(model, baseline)['a'] == pytest.approx(
        np.exp(1.2 - 0.5))

def test_ratio_above_one_means_model_beats_baseline():
    model = frame([('a', 'p1', 0, 0.5)])
    baseline = frame([('a', 'p1', 0, 1.2)])
    assert probability_ratio(model, baseline)['a'] > 1.0

def test_ratio_rejects_a_baseline_missing_experiments():
    model = frame([('a', 'p1', 0, 0.5), ('b', 'p1', 0, 0.5)])
    baseline = frame([('a', 'p1', 0, 1.0)])
    with pytest.raises(ValueError, match="missing 1 experiments"):
        probability_ratio(model, baseline)

def test_task_table_reports_the_counts_behind_each_number():
    scores = frame([
        ('a', 'p1', 0, 1.0),
        ('a', 'p2', 0, 0.0), ('a', 'p2', 1, 0.0), ('a', 'p2', 2, 0.0),
    ])
    row = task_table(scores).loc['a']
    assert row['n_participants'] == 2
    assert row['n_choices'] == 4
    assert row['task_nll'] == pytest.approx(0.5)
    assert row['task_probability'] == pytest.approx(np.exp(-0.5))

def test_matched_ratio_uses_only_the_choices_the_model_scored():
    # a windowed condition scores choice 1 alone; the unmatched aggregation
    # would average the baseline over both choices and give a different answer
    model = frame([('a', 'p1', 1, 0.5)])
    baseline = baseline_frame([
        ('a', 'p1', 0, 5.0),
        ('a', 'p1', 1, 1.2),
    ])
    ratio, coverage = matched_probability_ratio(model, baseline, 'uniform')
    assert ratio['a'] == pytest.approx(np.exp(1.2 - 0.5))
    assert ratio['a'] != pytest.approx(np.exp(3.1 - 0.5))
    assert coverage['a'] == pytest.approx(1.0)

def test_matched_ratio_drops_a_choice_with_no_baseline_row_and_says_so():
    model = frame([('a', 'p1', 0, 0.5), ('a', 'p1', 1, 0.5)])
    baseline = baseline_frame([('a', 'p1', 0, 1.0)])
    ratio, coverage = matched_probability_ratio(model, baseline, 'uniform')
    assert ratio['a'] == pytest.approx(np.exp(1.0 - 0.5))
    assert coverage['a'] == pytest.approx(0.5)

def test_matched_ratio_matches_participants_not_just_experiments():
    # the model lost p2 to OOM; the baseline still has it. Aggregating the two
    # sides independently would compare different participant sets.
    model = frame([('a', 'p1', 0, 1.0)])
    baseline = baseline_frame([('a', 'p1', 0, 2.0), ('a', 'p2', 0, 8.0)])
    ratio, _ = matched_probability_ratio(model, baseline, 'uniform')
    assert ratio['a'] == pytest.approx(np.exp(2.0 - 1.0))

def test_drop_excluded_removes_the_single_response_task():
    excluded = BENCHMARK_EXCLUDED_EXPERIMENTS[0]
    scores = frame([('a', 'p1', 0, 1.0), (excluded, 'p1', 0, 0.0)])
    assert drop_excluded(scores)['experiment'].tolist() == ['a']

def test_replace_sessions_swaps_only_the_named_sessions():
    source = frame([('a', 'p1', 0, 1.0), ('a', 'p2', 0, 2.0)])
    fixed = frame([('a', 'p1', 0, 9.0)])
    result = replace_sessions(source, fixed).set_index(
        ['experiment', 'participant'])['nll']
    assert result[('a', 'p1')] == 9.0
    assert result[('a', 'p2')] == 2.0
