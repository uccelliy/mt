"""Tests for the canonical-space population baselines (E2-pop)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from mt.models.baselines import population as pop

def test_fit_counts_laplace_smoothing_marginal():
    index, log_marginal, _ = pop.fit_counts([["a", "a", "b"]], ["a", "b"])
    assert index == {'a': 0, 'b': 1}
    assert math.isclose(math.exp(log_marginal[0]), 3 / 5)  # (2+1)/(3+2)
    assert math.isclose(math.exp(log_marginal[1]), 2 / 5)  # (1+1)/(3+2)

def test_fit_counts_transitions_row_normalized():
    _, _, log_transition = pop.fit_counts([["a", "b", "a", "b"]],
                                          ["a", "b"])
    # from 'a': two observed a->b transitions, none a->a
    assert math.isclose(math.exp(log_transition[0, 1]), 3 / 4)
    assert math.isclose(math.exp(log_transition[0, 0]), 1 / 4)
    # rows are proper distributions
    assert math.isclose(sum(math.exp(v) for v in log_transition[0]), 1.0)

def test_unseen_option_gets_smoothed_mass_not_zero():
    _, log_marginal, _ = pop.fit_counts([["a", "a"]], ["a", "b"])
    assert math.exp(log_marginal[1]) == pytest.approx(1 / 4)

def test_score_codes_first_choice_falls_back_to_marginal():
    index, log_marginal, log_transition = pop.fit_counts(
        [["a", "b", "a", "b"]], ["a", "b"])
    records = pop.score_codes([index["a"], index["b"]],
                              log_marginal, log_transition,
                              uniform=math.log(2))

    # no predecessor at position 0, so bigram must equal the marginal there
    assert records[0]['pop_bigram'] == pytest.approx(records[0]['pop_base_rate'])
    # position 1 uses the a->b transition instead
    assert records[1]['pop_bigram'] == pytest.approx(-log_transition[0, 1])
    assert [r['choice_index'] for r in records] == [0, 1]
    assert all(r['canonical_uniform'] == pytest.approx(math.log(2)) for r in records)

def test_summarize_is_participant_then_task_macro():
    frame = pd.DataFrame([
        {'experiment': 'e', 'participant': '1', 'choice_index': 0,
         'pop_base_rate': 1.0, 'pop_bigram': 1.0, 'canonical_uniform': 0.7},
        {'experiment': 'e', 'participant': '1', 'choice_index': 1,
         'pop_base_rate': 3.0, 'pop_bigram': 3.0, 'canonical_uniform': 0.7},
        {'experiment': 'e', 'participant': '2', 'choice_index': 0,
         'pop_base_rate': 5.0, 'pop_bigram': 5.0, 'canonical_uniform': 0.7},
    ])
    summary = pop.summarize(frame).loc['e']
    # participant means (2.0 and 5.0) average to 3.5, not the flat 3.0
    assert summary['pop_base_rate'] == pytest.approx(3.5)
    assert summary['participants'] == 2
    assert summary['choices'] == 3
