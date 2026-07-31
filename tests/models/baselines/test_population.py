"""Tests for the canonical-space population baselines (E2-pop)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"
                       / "experiments"))

import run_population_baselines as pop  # noqa: E402

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
