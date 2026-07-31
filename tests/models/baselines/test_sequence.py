"""Tests for count-based sequence baselines."""

from __future__ import annotations

import math

from mt.evaluation.sequence_baselines import (
    SMOOTHING,
    extract_choices,
    fit_tables,
    score_sequence,
    score_sequence_online,
)

def test_extract_choices_returns_labels_in_order():
    assert extract_choices("a <<L>> b <<R>> c <<L>>") == ["L", "R", "L"]

def test_fit_tables_counts_unigrams_bigrams_and_repeats():
    tables = fit_tables([["A", "A", "B"], ["B"]], vocabulary=["A", "B"])
    assert tables.labels == ("A", "B")
    assert tables.unigram['A'] == 2 and tables.unigram['B'] == 2
    assert tables.bigram[('A', 'A')] == 1 and tables.bigram[('A', 'B')] == 1
    assert tables.n_transitions == 2 and tables.n_repeats == 1

def test_unigram_prob_matches_smoothing_formula():
    tables = fit_tables([["A", "A", "B"]], vocabulary=["A", "B"])
    expected = (2 + SMOOTHING) / (3 + SMOOTHING * 2)
    assert math.isclose(tables.unigram_prob("A"), expected)

def test_bigram_prob_conditions_on_previous_label():
    tables = fit_tables([["A", "A", "B"]], vocabulary=["A", "B"])
    expected = (1 + SMOOTHING) / (2 + SMOOTHING * 2)
    assert math.isclose(tables.bigram_prob("A", "B"), expected)
    # unseen previous label falls back to the uniform distribution
    assert math.isclose(tables.bigram_prob("B", "A"), 0.5)

def test_sticky_prob_splits_mass_between_repeat_and_switch():
    tables = fit_tables([["A", "A", "B"]], vocabulary=["A", "B"])
    theta = (1 + SMOOTHING) / (2 + 2 * SMOOTHING)
    assert math.isclose(tables.sticky_prob("A", "A"), theta)
    assert math.isclose(tables.sticky_prob("A", "B"), 1 - theta)

def test_score_sequence_first_trial_uses_base_rate():
    tables = fit_tables([["A", "A", "B"]], vocabulary=["A", "B"])
    records = score_sequence(tables, ["B", "B"])
    assert records[0]['choice_index'] == 0
    assert math.isclose(records[0]['uniform'], math.log(2))
    assert math.isclose(records[0]['sticky'], records[0]['base_rate'])
    assert math.isclose(records[0]['bigram'], records[0]['base_rate'])
    assert math.isclose(records[1]['sticky'],
                        -math.log(tables.sticky_prob("B", "B")))

def test_online_scores_match_hand_computed_probabilities():
    records = score_sequence_online(["A", "B", "B"])
    # t0: no history, base rate = 0.5/(0+1); sticky/bigram fall back
    assert math.isclose(records[0]['uniform'], math.log(2))
    assert math.isclose(records[0]['base_rate'], -math.log(0.5))
    assert math.isclose(records[0]['sticky'], -math.log(0.5))
    # t1: B unseen, base = 0.5/(1+1); theta = 0.5, switch = (1-0.5)/1
    assert math.isclose(records[1]['base_rate'], -math.log(0.25))
    assert math.isclose(records[1]['sticky'], -math.log(0.5))
    assert math.isclose(records[1]['bigram'], -math.log(0.5))
    # t2: repeat of B, theta = 0.5/(1+1) = 0.25; B->B transition unseen
    assert math.isclose(records[2]['base_rate'], -math.log(0.5))
    assert math.isclose(records[2]['sticky'], -math.log(0.25))
    assert math.isclose(records[2]['bigram'], -math.log(0.5))

def test_online_scoring_is_causal():
    prefix = score_sequence_online(["A", "B", "A"])
    longer = score_sequence_online(["A", "B", "A", "B", "B"])
    for a, b in zip(prefix, longer):
        assert a == b

def test_online_single_label_session_has_zero_nll_after_first():
    records = score_sequence_online(["A", "A", "A"])
    assert records[0]['sticky'] == 0.0 and records[2]['bigram'] == 0.0
    assert all(r['uniform'] == 0.0 for r in records)

def test_score_sequence_rejects_unknown_label():
    tables = fit_tables([["A"]], vocabulary=["A"])
    try:
        score_sequence(tables, ["Z"])
    except ValueError as error:
        assert "missing from vocabulary" in str(error)
    else:
        raise AssertionError("expected ValueError for unknown label")
