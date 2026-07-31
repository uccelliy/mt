"""Population count baselines in canonical choice space (E2-pop).

The transcript-space baselines in :mod:`mt.models.baselines.sequence` cannot
pool across participants, because Psych-101 randomizes response keys per
participant. The original HF tables code choices canonically, so
cross-participant counting becomes meaningful there.

This module owns the counting and scoring math only. Loading tables, aligning
participants against transcripts, and writing CSVs stay in the runner.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POPULATION_BASELINES = ('pop_base_rate', 'pop_bigram', 'canonical_uniform')

def fit_counts(sequences, support):
    """Return Laplace-smoothed log marginal and transition matrices.

    ``sequences`` are the training participants' canonical choice lists and
    ``support`` is the task's full option inventory, so an option never chosen
    in training still receives smoothed mass rather than zero.
    """

    index = {value: i for i, value in enumerate(support)}
    k = len(support)
    marginal = np.zeros(k)
    transitions = np.zeros((k, k))
    for values in sequences:
        codes = [index[v] for v in values]
        for code in codes:
            marginal[code] += 1
        for previous, current in zip(codes, codes[1:]):
            transitions[previous, current] += 1
    log_marginal = np.log(marginal + 1) - np.log(marginal.sum() + k)
    log_transition = (np.log(transitions + 1)
                      - np.log(transitions.sum(axis=1, keepdims=True) + k))
    return index, log_marginal, log_transition

def score_codes(codes, log_marginal, log_transition, uniform) -> list[dict[str, Any]]:
    """Return per-choice NLLs of each population baseline for one participant.

    The first choice of a session has no predecessor, so the bigram model
    falls back to the marginal there.
    """

    records = []
    for position, code in enumerate(codes):
        if position == 0:
            bigram = -log_marginal[code]
        else:
            bigram = -log_transition[codes[position - 1], code]
        records.append({'choice_index': position,
                        'pop_base_rate': -log_marginal[code],
                        'pop_bigram': bigram,
                        'canonical_uniform': uniform})
    return records

def summarize(frame) -> pd.DataFrame:
    """Choice -> participant -> experiment macro means per baseline."""

    per_participant = frame.groupby(['experiment', 'participant'])[
        list(POPULATION_BASELINES)].mean()
    summary = per_participant.groupby('experiment').mean()
    summary['participants'] = per_participant.groupby('experiment').size()
    summary['choices'] = frame.groupby('experiment').size()
    return summary
