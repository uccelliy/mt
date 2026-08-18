"""The benchmark's aggregation rule and the three metrics derived from it.

`docs/benchmark design.md` sections 3 and 4 define one aggregation and three
metrics layered on it.  Every analysis needs them, so they live here rather
than being re-derived per script.

Aggregation (section 4), three levels:

    c_j     = sum of the token NLLs inside one choice
            = -log P(the exact string the human produced)
    l_i     = mean of c_j over the choices of one session
    L_f     = mean of l_i over the participants of one experiment

The first level is a **sum**, not a mean: by the chain rule the token NLLs of
one choice add up to the negative log probability of the whole choice string,
so exp(-c_j) is literally the probability the model gave the human's answer.
Averaging there would instead yield a per-token geometric mean, a quantity no
event has the probability of, and one that shifts with how the tokenizer
happens to split the answer.

The scorer already writes `nll` as that per-choice sum, so level one is
already done in the stored data and `task_nll` starts at level two.

Metrics (section 3):

    L_f                                   nat, checks against published numbers
    p_f = exp(-L_f)                       typical probability, readable alone
    R_f = p_model / p_baseline            "how many times better than counting"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ['experiment', 'participant']
CHOICE_KEYS = ['experiment', 'participant', 'choice_index']

# Design section 9 clause 7. The legal response set has one element, so every
# model necessarily "guesses right" and the task carries no signal. This is an
# analysis-time exclusion: the sessions are scored and stored like any other,
# they just do not enter a cross-model comparison.
BENCHMARK_EXCLUDED_EXPERIMENTS = ('enkavi2019gonogo/exp1.csv',)

def load_scores(path):
    """Read a score CSV, normalizing the identifiers that early runs mangled."""

    frame = pd.read_csv(path, low_memory=False)
    # CP936 mojibake left by the Windows runs; the scores themselves are fine
    frame['experiment'] = frame['experiment'].str.replace(
        'collsi枚枚', 'collsiöö', regex=False)
    frame['participant'] = frame['participant'].astype(str)
    return frame

def replace_sessions(source, replacement):
    """Swap in re-scored sessions, matching on experiment and participant."""

    replaced = set(map(tuple, replacement[KEYS].drop_duplicates()
                       .itertuples(index=False)))
    keep = ~source.set_index(KEYS).index.isin(replaced)
    return pd.concat([source[keep], replacement], ignore_index=True)

def task_nll(frame, value='nll'):
    """L_f per experiment: mean over choices, then mean over participants.

    `value` must already be per choice — for the scorer's output that is the
    `nll` column, which sums the tokens of the choice (see module docstring).
    """

    per_participant = frame.groupby(KEYS)[value].mean()
    return per_participant.groupby('experiment').mean()

def task_probability(frame, value='nll'):
    """p_f per experiment: the typical probability of the human's answer."""

    return np.exp(-task_nll(frame, value))

def probability_ratio(model_frame, baseline_frame, value='nll',
                      baseline_value=None):
    """R_f per experiment: how many times likelier than the baseline.

    Equivalently exp(L_baseline - L_model); the ratio of probabilities is the
    exponentiated difference of NLLs, which is what makes a 0.01 nat gap
    readable as "1% more probability".

    Both sides must be aggregated identically, or the ratio means nothing.
    Counting baselines predict a choice and have no tokens, so level one of
    the aggregation is a no-op for them and the two sides line up directly.
    """

    model = task_nll(model_frame, value)
    baseline = task_nll(baseline_frame, baseline_value or value)
    shared = model.index.intersection(baseline.index)
    if len(shared) < len(model.index):
        missing = sorted(set(model.index) - set(shared))
        raise ValueError(f"baseline is missing {len(missing)} experiments: "
                         f"{missing[:5]}")
    return np.exp(baseline[shared] - model[shared])

def task_table(frame, value='nll'):
    """Per-experiment L_f and p_f, with the counts they were computed from.

    The counts are not decoration: L_f averages participants equally, so two
    models are only comparable on a task when they scored the same sessions.
    """

    per_participant = frame.groupby(KEYS)[value].agg(['mean', 'size'])
    grouped = per_participant.groupby('experiment')
    table = pd.DataFrame({
        'n_participants': grouped.size(),
        'n_choices': grouped['size'].sum(),
        'task_nll': grouped['mean'].mean(),
    })
    table['task_probability'] = np.exp(-table['task_nll'])
    return table

def matched_probability_ratio(model_frame, baseline_frame, baseline_value,
                              value='nll'):
    """R_f computed on exactly the choices the model scored.

    `probability_ratio` aggregates the two sides independently, which is right
    when both cover the same choices and quietly wrong when they do not -- a
    model that lost sessions to OOM, or a truncated-window condition that only
    scores one segment, would otherwise be compared against a baseline average
    taken over a different set. Joining per choice first makes the requirement
    that both sides use one aggregation (section 3.3) hold by construction.

    Choices present on only one side are dropped -- the ratio is taken on the
    intersection -- and the per-task coverage comes back with it, so a
    shrinking denominator cannot hide.

    Returns (ratio, coverage), both indexed by experiment.
    """

    merged = model_frame.merge(
        baseline_frame[CHOICE_KEYS + [baseline_value]],
        on=CHOICE_KEYS, how='inner', validate='one_to_one')
    ratio = np.exp(task_nll(merged, baseline_value) - task_nll(merged, value))
    scored = model_frame.groupby('experiment').size()
    coverage = (merged.groupby('experiment').size() / scored).reindex(
        ratio.index).fillna(0.0)
    return ratio, coverage.rename('coverage')

def drop_excluded(frame, excluded=BENCHMARK_EXCLUDED_EXPERIMENTS):
    """Remove the experiments the design excludes from model comparison."""

    return frame[~frame['experiment'].isin(excluded)]
