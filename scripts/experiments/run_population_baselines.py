"""Population count baselines in canonical choice space (E2-pop).

The transcript-space E2 baselines cannot pool across participants because
Psych-101 randomizes response keys per participant; the original HF
tables code choices canonically, so cross-participant counting becomes
meaningful.  For each covered experiment this script fits a base-rate
(marginal) and a bigram (transition) model on the participants NOT in
the held-out test split, then scores the test participants' canonical
choice sequences, Laplace-smoothed over the task's option support.
Alignment is guarded per participant: table row count must equal the
transcript's marked-choice count, else the participant is skipped and
logged.  Aggregation mirrors E2: choice -> participant -> task macro.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_canonical_choice_figures import (
    COLUMN_OVERRIDES,
    FAMILIES,
    FAMILY_RENAMES,
)
from build_report_figures import REPO
from mt.data._llm_supervision import find_target_spans

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",
                        default=REPO / "data/psych-101-test/"
                                       "prompts_testing_t1.jsonl",
                        help="Psych-101-test prompts .jsonl")
    parser.add_argument("--tables", default=REPO / "data/psych101_tables",
                        help="Parquet cache from the canonical-choice step")
    parser.add_argument("--output",
                        default=REPO / "outputs/scoring/"
                                       "e2pop_canonical.csv",
                        help="Per-choice output CSV")
    parser.add_argument("--summary",
                        default=REPO / "outputs/scoring/"
                                       "e2pop_canonical_summary.csv",
                        help="Per-experiment macro summary CSV")
    args = parser.parse_args()

    test_counts = load_test_choice_counts(args.data)
    records = []
    skipped = []
    for family in FAMILIES:
        name = FAMILY_RENAMES.get(family, family)
        for config_dir in sorted(Path(args.tables, family).iterdir()):
            if not config_dir.is_dir():
                continue
            experiment = f"{name}/{config_dir.name}.csv"
            if experiment not in test_counts:
                continue
            rows, missed = score_experiment(config_dir, experiment,
                                            test_counts[experiment])
            records.extend(rows)
            skipped.extend(missed)

    frame = pd.DataFrame(records)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    summary = summarize(frame)
    summary.to_csv(args.summary)
    print(f"scored {len(frame)} choices across "
          f"{frame['experiment'].nunique()} experiments; "
          f"skipped {len(skipped)} misaligned participants")
    if skipped:
        print("  skipped:", skipped[:10],
              "..." if len(skipped) > 10 else "")
    print(summary[['pop_base_rate', 'pop_bigram', 'canonical_uniform']]
          .mean().round(4).to_string())

def load_test_choice_counts(path):
    """Return {experiment -> {participant -> transcript choice count}}."""

    counts = defaultdict(dict)
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        counts[row['experiment']][str(row['participant'])] = len(
            find_target_spans(row['text']))
    return counts

def load_sequences(config_dir, family):
    """Return {participant -> canonical choice list, file order kept}."""

    columns = COLUMN_OVERRIDES.get(family, ["choice"])
    frame = pd.concat([pd.read_parquet(p,
                                       columns=['participant'] + columns)
                       for p in sorted(config_dir.rglob("*.parquet"))])
    sequences = {}
    for participant, group in frame.groupby('participant', sort=False):
        values = []
        for _, row in group.iterrows():
            # multi-column tasks interleave stages within each trial row
            values.extend(str(row[c]) for c in columns
                          if pd.notna(row[c]))
        sequences[str(participant)] = values
    return sequences

def fit_counts(sequences, support):
    """Return Laplace-smoothed log marginal and transition matrices."""

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

def score_experiment(config_dir, experiment, transcript_counts):
    family = config_dir.parent.name
    sequences = load_sequences(config_dir, family)
    test_sequences = {}
    skipped = []
    for participant, expected in transcript_counts.items():
        values = sequences.get(participant)
        if values is None or len(values) != expected:
            skipped.append((experiment, participant))
        else:
            test_sequences[participant] = values
    train = [values for participant, values in sequences.items()
             if participant not in transcript_counts]
    support = sorted({v for values in sequences.values() for v in values})
    index, log_marginal, log_transition = fit_counts(train, support)

    uniform = np.log(len(support))
    rows = []
    for participant, values in test_sequences.items():
        codes = [index[v] for v in values]
        for position, code in enumerate(codes):
            if position == 0:
                bigram = -log_marginal[code]
            else:
                bigram = -log_transition[codes[position - 1], code]
            rows.append({'experiment': experiment,
                         'participant': participant,
                         'choice_index': position,
                         'pop_base_rate': -log_marginal[code],
                         'pop_bigram': bigram,
                         'canonical_uniform': uniform})
    return rows, skipped

def summarize(frame):
    """Choice -> participant -> experiment macro means per baseline."""

    per_participant = frame.groupby(['experiment', 'participant'])[
        ['pop_base_rate', 'pop_bigram', 'canonical_uniform']].mean()
    summary = per_participant.groupby('experiment').mean()
    summary['participants'] = per_participant.groupby('experiment').size()
    summary['choices'] = frame.groupby('experiment').size()
    return summary

if __name__ == "__main__":
    main()
