"""Fit and score count-based sequence baselines (E2 entry point)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd

from _common import load_sessions
from mt.evaluation.sequence_baselines import (
    TableBuilder,
    extract_choices,
    score_sequence,
    score_sequence_online,
)

BASELINES = ['uniform', 'base_rate', 'sticky', 'bigram']

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="online",
                        choices=["online", "population"],
                        help="online: within-session causal counts (valid "
                             "under per-participant key randomization); "
                             "population: tables fitted on --train")
    parser.add_argument("--train", default=None,
                        help="Training prompts .jsonl (population mode)")
    parser.add_argument("--test", required=True,
                        help="Held-out prompts .jsonl (scoring data)")
    parser.add_argument("--experiment", default=None,
                        help="Filter to one experiment id")
    parser.add_argument("--max-participants", type=int, default=None,
                        help="Sample at most this many sessions per "
                             "experiment (seeded, reproducible)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed")
    parser.add_argument("--output", required=True,
                        help="Per-choice output CSV path")
    parser.add_argument("--summary", default=None,
                        help="Per-experiment summary CSV path")
    args = parser.parse_args()

    test_rows = load_sessions(args.test, experiment=args.experiment,
                              max_participants=args.max_participants,
                              seed=args.seed)

    sequences = [extract_choices(r['text']) for r in test_rows]
    n_experiments = len({r['experiment'] for r in test_rows})
    print(f"scoring {len(test_rows)} sessions from "
          f"{n_experiments} experiments in {args.mode} mode")

    if args.mode == "population":
        tables = fit_population_tables(args.train, test_rows, sequences)

    results = []
    for row, sequence in zip(test_rows, sequences):
        if args.mode == "online":
            records = score_sequence_online(sequence)
        else:
            records = score_sequence(tables[row['experiment']], sequence)
        for record in records:
            record.update(experiment=row['experiment'],
                          participant=row['participant'])
            results.append(record)

    frame = pd.DataFrame(results)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {len(frame)} choice scores to {output}")

    # macro-aggregate trials -> participant -> experiment (design 9.2)
    participant = frame.groupby(['experiment', 'participant'])[
        BASELINES].mean()
    experiment = participant.groupby('experiment').mean()
    if args.summary:
        summary = Path(args.summary)
        summary.parent.mkdir(parents=True, exist_ok=True)
        experiment.to_csv(summary)
        print(f"wrote per-experiment summary to {summary}")
    print("macro NLL (trial -> participant -> experiment -> overall):")
    print(experiment.mean().to_string())
    print("micro NLL (per choice):")
    print(frame[BASELINES].mean().to_string())

def fit_population_tables(train_path, test_rows, sequences):
    if not train_path:
        raise SystemExit("population mode requires --train")
    vocabulary = defaultdict(set)
    for row, sequence in zip(test_rows, sequences):
        vocabulary[row['experiment']].update(sequence)
    wanted = set(vocabulary)
    builders = defaultdict(TableBuilder)
    train_sessions = defaultdict(int)
    for line in open(train_path):
        row = json.loads(line)
        if row['experiment'] not in wanted:
            continue
        builders[row['experiment']].observe(extract_choices(row['text']))
        train_sessions[row['experiment']] += 1
    unfitted = sorted(exp for exp in wanted if train_sessions[exp] == 0)
    if unfitted:
        print(f"WARNING: {len(unfitted)} experiments have no training "
              f"sessions, their tables are smoothing-only: {unfitted}")
    return {exp: builders[exp].build(vocabulary[exp]) for exp in wanted}

if __name__ == "__main__":
    main()
