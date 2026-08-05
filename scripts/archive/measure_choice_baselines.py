"""What accuracy must a model beat before "it predicts humans" means anything?

The benchmark's primary metric is now accuracy, not NLL, so it needs accuracy
nulls.  These are the same four counting rules E2 already established as the
"not cognitive evidence" floor (design 9.4), re-expressed as argmax accuracy:
whatever a lookup table achieves is not evidence the model understood the task.

All four are prequential -- trial t is predicted from trials 1..t-1 of the same
session only, never from the future and never across participants (Psych-101
randomizes response keys per participant, so cross-participant counting in the
raw label space is noise; see handoff decision 2).  Zero GPU: the human choice
sequence is read straight out of the transcript markers.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/psych-101-test/prompts_testing_t1.jsonl"
MARKER = re.compile(r"<<(.*?)>>", re.S)
RULES = ('uniform', 'base_rate', 'sticky', 'bigram')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA, help="Path to prompts .jsonl")
    parser.add_argument("--output", default=REPO / "outputs/analysis_choice_baselines.csv",
                        help="Per-experiment baseline accuracy table")
    args = parser.parse_args()

    rows = [score_session(record) for record in read_sessions(Path(args.data))]
    frame = pd.DataFrame([r for r in rows if r is not None])
    per_experiment = frame.groupby(['experiment', 'participant'])[list(RULES)].mean()
    per_experiment = per_experiment.groupby('experiment').mean()
    per_experiment.to_csv(args.output)
    report(frame, per_experiment)

def read_sessions(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)

def score_session(record):
    choices = MARKER.findall(record['text'])
    if not choices:
        return None
    row = {
        'experiment': record['experiment'],
        'participant': str(record['participant']),
        'n_choices': len(choices),
        'n_labels': len(set(choices)),
    }
    row['uniform'] = 1.0 / row['n_labels']
    for rule in ('base_rate', 'sticky', 'bigram'):
        row[rule] = prequential_accuracy(choices, rule)
    return row

def prequential_accuracy(choices, rule):
    """Accuracy of an online counting rule; trial t sees only 1..t-1.

    The first trial has no history under every rule, so it is scored by no
    rule at all rather than being charged to whichever rule guesses first.
    """

    seen = collections.Counter()
    followers = collections.defaultdict(collections.Counter)
    previous = None
    hits = scored = 0
    for choice in choices:
        prediction = predict(rule, seen, followers, previous)
        if prediction is not None:
            scored += 1
            hits += prediction == choice
        seen[choice] += 1
        if previous is not None:
            followers[previous][choice] += 1
        previous = choice
    return hits / scored if scored else float('nan')

def predict(rule, seen, followers, previous):
    if rule == 'sticky':
        return previous
    if rule == 'bigram' and followers[previous]:
        return followers[previous].most_common(1)[0][0]
    return seen.most_common(1)[0][0] if seen else None

def report(frame, per_experiment):
    print(f"{len(frame)} sessions, {frame['experiment'].nunique()} experiments, "
          f"{frame['n_choices'].sum()} choices")
    print()
    print("task-macro accuracy (participant -> experiment -> equal weight):")
    for rule in RULES:
        print(f"  {rule:10s} {per_experiment[rule].mean():.4f}")
    print()
    hardest = per_experiment['bigram'].nsmallest(5)
    easiest = per_experiment['bigram'].nlargest(5)
    print("bigram is weakest here (the clean battlefield for a model):")
    for name, value in hardest.items():
        print(f"  {name:44s} {value:.3f}")
    print("bigram already wins here (beating it proves little):")
    for name, value in easiest.items():
        print(f"  {name:44s} {value:.3f}")

if __name__ == "__main__":
    main()
