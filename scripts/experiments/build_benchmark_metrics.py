"""Derive the design's three per-task metrics from completed Track P runs.

Zero GPU: everything here is a transform of the `nll` column the scorer already
wrote (design section 3). One row of input is one scored choice; one row of
output is one task.

    metrics_by_task.csv     L_f and p_f for every model x condition x task
    probability_ratio.csv   R_f against each counting baseline (condition=full)
    comparability.csv       tasks where the models did not score the same
                            sessions, so their L_f are not comparable

The third output is the reason this is a script and not three calls: L_f
averages participants equally, so a model that lost sessions to OOM produces a
number on a different participant set than its neighbours. That difference is
invisible in the metric and would be read as a model effect.

    python scripts/experiments/build_benchmark_metrics.py \
        --run llama31_8b=outputs/runs/<tag>/4bit/llama31_8b/e3/full \
        --run centaur8b=outputs/runs/<tag>/4bit/centaur8b/e3/full \
        --baselines outputs/baselines/psych-101-test/sequence_online.csv \
        --output-dir outputs/analysis/<tag>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mt.evaluation.aggregate import (
    BENCHMARK_EXCLUDED_EXPERIMENTS,
    CHOICE_KEYS,
    drop_excluded,
    matched_probability_ratio,
    task_table,
)

BASELINES = ('uniform', 'base_rate', 'sticky', 'bigram')
NEEDED = ['condition', 'experiment', 'participant', 'choice_index', 'nll',
          'num_tokens']
FULL = 'full'


def parse_run(text):
    """Parse a `label=path` run specification."""

    label, _, path = text.partition('=')
    if not label or not path:
        raise argparse.ArgumentTypeError(
            f"expected label=path, got {text!r}")
    return label, Path(path)


def read_predictions(path):
    """Read one run's predictions table, by directory or by file.

    Accepts the gzipped form, because only seven of the table's columns are
    needed here and shipping the run off the cluster in that shape is an
    order of magnitude cheaper.
    """

    if path.is_dir():
        for name in ('predictions.csv', 'predictions.csv.gz'):
            if (path / name).exists():
                path = path / name
                break
        else:
            raise SystemExit(f"No predictions table in {path}")
    if not path.exists():
        raise SystemExit(f"No predictions table at {path}")
    frame = pd.read_csv(path, usecols=NEEDED, low_memory=False)
    frame['participant'] = frame['participant'].astype(str)
    nan = int(frame['nll'].isna().sum())
    if nan:
        raise SystemExit(
            f"{path}: {nan} of {len(frame)} choices have a non-finite NLL. "
            "Every metric here sums that column, and pandas drops NaN from a "
            "sum while its num_tokens still counts, so the tables would "
            "silently report a lower NLL than the model earned.")
    return frame


def read_baselines(path):
    """Read the counting baselines, one column of per-choice NLL each."""

    frame = pd.read_csv(path, usecols=CHOICE_KEYS + list(BASELINES),
                        low_memory=False)
    frame['participant'] = frame['participant'].astype(str)
    return frame


def comparability(runs):
    """Report tasks where the runs did not cover the same sessions.

    Compared on the participant set of `condition=full`, which is the one that
    scores every choice of every session.
    """

    sets = {
        label: frame[frame['condition'] == FULL]
        .groupby('experiment')['participant'].agg(lambda s: frozenset(s))
        for label, frame in runs.items()
    }
    table = pd.DataFrame(sets)
    rows = []
    for experiment, per_model in table.iterrows():
        present = {k: v for k, v in per_model.items() if isinstance(v, frozenset)}
        sizes = {k: len(v) for k, v in present.items()}
        agree = len(set(present.values())) == 1 and len(present) == len(runs)
        if agree:
            continue
        shared = set.intersection(*(set(v) for v in present.values())) if present else set()
        rows.append({
            'experiment': experiment,
            'n_shared_participants': len(shared),
            'missing_models': ','.join(sorted(set(runs) - set(present))),
            **{f'n_{label}': sizes.get(label, 0) for label in runs},
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', type=parse_run, required=True,
                        metavar='LABEL=PATH',
                        help='One completed run; repeat per model')
    parser.add_argument('--baselines', required=True,
                        help='Per-choice counting-baseline CSV (zero GPU, '
                             'scripts/experiments/run_sequence_baselines.py)')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--keep-excluded', action='store_true',
                        help='Keep the experiments design section 9 clause 7 '
                             'excludes from model comparison')
    args = parser.parse_args()

    runs = {}
    for label, path in args.run:
        frame = read_predictions(path)
        if not args.keep_excluded:
            frame = drop_excluded(frame)
        runs[label] = frame
        conditions = frame['condition'].nunique()
        print(f"{label}: {len(frame)} choices, "
              f"{frame['experiment'].nunique()} tasks, {conditions} conditions")
    if not args.keep_excluded:
        print(f"excluded from comparison: "
              f"{', '.join(BENCHMARK_EXCLUDED_EXPERIMENTS)} (clause 7)")

    baselines = read_baselines(Path(args.baselines))
    print(f"baselines: {len(baselines)} choices, "
          f"{baselines['experiment'].nunique()} tasks")

    metrics = []
    for label, frame in runs.items():
        for condition, part in frame.groupby('condition'):
            table = task_table(part).reset_index()
            table.insert(0, 'model', label)
            table.insert(1, 'condition', condition)
            metrics.append(table)
    metrics = pd.concat(metrics, ignore_index=True)

    # R_f only for `full`. The counting baselines are prequential over the whole
    # session, so under a truncated window the model sees w segments while the
    # baseline still sees everything -- a defensible comparison, but a different
    # question, and not one section 3.3 defines.
    ratios = []
    for label, frame in runs.items():
        full = frame[frame['condition'] == FULL]
        for baseline in BASELINES:
            ratio, coverage = matched_probability_ratio(full, baselines,
                                                        baseline)
            ratios.append(pd.DataFrame({
                'model': label,
                'baseline': baseline,
                'experiment': ratio.index,
                'probability_ratio': ratio.to_numpy(),
                'coverage': coverage.to_numpy(),
            }))
    ratios = pd.concat(ratios, ignore_index=True)

    mismatches = comparability(runs)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / 'metrics_by_task.csv', index=False)
    ratios.to_csv(out / 'probability_ratio.csv', index=False)
    mismatches.to_csv(out / 'comparability.csv', index=False)
    print(f"\nwrote {len(metrics)} task rows and {len(ratios)} ratio rows "
          f"to {out}")

    full = metrics[metrics['condition'] == FULL]
    print("\ncondition=full, averaged over tasks (per-task table is the result;"
          " this line is orientation only):")
    print(full.groupby('model')[['task_nll', 'task_probability']]
          .mean().to_string())
    print("\nmedian R_f over tasks:")
    print(ratios.groupby(['model', 'baseline'])['probability_ratio']
          .median().unstack().to_string())
    if len(mismatches):
        print(f"\n{len(mismatches)} task(s) NOT comparable across models "
              "-- see comparability.csv:")
        print(mismatches.to_string(index=False))
    else:
        print("\nevery task covers the same sessions in every run")


if __name__ == '__main__':
    main()
