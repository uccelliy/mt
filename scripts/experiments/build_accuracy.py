"""A15: how often the model's own readout equals the human's answer.

Zero GPU. `pred_choice == human_choice`, aggregated by section 4 (mean over a
participant's choices, then mean over participants), one number per task.

Two things this is not:

  not a replacement for L_f
      A model that always names the mode scores the same accuracy as a
      perfectly calibrated one on a 60/40 trial (design section 2.3). Accuracy
      answers "would it act like the human", NLL answers "does it hold the
      human's uncertainty". Both are reported; neither substitutes.

  not defined where the answer is multi-token
      The readout is one token by protocol (section 9 clause 8), so on the
      experiments whose choices are strings like `77.37` it can never match.
      Those tasks are **left out**, not scored as zero.

Chance is 1/(canonical option count), taken from the per-task tables published
with Psych-101 rather than from the transcript. The transcript's own `<<...>>`
union is a poor denominator twice over: response keys are randomized per
participant, so labels are not comparable across people, and an option the
participant never pressed is simply absent -- 98% of sessions have a smaller
union than their task, and 196 sessions have exactly one option, where "above
chance" is unreachable by construction.

Restricted to the experiments where that count is usable:

  no table          15 experiments have no canonical table at all
  union != trial    `hebart2023things` has 1,823 canonical objects and shows 3
                    per trial; `jansen2021dunningkruger` switches from sentence
                    options to a numeric scale partway through. For these the
                    canonical union is a WORSE denominator than the transcript
                    union, so both are dropped rather than given a wrong number
  sentinels         `-1` / NaN mark a missed trial, not an option, and are
                    filtered out of the count (this alone brings `kool2016when`
                    from 5 to 4, matching the transcript union exactly)

    python scripts/experiments/build_accuracy.py \
        --run centaur8b=outputs/runs/<tag>/4bit/centaur8b/e3/full \
        ... \
        --output-dir outputs/analysis/<tag>/a15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from _plotting import GRID, INK, SECONDARY, STYLE, short_name, style_axis
from mt.evaluation.aggregate import KEYS, drop_excluded
from mt.models.baselines.canonical_tables import option_counts

NEEDED = ['condition', 'experiment', 'participant', 'choice_index',
          'num_tokens', 'pred_choice', 'human_choice', 'k_options']
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']

# Their canonical union is the whole task, not what a trial offers; see above.
UNION_IS_NOT_TRIAL_SET = ('hebart2023things/exp1.csv',
                          'jansen2021dunningkruger/exp1.csv')


def parse_run(text):
    label, _, path = text.partition('=')
    if not label or not path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {text!r}")
    return label, Path(path)


def read_run(path, condition):
    if path.is_dir():
        for name in ('predictions.csv', 'predictions.csv.gz'):
            if (path / name).exists():
                path = path / name
                break
        else:
            raise SystemExit(f"No predictions table in {path}")
    frame = pd.read_csv(path, usecols=NEEDED, low_memory=False)
    frame = frame[frame['condition'] == condition]
    frame['participant'] = frame['participant'].astype(str)
    for column in ('pred_choice', 'human_choice'):
        frame[column] = frame[column].fillna('').astype(str)
    return drop_excluded(frame)


def accuracy_by_task(frame, counts):
    """Section 4 aggregation of the hit rate, single-token choices only."""

    single = frame[frame['num_tokens'] == 1].copy()
    single['hit'] = single['pred_choice'] == single['human_choice']
    single['chance'] = 1.0 / single['experiment'].map(counts)
    per_participant = single.groupby(KEYS)[['hit', 'chance']].mean()
    grouped = per_participant.groupby('experiment')
    table = pd.DataFrame({
        'accuracy': grouped['hit'].mean(),
        'chance': grouped['chance'].mean(),
        'n_participants': grouped.size(),
    })
    table['n_choices'] = single.groupby('experiment').size()
    # tasks that mix single- and multi-token answers are scored on the
    # single-token part only, so say what fraction that was
    table['n_total'] = frame.groupby('experiment').size()
    table['coverage'] = table['n_choices'] / table['n_total']
    table['lift'] = table['accuracy'] / table['chance']
    return table.reset_index()


def figure_by_task(tasks, models, output):
    """One row per task: chance, then where each model landed."""

    wide = tasks.pivot_table(index='experiment', columns='model',
                             values='accuracy')[models]
    chance = tasks.groupby('experiment')['chance'].mean()
    order = list(wide.mean(axis=1).sort_values().index)
    fig, ax = plt.subplots(figsize=(8.4, 0.19 * len(order) + 1.8))
    y = range(len(order))
    for row, task in enumerate(order):
        values = list(wide.loc[task]) + [chance[task]]
        ax.hlines(row, min(values), max(values), color=GRID, lw=0.9, zorder=1)
    ax.scatter(chance.loc[order], y, marker='|', s=90, color=SECONDARY,
               zorder=3, label='chance (1/k)')
    for color, model in zip(SERIES_COLORS, models):
        ax.scatter(wide.loc[order, model], y, s=16, color=color, zorder=2,
                   label=model)
    ax.set_yticks(list(y), [short_name(t) for t in order], fontsize=6.2)
    ax.set_ylim(-1, len(order))
    ax.invert_yaxis()
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel('accuracy — share of choices where the readout equalled the '
                  "human's answer", fontsize=9)
    style_axis(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=4,
               loc='upper center', bbox_to_anchor=(0.5, 0.968))
    fig.suptitle('A15  Accuracy against the human, per task', fontsize=11,
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', type=parse_run, required=True,
                        metavar='LABEL=PATH')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--condition', default='full')
    parser.add_argument('--tables', default='data/psych101_tables',
                        help='Directory of canonical per-task tables')
    args = parser.parse_args()

    counts = {k: v for k, v in option_counts(args.tables).items()
              if k not in UNION_IS_NOT_TRIAL_SET}
    print(f"canonical option counts for {len(counts)} experiments "
          f"({len(UNION_IS_NOT_TRIAL_SET)} dropped: their union is the task, "
          "not the trial)")

    models = [label for label, _ in args.run]
    tasks = []
    for label, path in args.run:
        frame = read_run(path, args.condition)
        frame = frame[frame['experiment'].isin(counts)]
        table = accuracy_by_task(frame, counts)
        table.insert(0, 'model', label)
        tasks.append(table)
        scored = set(table['experiment'])
        gone = sorted(set(frame['experiment']) - scored)
        partial = table.loc[table['coverage'] < 1, 'experiment']
        print(f"{label}: {len(table)} tasks scored; {len(gone)} entirely "
              f"multi-token and left out {gone}; {len(partial)} scored on "
              "part of their choices only")
    tasks = pd.concat(tasks, ignore_index=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(out / 'a15_accuracy_by_task.csv', index=False)
    with plt.rc_context(STYLE):
        figure_by_task(tasks, models, out / 'a15_by_task.png')
    print(f"\nwrote 1 table and 1 figure to {out}")

    summary = tasks.groupby('model')[['accuracy', 'chance', 'lift']].mean()
    summary['tasks_above_chance'] = (
        tasks.assign(above=tasks['accuracy'] > tasks['chance'])
        .groupby('model')['above'].sum())
    print("\naveraged over tasks (orientation; the per-task table is the "
          "result):")
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
