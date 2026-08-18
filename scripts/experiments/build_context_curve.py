"""A4: how a model's prediction changes as the visible history is truncated.

Zero GPU. The E3 arm scores five anchor segments per session under six finite
windows; `condition=full` supplies the seventh point.

Three things the naive version of this plot gets wrong:

  w = infinity is not `full`'s average
      `full` scores every choice of the session, 1.16M of them; the anchors are
      65k. Comparing the two averages compares different questions. The seventh
      point is `full` **restricted by an inner join** to exactly the anchor
      choices, so all seven points describe the same choices.

  the nominal window is not the visible history
      `effective_window = min(target_index, window)`, so at the first anchor
      (target 0) every window shows the same prompt, and 60% of anchors give
      identical numbers at w=10 and w=20. Averaging over anchors by nominal w
      pulls the right half of the curve flat -- an artefact of the grid, not a
      finding about context.

  so the main curve uses anchors with `target_index >= 20` only
      There all six finite windows are unsaturated and the seven points share
      one fixed set of choices. 17,078 anchors over 73 tasks. The remaining
      anchors are reported separately, stratified by their rank in the grid and
      plotted against `effective_window`, where duplicates collapse honestly.

Second output, free from the same table: at `target_index == 0` the w=0 prompt
and the full transcript share a byte-identical prefix up to the scored choice,
so their NLLs are mathematically equal. The observed spread is this run's
per-choice numerical floor, and no effect smaller than it may be asserted.

    python scripts/experiments/build_context_curve.py \
        --run centaur8b=outputs/runs/<tag>/4bit/centaur8b/e3/full \
        ... \
        --output-dir outputs/analysis/<tag>/a4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _plotting import INK, SECONDARY, STYLE, short_name, style_axis
from mt.evaluation.aggregate import KEYS, drop_excluded, task_table

CHOICE_KEYS = ['experiment', 'participant', 'choice_index']
NEEDED = ['condition', *CHOICE_KEYS, 'nll', 'window', 'target_index',
          'effective_window']
WINDOWS = [0, 1, 2, 5, 10, 20]
INFINITY = 10 ** 6          # sorts last; displayed as the infinity sign
UNSATURATED_FROM = 20
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']


def parse_run(text):
    label, _, path = text.partition('=')
    if not label or not path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {text!r}")
    return label, Path(path)


def read_run(path):
    """Return the anchor choices under all seven windows, w=inf included."""

    if path.is_dir():
        for name in ('predictions.csv', 'predictions.csv.gz'):
            if (path / name).exists():
                path = path / name
                break
        else:
            raise SystemExit(f"No predictions table in {path}")
    frame = pd.read_csv(path, usecols=NEEDED, low_memory=False)
    frame['participant'] = frame['participant'].astype(str)

    e3 = frame[frame['condition'] != 'full'].copy()
    e3['window'] = e3['window'].astype(int)

    anchors = e3[CHOICE_KEYS + ['target_index']].drop_duplicates()
    full = frame[frame['condition'] == 'full']
    unlimited = anchors.merge(full[CHOICE_KEYS + ['nll']], on=CHOICE_KEYS,
                              how='inner', validate='one_to_one')
    if len(unlimited) != len(anchors):
        raise SystemExit(f"{path}: {len(anchors) - len(unlimited)} anchor "
                         "choices are missing from condition=full, so w=inf "
                         "cannot be put on the same choices")
    unlimited['window'] = INFINITY
    unlimited['effective_window'] = unlimited['target_index']

    columns = CHOICE_KEYS + ['nll', 'window', 'target_index', 'effective_window']
    return pd.concat([e3[columns], unlimited[columns]], ignore_index=True)


def anchor_rank(frame):
    """Which of the five grid positions each anchor is (0 = first segment)."""

    per_session = (frame[['experiment', 'participant', 'target_index']]
                   .drop_duplicates())
    per_session['anchor_rank'] = (
        per_session.groupby(['experiment', 'participant'])['target_index']
        .rank(method='dense').astype(int) - 1)
    return frame.merge(per_session, on=['experiment', 'participant',
                                        'target_index'], how='left')


def curve(frame, column='window'):
    """Section 4 aggregation inside each window bucket, per task."""

    pieces = []
    for value, part in frame.groupby(column):
        table = task_table(part).reset_index()
        table.insert(1, column, value)
        pieces.append(table)
    return pd.concat(pieces, ignore_index=True)


def numerical_floor(frame):
    """w=0 against w=inf at target 0, where the two are provably equal."""

    at_zero = frame[frame['target_index'] == 0]
    wide = at_zero.pivot_table(index=CHOICE_KEYS, columns='window',
                               values='nll')
    if 0 not in wide.columns or INFINITY not in wide.columns:
        return pd.Series(dtype=float)
    return (wide[0] - wide[INFINITY]).abs().dropna()


def tick_labels(values):
    return ['∞' if v == INFINITY else str(int(v)) for v in values]


def figure_main(curves, models, output):
    """The unsaturated curve: seven windows, one fixed set of choices."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for value, ylabel in (('task_nll', 'L  (nat, lower is better)'),
                          ('task_probability', 'p = exp(-L)')):
        ax = axes[0] if value == 'task_nll' else axes[1]
        for color, model in zip(SERIES_COLORS, models):
            part = (curves[curves['model'] == model]
                    .groupby('window')[value].mean().sort_index())
            ax.plot(range(len(part)), part.to_numpy(), color=color, lw=2,
                    marker='o', markersize=4, label=model,
                    solid_capstyle='round')
            ticks = part.index
        ax.set_xticks(range(len(ticks)), tick_labels(ticks))
        ax.set_xlabel('visible history (marked-choice segments)', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        style_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=6,
               loc='upper center', bbox_to_anchor=(0.5, 0.93))
    fig.suptitle('A4  Context-length curve on anchors with at least 20 segments '
                 'of history (no window saturates; all seven points are the '
                 'same choices)', fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    fig.savefig(output, dpi=200)
    plt.close(fig)


ANCHOR_LABELS = {0: 'anchor 0 — first segment', 1: 'anchor 1 — second segment',
                 2: 'anchor 2 — 10% in', 3: 'anchor 3 — 50% in',
                 4: 'anchor 4 — last segment'}


def figure_by_anchor(frames, models, output):
    """All five grid positions, so the early anchors are not invisible.

    The main curve keeps only anchors with >= 20 segments of history, which by
    construction excludes anchors 0 and 1 entirely -- at the first segment
    there is no history for a window to reveal. Those anchors still belong in
    the record: their curves are flat, and the flatness is the point.
    """

    ranks = sorted(ANCHOR_LABELS)
    fig, axes = plt.subplots(1, len(ranks), figsize=(3.0 * len(ranks), 3.6),
                             sharey=True)
    for ax, rank in zip(axes, ranks):
        for color, model in zip(SERIES_COLORS, models):
            part = frames[model]
            part = part[part['anchor_rank'] == rank]
            line = curve(part).groupby('window')['task_nll'].mean().sort_index()
            ax.plot(range(len(line)), line.to_numpy(), color=color, lw=1.8,
                    marker='o', markersize=3.4, label=model)
            ticks = line.index
        ax.set_xticks(range(len(ticks)), tick_labels(ticks), fontsize=7)
        ax.set_title(ANCHOR_LABELS[rank], fontsize=8.5)
        ax.set_xlabel('nominal window', fontsize=8)
        style_axis(ax)
    axes[0].set_ylabel('L  (nat)', fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=6,
               loc='upper center', bbox_to_anchor=(0.5, 0.93))
    fig.suptitle('A4  Every anchor position — a flat line means the window had '
                 'no history to reveal (effective_window = min(target, w))',
                 fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def figure_by_task(curves, models, output):
    """One panel per task, on the same unsaturated anchors."""

    tasks = sorted(curves['experiment'].unique())
    cols = 7
    rows = -(-len(tasks) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 1.55 * rows),
                             sharex=True)
    for ax, task in zip(axes.ravel(), tasks):
        part = curves[curves['experiment'] == task]
        for color, model in zip(SERIES_COLORS, models):
            line = (part[part['model'] == model]
                    .set_index('window')['task_probability'].sort_index())
            ax.plot(range(len(line)), line.to_numpy(), color=color, lw=1.1)
            ticks = line.index
        ax.set_xticks(range(len(ticks)), tick_labels(ticks))
        ax.set_title(short_name(task), fontsize=5.6, pad=2)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=5)
        style_axis(ax)
    for ax in axes.ravel()[len(tasks):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in SERIES_COLORS]
    fig.legend(handles, list(models), frameon=False, fontsize=8, ncol=6,
               loc='lower center')
    fig.suptitle('A4  Per task — p by visible history, anchors with ≥ 20 '
                 'segments available', fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', type=parse_run, required=True,
                        metavar='LABEL=PATH')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    models = [label for label, _ in args.run]
    unsaturated, stratified, floors, frames = [], [], {}, {}
    for label, path in args.run:
        frame = anchor_rank(drop_excluded(read_run(path)))
        frames[label] = frame

        clean = frame[frame['target_index'] >= UNSATURATED_FROM]
        table = curve(clean)
        table.insert(0, 'model', label)
        unsaturated.append(table)

        by_rank = []
        for rank, part in frame.groupby('anchor_rank'):
            piece = curve(part, column='effective_window')
            piece.insert(0, 'anchor_rank', rank)
            by_rank.append(piece)
        by_rank = pd.concat(by_rank, ignore_index=True)
        by_rank.insert(0, 'model', label)
        stratified.append(by_rank)

        floors[label] = numerical_floor(frame)
        anchors = len(clean[CHOICE_KEYS].drop_duplicates())
        print(f"{label}: {anchors:,} unsaturated anchors over "
              f"{clean['experiment'].nunique()} tasks")

    unsaturated = pd.concat(unsaturated, ignore_index=True)
    stratified = pd.concat(stratified, ignore_index=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    unsaturated.to_csv(out / 'a4_curve_unsaturated.csv', index=False)
    stratified.to_csv(out / 'a4_curve_by_anchor.csv', index=False)
    with plt.rc_context(STYLE):
        figure_main(unsaturated, models, out / 'a4_curve.png')
        figure_by_anchor(frames, models, out / 'a4_by_anchor.png')
        figure_by_task(unsaturated, models, out / 'a4_by_task.png')
    print(f"\nwrote 2 tables and 3 figures to {out}")

    summary = unsaturated.pivot_table(index='model', columns='window',
                                      values='task_nll')
    summary.columns = tick_labels(summary.columns)
    print("\nL averaged over tasks, by visible history "
          "(orientation; the per-task table is the result):")
    print(summary.round(4).to_string())

    print("\nnumerical floor — |w=0 minus w=inf| at target_index 0, "
          "where the two are provably equal:")
    for label, values in floors.items():
        if values.empty:
            continue
        print(f"  {label:<22} n={len(values):>6,}  median {values.median():.2e}"
              f"  p95 {values.quantile(0.95):.2e}  max {values.max():.2e}")


if __name__ == '__main__':
    main()
