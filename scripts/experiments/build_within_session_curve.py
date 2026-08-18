"""A17: how a model's prediction changes as a session goes on.

Zero GPU. Uses `condition=full` only -- the E3 anchors sit at five positions,
which cannot show a continuous trend.

Two x-axes, both needed:

  decile    within-session quantile, so sessions of wildly different length
            line up; this is the axis the within-session claim is made on
  absolute  the first 20 choices, because a 1,000-trial session puts ~100
            trials inside decile 0 and the cold start disappears

The four counting baselines are drawn on the same axes from the same choices.
They are prequential -- they count within the session -- so they get better with
position for the trivial reason that the human becomes more predictable. That is
the null hypothesis for this whole analysis: a model curve that falls no faster
than the baselines has not learned anything the counts did not.

    python scripts/experiments/build_within_session_curve.py \
        --run centaur8b=outputs/runs/<tag>/4bit/centaur8b/e3/full \
        ... \
        --baselines outputs/baselines/psych-101-test/sequence_online.csv \
        --output-dir outputs/analysis/<tag>/a17
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _plotting import GRID, INK, MUTED, SECONDARY, STYLE, short_name, style_axis
from mt.evaluation.aggregate import drop_excluded

CHOICE_KEYS = ['experiment', 'participant', 'choice_index']
BASELINES = ('uniform', 'base_rate', 'sticky', 'bigram')
N_DECILES = 10
N_ABSOLUTE = 20

# Fixed hue order, never cycled (validated: worst adjacent CVD dE 9.1,
# normal-vision 19.6, on the light surface). Baselines are deliberately not
# given categorical hues -- they are reference lines, a different job.
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
BASELINE_STYLES = [(0, (1, 1.6)), (0, (4, 1.8)), (0, (6, 1.6, 1, 1.6)), (0, ())]


def parse_run(text):
    label, _, path = text.partition('=')
    if not label or not path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {text!r}")
    return label, Path(path)


def read_full_predictions(path):
    """One run's `condition=full` choices -- every choice of every session."""

    if path.is_dir():
        for name in ('predictions.csv', 'predictions.csv.gz'):
            if (path / name).exists():
                path = path / name
                break
        else:
            raise SystemExit(f"No predictions table in {path}")
    frame = pd.read_csv(path, usecols=['condition', *CHOICE_KEYS, 'nll',
                                       'num_tokens'], low_memory=False)
    frame = frame[frame['condition'] == 'full'].drop(columns='condition')
    frame['participant'] = frame['participant'].astype(str)
    if frame['nll'].isna().any():
        raise SystemExit(f"{path}: non-finite NLL; the curve would be biased "
                         "towards whichever positions the failure hit")
    return frame


def read_baselines(path, keys):
    """The counting baselines, restricted to exactly the model's choices."""

    frame = pd.read_csv(path, usecols=CHOICE_KEYS + list(BASELINES),
                        low_memory=False)
    frame['participant'] = frame['participant'].astype(str)
    frame = frame.merge(keys, on=CHOICE_KEYS, how='inner')
    long = frame.melt(id_vars=CHOICE_KEYS, value_vars=list(BASELINES),
                      var_name='source', value_name='nll')
    return long


def add_positions(frame):
    """Label each choice by its rank in its session, absolute and by decile.

    Grouped by `source` as well: every source scores the same choices, so a
    session's rows appear once per source and counting without it would run the
    rank across all of them at once.
    """

    frame = frame.sort_values(['source', 'experiment', 'participant',
                               'choice_index'])
    session = frame.groupby(['source', 'experiment', 'participant'], sort=False)
    frame['rank'] = session.cumcount()
    length = session['choice_index'].transform('size')
    # floor into N_DECILES bins; the last choice of a session lands in bin 9
    frame['decile'] = np.minimum((frame['rank'] * N_DECILES) // length,
                                 N_DECILES - 1)
    return frame


def curve(frame):
    """Section 4 aggregation inside each position bucket.

    Two levels, exactly as elsewhere: mean over a participant's choices in the
    bucket, then mean over participants. The third level (across tasks) is not
    taken here -- per-task is the deliverable.
    """

    pieces = []
    for axis, column in (('decile', 'decile'), ('absolute', 'rank')):
        part = frame if axis == 'decile' else frame[frame['rank'] < N_ABSOLUTE]
        keys = ['source', 'experiment', column]
        per_participant = part.groupby(keys + ['participant'])['nll'].mean()
        grouped = per_participant.groupby(keys)
        table = pd.DataFrame({
            'n_participants': grouped.size(),
            'task_nll': grouped.mean(),
        }).reset_index().rename(columns={column: 'bucket'})
        table['n_choices'] = part.groupby(keys).size().to_numpy()
        table.insert(1, 'axis', axis)
        pieces.append(table)
    table = pd.concat(pieces, ignore_index=True)
    table['task_probability'] = np.exp(-table['task_nll'])
    return table


def overview(table):
    """Mean over tasks -- orientation only, never the reported result."""

    return (table.groupby(['source', 'axis', 'bucket'])['task_nll']
            .mean().rename('nll').reset_index())


def label_ends(ax, ends):
    """Direct-label line ends, pushed apart so the four never collide."""

    low, high = ax.get_ylim()
    gap = 0.045 * (high - low)
    placed = []
    for x, y, text in sorted(ends, key=lambda item: item[1]):
        if placed and y - placed[-1] < gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(text, (x, y), xytext=(5, 0), textcoords='offset points',
                    fontsize=6.5, color=SECONDARY, va='center',
                    annotation_clip=False)


def draw_series(ax, frame, models, baselines, value):
    """One panel: model lines in categorical hues, baselines recessive."""

    ends = []
    for style, name in zip(BASELINE_STYLES, baselines):
        part = frame[frame['source'] == name].sort_values('bucket')
        ax.plot(part['bucket'], part[value], color=MUTED, lw=1.2,
                linestyle=style, zorder=1)
        ends.append((part['bucket'].iloc[-1], part[value].iloc[-1], name))
    for color, name in zip(SERIES_COLORS, models):
        part = frame[frame['source'] == name].sort_values('bucket')
        ax.plot(part['bucket'], part[value], color=color, lw=2, zorder=2,
                label=name, solid_capstyle='round')
    style_axis(ax)
    label_ends(ax, ends)


def figure_overview(table, models, output):
    """L and p against both axes, models over the counting baselines."""

    summary = overview(table)
    summary['probability'] = np.exp(-summary['nll'])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
    panels = [
        ('decile', 'nll', 'within-session decile', 'L  (nat, lower is better)'),
        ('decile', 'probability', 'within-session decile', 'p = exp(-L)'),
        ('absolute', 'nll', 'absolute position (first 20 choices)', 'L  (nat)'),
        ('absolute', 'probability', 'absolute position (first 20 choices)', 'p'),
    ]
    for ax, (axis, value, xlabel, ylabel) in zip(axes.ravel(), panels):
        draw_series(ax, summary[summary['axis'] == axis], models,
                    BASELINES, value)
        ax.set_xlabel(xlabel, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=8.5)
        ax.margins(x=0.16)
        # deciles are 0-9; without this the padding draws a tick at 10 and
        # reads as a missing bucket
        ticks = range(N_DECILES) if axis == 'decile' else range(0, N_ABSOLUTE, 5)
        ax.set_xticks(list(ticks))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=6,
               loc='upper center', bbox_to_anchor=(0.5, 0.955))
    fig.suptitle('A17  Within-session trend, averaged over 73 tasks '
                 '(orientation only \u2014 the per-task table is the result)',
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def figure_decile(table, models, output):
    """Only L against the decile axis -- the overview's first panel, alone.

    The four-panel overview says more than a talk slide needs; this is the one
    panel that carries the claim, at a size where the lines are readable.
    """

    summary = overview(table)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    draw_series(ax, summary[summary['axis'] == 'decile'], models, BASELINES,
                'nll')
    ax.set_xlabel('within-session decile', fontsize=9)
    ax.set_ylabel('L  (nat, lower is better)', fontsize=9)
    ax.margins(x=0.16)
    ax.set_xticks(list(range(N_DECILES)))
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=3,
               loc='upper center', bbox_to_anchor=(0.5, 0.94))
    fig.suptitle('A17  Within-session trend, averaged over 73 tasks',
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.845))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def figure_ratio(table, models, output):
    """R against each baseline: exp(L_baseline - L_model), one panel each."""

    summary = overview(table).pivot_table(index=['axis', 'bucket'],
                                          columns='source', values='nll')
    fig, axes = plt.subplots(1, len(BASELINES), figsize=(13, 3.4), sharey=True)
    for ax, name in zip(axes, BASELINES):
        decile = summary.loc['decile']
        for color, model in zip(SERIES_COLORS, models):
            ratio = np.exp(decile[name] - decile[model])
            ax.plot(ratio.index, ratio.to_numpy(), color=color, lw=2,
                    label=model, solid_capstyle='round')
        ax.axhline(1.0, color=SECONDARY, lw=1, linestyle=(0, (4, 2)), zorder=1)
        ax.annotate('tie', (ax.get_xlim()[1], 1.0), xytext=(-14, 4),
                    textcoords='offset points', fontsize=6.5, color=SECONDARY)
        ax.set_title(f'vs {name}', fontsize=9)
        ax.set_xlabel('within-session decile', fontsize=8.5)
        ax.set_xticks(list(range(N_DECILES)))
        style_axis(ax)
    axes[0].set_ylabel('R = p(model) / p(baseline)', fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7, loc='upper left')
    fig.suptitle('A17  Metric three by session position '
                 '(ratio taken after averaging L over tasks)',
                 fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def figure_by_task(table, models, output):
    """Every task its own panel -- the per-task deliverable, not a summary."""

    decile = table[table['axis'] == 'decile']
    tasks = sorted(decile['experiment'].unique())
    cols = 7
    rows = -(-len(tasks) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 1.55 * rows),
                             sharex=True)
    for ax, task in zip(axes.ravel(), tasks):
        part = decile[decile['experiment'] == task]
        for name in BASELINES:
            line = part[part['source'] == name].sort_values('bucket')
            if name == 'uniform':
                # random choice, drawn in every panel: on this axis it sits at
                # 1/k, so each panel carries its own "what guessing buys you"
                ax.plot(line['bucket'], line['task_probability'],
                        color=SECONDARY, lw=1.1, linestyle=(0, (3, 2)),
                        zorder=3)
            else:
                ax.plot(line['bucket'], line['task_probability'], color=GRID,
                        lw=1, zorder=1)
        for color, model in zip(SERIES_COLORS, models):
            line = part[part['source'] == model].sort_values('bucket')
            ax.plot(line['bucket'], line['task_probability'], color=color,
                    lw=1.1, zorder=2)
        ax.set_title(short_name(task), fontsize=5.6, pad=2)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=5)
        style_axis(ax)
    for ax in axes.ravel()[len(tasks):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in SERIES_COLORS]
    handles.append(plt.Line2D([], [], color=SECONDARY, lw=1.1,
                              linestyle=(0, (3, 2))))
    handles.append(plt.Line2D([], [], color=GRID, lw=1))
    fig.legend(handles, list(models) + ['uniform (random)',
                                        'base rate / sticky / bigram'],
               frameon=False, fontsize=8, ncol=8, loc='lower center')
    fig.suptitle('A17  Per task \u2014 p = exp(-L) by within-session decile',
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.975))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', type=parse_run, required=True,
                        metavar='LABEL=PATH')
    parser.add_argument('--baselines', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    models = [label for label, _ in args.run]
    if len(models) > len(SERIES_COLORS):
        raise SystemExit(
            f"{len(models)} runs but only {len(SERIES_COLORS)} validated hues; "
            "a further series must fold into small multiples, not a new colour")

    frames = []
    for label, path in args.run:
        frame = read_full_predictions(path)
        frame['source'] = label
        frames.append(frame)
        print(f"{label}: {len(frame):,} full-condition choices")
    models_frame = pd.concat(frames, ignore_index=True)

    keys = frames[0][CHOICE_KEYS]
    baselines = read_baselines(Path(args.baselines), keys)
    print(f"baselines: {len(baselines):,} rows over {len(BASELINES)} baselines "
          f"({len(baselines) // len(BASELINES):,} choices each)")

    combined = drop_excluded(pd.concat(
        [models_frame[['source', *CHOICE_KEYS, 'nll']], baselines],
        ignore_index=True))
    combined = add_positions(combined)
    table = curve(combined)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / 'within_session_curve.csv', index=False)
    print(f"\nwrote {len(table):,} rows to {out / 'within_session_curve.csv'}")

    with plt.rc_context(STYLE):
        figure_overview(table, models, out / 'within_session_overview.png')
        figure_decile(table, models, out / 'within_session_decile.png')
        figure_ratio(table, models, out / 'within_session_ratio.png')
        figure_by_task(table, models, out / 'within_session_by_task.png')
    print(f"wrote 4 figures to {out}")

    summary = overview(table)
    decile = summary[summary['axis'] == 'decile'].pivot_table(
        index='source', columns='bucket', values='nll')
    print("\nL averaged over tasks, by within-session decile "
          "(orientation, not the result):")
    print(decile[[0, 4, 9]].rename(columns={0: 'decile 0', 4: 'decile 4',
                                            9: 'decile 9'})
          .assign(**{'0→9': lambda d: d['decile 9'] - d['decile 0']})
          .round(4).to_string())


if __name__ == '__main__':
    main()
