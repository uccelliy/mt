"""A14: how much probability the model puts on the legal option set.

Zero GPU. `format_ok` (A8/A9) asks whether the single most likely token was a
legal option -- a boolean over one point of the distribution. `legal_mass` asks
how much of the whole distribution landed inside the legal set. A readout can
be `format_ok=True` while four fifths of the mass sits outside, which reads as
"guessed the mode of a task it was not really answering".

Two cuts:

  by position   `legal_mass` against `choice_index`, the mechanism behind the
                cold start in A17: at the first choice of a session the base
                models put as little as 5% of their mass on legal options, so
                they score worse than uniform -- not because they are wrong
                about the human but because they are not answering yet. One
                in-context example fixes most of it.

  by task       the per-task distribution, plotted as leaked mass
                (1 - legal_mass) on a log axis: most tasks sit above 0.999 and
                a linear axis crushes them into the right edge.

Coverage: `legal_mass` exists only where (3) was scored, which excludes the
experiments carrying a multi-token option (section 9 clause 9). Those tasks are
absent from the tables rather than counted as zero.

    python scripts/experiments/build_legal_mass.py \
        --run centaur8b=outputs/runs/<tag>/4bit/centaur8b/e3/full \
        ... \
        --output-dir outputs/analysis/<tag>/a14
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from _plotting import INK, STYLE, short_name, style_axis
from mt.evaluation.aggregate import drop_excluded

NEEDED = ['condition', 'experiment', 'participant', 'choice_index',
          'legal_mass', 'options_status']
N_POSITIONS = 30
# The same fixed, validated hue order used elsewhere; never cycled.
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']


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
    return drop_excluded(frame)


def by_position(frame):
    """Mean legal_mass at each of the first N_POSITIONS choices."""

    part = frame[frame['choice_index'] < N_POSITIONS]
    grouped = part.groupby('choice_index')['legal_mass']
    return pd.DataFrame({'legal_mass': grouped.mean(),
                         'n': grouped.size()}).reset_index()


def by_task(frame):
    """Per-task mean legal_mass, with the coverage it was computed from."""

    scored = frame[frame['legal_mass'].notna()]
    grouped = scored.groupby('experiment')['legal_mass']
    table = pd.DataFrame({
        'legal_mass': grouped.mean(),
        'legal_mass_p05': grouped.quantile(0.05),
        'n_scored': grouped.size(),
    })
    table['n_choices'] = frame.groupby('experiment').size()
    table['coverage'] = table['n_scored'] / table['n_choices']
    return table.reset_index()


def figure_position(positions, output):
    """The recovery curve: how fast the mass returns to the legal set."""

    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    for color, (label, part) in zip(SERIES_COLORS, positions.items()):
        part = part.sort_values('choice_index')
        ax.plot(part['choice_index'], part['legal_mass'], color=color, lw=2,
                label=label, solid_capstyle='round', marker='o', markersize=3.4)
    ax.set_xlabel('choice index within the session', fontsize=9)
    ax.set_ylabel('probability mass on the legal option set', fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc='lower right')
    style_axis(ax)
    fig.suptitle('A14  Recovery of the answering format — one in-context '
                 'example does most of the work', fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def figure_by_task(frames, models, output):
    """One panel per task -- the position curve without averaging tasks away.

    The cross-task curve (figure one) hides that the dip is not a property of
    models at all: it is one experiment changing what it asks for partway
    through. Only the per-task panels show which.
    """

    tasks = sorted({t for frame in frames.values() for t in frame['experiment']})
    cols = 7
    rows = -(-len(tasks) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 1.55 * rows),
                             sharex=True, sharey=True)
    for ax, task in zip(axes.ravel(), tasks):
        for color, model in zip(SERIES_COLORS, models):
            part = frames[model]
            line = (part[part['experiment'] == task]
                    .groupby('choice_index')['legal_mass'].mean())
            ax.plot(line.index, line.to_numpy(), color=color, lw=1.1)
        ax.set_title(short_name(task), fontsize=5.6, pad=2)
        ax.set_ylim(0, 1.03)
        ax.tick_params(labelsize=5)
        style_axis(ax)
    for ax in axes.ravel()[len(tasks):]:
        ax.set_visible(False)
    handles = [plt.Line2D([], [], color=c, lw=2) for c in SERIES_COLORS]
    fig.legend(handles, list(models), frameon=False, fontsize=8, ncol=6,
               loc='lower center')
    fig.suptitle('A14  Per task \u2014 probability mass on the legal option set '
                 'by choice index', fontsize=12, color=INK)
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
    parser.add_argument('--condition', default='full')
    args = parser.parse_args()

    models = [label for label, _ in args.run]
    if len(models) > len(SERIES_COLORS):
        raise SystemExit(f"{len(models)} runs but only {len(SERIES_COLORS)} "
                         "validated hues")

    positions, tasks, frames = {}, [], {}
    for label, path in args.run:
        frame = read_run(path, args.condition)
        frames[label] = frame[frame['choice_index'] < N_POSITIONS][
            ['experiment', 'choice_index', 'legal_mass']].dropna()
        positions[label] = by_position(frame)
        table = by_task(frame)
        table.insert(0, 'model', label)
        tasks.append(table)
        missing = frame['legal_mass'].isna().sum()
        print(f"{label}: {len(frame):,} choices, "
              f"{missing:,} without legal_mass "
              f"({frame.loc[frame['legal_mass'].isna(), 'experiment'].nunique()} "
              "tasks have none -- multi-token options, clause 9)")
    tasks = pd.concat(tasks, ignore_index=True)
    position_table = pd.concat(
        [part.assign(model=label) for label, part in positions.items()],
        ignore_index=True)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(out / 'a14_by_task.csv', index=False)
    position_table.to_csv(out / 'a14_by_position.csv', index=False)
    with plt.rc_context(STYLE):
        figure_position(positions, out / 'a14_recovery.png')
        figure_by_task(frames, models, out / 'a14_by_task.png')
    print(f"\nwrote 2 tables and 2 figures to {out}")

    early = position_table[position_table['choice_index'] < 6]
    print("\nlegal_mass at the first choices:")
    print(early.pivot_table(index='model', columns='choice_index',
                            values='legal_mass').round(3).to_string())
    worst = (tasks.pivot_table(index='experiment', columns='model',
                               values='legal_mass')[models])
    print(f"\nlowest legal_mass tasks ({len(worst)} tasks have (3) scored):")
    print(worst.nsmallest(8, models[0]).round(4).to_string())


if __name__ == '__main__':
    main()
