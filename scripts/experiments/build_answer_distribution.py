"""A13: the model's answer distribution against the human's, per task.

Zero GPU. Two readouts, reported side by side and never merged:

  (1) the frequencies of `pred_choice` -- "take greedy decoding as a simulator
      and look at the behaviour it produces". This is the degeneracy test of
      design section 2.3: a model that always names the mode and a perfectly
      calibrated one score the same accuracy (A15), and only the distribution
      separates them. Read it knowing that an argmax readout is *structurally*
      more concentrated than a human; that is not a defect.

  (2) the expected frequencies implied by `pred_options`, f(o) = mean_j P_j(o).
      Same units as an empirical frequency, so it can sit next to the human's.

Both live in canonical code space. Psych-101 randomizes each participant's
response keys, so pooling transcript labels across people stacks options that
mean different things. `canonical_tables.align_sessions` pairs every choice
with the published table's canonical code and `label_map` turns that into the
key dictionary of each block; sessions that fail the one-to-one test, and the
experiments that need task-specific row filters, are reported as uncovered
rather than guessed at.

Two things the option list is not:

  not the trial's own options   the scorer froze the *session* union as the
                                legal set, so a trial of a multi-block task
                                lists keys belonging to other blocks too. Their
                                probability is pulled out into `off_block_mass`
                                rather than folded into the distribution -- it
                                answers a different question ("how much does
                                the model bet on keys this trial does not use").

  not the same coverage twice   (1) needs a single-token answer, (2) needs
                                `options_status='scored'`. The two subsets
                                differ, so each is aggregated over its own
                                trials and reported with its own coverage.

Distance is total variation, 0.5 * sum |human - model|, taken inside a session
and then averaged over participants (design section 4). It is reported against
two resampling nulls, because a simulator drawing from the human's own
frequencies does not score zero on a finite session: `null_pred` draws both
sides afresh (the null for (1), which samples), `null_options` draws one side
(the null for (2), which does not).

The null is a reference, **not a lower bound**. A model reads the same history
the human did, so it is not an independent sampler and can land below it -- and
most tasks do. That is the interesting direction: it says the model is tracking
the trial, not just reproducing the session's marginals.

    python scripts/experiments/build_answer_distribution.py \
        --run centaur8b=outputs/runs/<tag>/4bit/centaur8b/e3/full \
        ... \
        --output-dir outputs/analysis/<tag>/a13
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _plotting import GRID, INK, MUTED, STYLE, short_name, style_axis
from mt.evaluation.aggregate import KEYS, drop_excluded
from mt.models.baselines.canonical_tables import align_sessions, label_map

CHOICE = ['experiment', 'participant', 'choice_index']
NEEDED = ['condition', 'experiment', 'participant', 'choice_index',
          'pred_choice', 'human_choice', 'num_tokens', 'options_status']
OPTION_NEEDED = ['condition', 'experiment', 'participant', 'choice_index',
                 'option', 'logprob']
SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
TOP_CODES = 10
DRAWS = 64
RNG = np.random.default_rng(0)


def parse_run(text):
    label, _, path = text.partition('=')
    if not label or not path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {text!r}")
    return label, Path(path)


def read_table(path, name, columns, condition):
    if path.is_dir():
        for candidate in (f'{name}.csv', f'{name}.csv.gz'):
            if (path / candidate).exists():
                path = path / candidate
                break
        else:
            raise SystemExit(f"No {name} table in {path}")
    frame = pd.read_csv(path, usecols=columns, low_memory=False)
    frame = frame[frame['condition'] == condition].drop(columns='condition')
    frame['participant'] = frame['participant'].astype(str)
    return drop_excluded(frame)


def session_frequencies(frame, column, weight=None):
    """Share of a session's trials falling on each canonical code."""

    # trials, not rows: (2) carries one row per option of a trial
    total = frame.groupby(KEYS)['choice_index'].transform('nunique')
    part = frame.assign(share=(1.0 if weight is None else frame[weight]) / total)
    return (part.groupby(KEYS + [column])['share'].sum()
            .rename('share').reset_index().rename(columns={column: 'code'}))


def total_variation(human, model):
    """0.5 * sum |h - m| per session, over the union of the two supports."""

    joined = human.merge(model, on=KEYS + ['code'], how='outer',
                         suffixes=('_human', '_model')).fillna(0.0)
    gap = (joined['share_human'] - joined['share_model']).abs()
    return (0.5 * gap.groupby([joined['experiment'], joined['participant']])
            .sum()).rename('tvd')


def resampling_null(human, name, two_sided):
    """What total variation a perfect simulator still scores on J trials.

    The two readouts need different nulls: (1) is itself J draws, so both sides
    move and the null compares two fresh samples; (2) averages probabilities
    and carries no sampling noise of its own, so only the human side is drawn.
    Each is computed on its own subset of trials, which is why this is called
    once per readout rather than sharing one column.
    """

    rows = []
    for (experiment, participant), part in human.groupby(KEYS):
        probability = part['share'].to_numpy()
        probability = probability / probability.sum()
        trials = int(round(part['n_choices'].iat[0]))
        first = RNG.multinomial(trials, probability, size=DRAWS) / trials
        other = (RNG.multinomial(trials, probability, size=DRAWS) / trials
                 if two_sided else probability)
        rows.append({'experiment': experiment, 'participant': participant,
                     name: float(np.abs(first - other).sum(axis=1).mean() / 2)})
    return pd.DataFrame(rows)


def readout_pred(frame, codes, aligned):
    """(1): where greedy decoding put the session's choices."""

    single = frame[frame['num_tokens'] == 1].merge(aligned, on=CHOICE)
    named = single.merge(
        codes.rename(columns={'label': 'pred_choice', 'code': 'pred_code'}),
        on=['experiment', 'participant', 'block', 'pred_choice'], how='left')
    # a label the dictionary knows from another block of the same session is a
    # different failure from one it has never seen; A8/A14 own the second
    elsewhere = (codes.rename(columns={'label': 'pred_choice'})
                 [['experiment', 'participant', 'pred_choice']]
                 .assign(seen=True).drop_duplicates())
    named = named.merge(elsewhere, on=['experiment', 'participant',
                                       'pred_choice'], how='left')
    named['seen'] = named['seen'].eq(True)
    named['bucket'] = np.where(named['pred_code'].notna(), 'code',
                               np.where(named['seen'], 'off_block', 'off_set'))

    human = session_frequencies(named, 'code')
    scored = named[named['bucket'] == 'code']
    model = session_frequencies(scored, 'pred_code')
    # renormalize over the trials the dictionary could name, so that both
    # vectors are distributions; the discarded share is reported beside them
    model['share'] /= model.groupby(KEYS)['share'].transform('sum')
    shares = (named.assign(off_block=named['bucket'] == 'off_block',
                           off_set=named['bucket'] == 'off_set')
              .groupby(KEYS)[['off_block', 'off_set']].mean())
    counts = named.groupby(KEYS).size().rename('n_choices')
    return human.merge(counts.reset_index(), on=KEYS), model, shares


def readout_options(frame, options, codes, aligned):
    """(2): the frequencies the model's own probabilities imply."""

    trials = frame[frame['options_status'] == 'scored'][CHOICE]
    part = (options.merge(trials, on=CHOICE)
            .merge(aligned, on=CHOICE)
            .merge(codes, left_on=['experiment', 'participant', 'block',
                                   'option'],
                   right_on=['experiment', 'participant', 'block', 'label'],
                   how='left'))
    part['p'] = np.exp(part['logprob'])
    legal = part.groupby(CHOICE)['p'].transform('sum')
    inside = part[part['code_y'].notna()].copy()
    inside['share'] = inside['p'] / inside.groupby(CHOICE)['p'].transform('sum')

    human = session_frequencies(part.drop_duplicates(CHOICE), 'code_x')
    counts = part.drop_duplicates(CHOICE).groupby(KEYS).size().rename('n_choices')
    model = session_frequencies(inside, 'code_y', weight='share')
    model['share'] /= model.groupby(KEYS)['share'].transform('sum')
    off = (part.assign(off=np.where(part['code_y'].notna(), 0.0, part['p'] / legal))
           .groupby(CHOICE)['off'].sum().groupby(KEYS).mean()
           .rename('off_block_mass'))
    return human.merge(counts.reset_index(), on=KEYS), model, off


def by_task(pred, options, nulls, shares, off, human_pred, human_options,
            pred_vector, options_vector):
    """Section 4: mean over a task's participants, never over its trials."""

    frame = (pred.rename('tvd_pred').to_frame()
             .join(options.rename('tvd_options'), how='outer')
             .join([null.set_index(KEYS) for null in nulls])
             .join(shares).join(off))
    # the degeneracy readout of section 2.3: a model that always names the mode
    # is invisible in accuracy and obvious here. (1) is structurally the more
    # concentrated of the two -- an argmax cannot hedge -- so it is only
    # comparable against the human on the same subset, never against (2)
    for name, part in (('human_top_pred', human_pred), ('pred_top', pred_vector),
                       ('human_top_options', human_options),
                       ('options_top', options_vector)):
        frame = frame.join(part.groupby(KEYS)['share'].max().rename(name))
    grouped = frame.groupby('experiment')
    table = grouped.mean(numeric_only=True)
    table['n_participants'] = grouped.size()
    table['n_pred'] = frame['tvd_pred'].notna().groupby(
        frame.index.get_level_values('experiment')).sum()
    table['n_options'] = frame['tvd_options'].notna().groupby(
        frame.index.get_level_values('experiment')).sum()
    return table.reset_index()


def task_distribution(human_pred, pred, human_options, options):
    """The vectors a task is read from, averaged over participants.

    Each readout is paired with the human vector computed on **its own** subset
    of trials. (1) is defined where the answer is a single token and (2) where
    the options were scored; on a task that mixes the two, comparing one
    model's readout against the other's human baseline compares different
    trials -- which showed up as a population-level gap larger than the
    per-session one, something total variation cannot do.

    The average runs over **every** session of the task, so a code one
    participant never used counts as a zero for them rather than dropping out
    of its own denominator. Taking the mean of the rows present instead would
    let each code carry a different divisor, and the task vector would not sum
    to one -- on `ruggeri2022globalizability` it came to 1.024.
    """

    def average(frame, name):
        sessions = frame.groupby('experiment')['participant'].nunique()
        total = frame.groupby(['experiment', 'code'])['share'].sum()
        return total.div(sessions, level='experiment').rename(name)

    frame = (average(human_pred, 'human_pred').to_frame()
             .join(average(pred, 'pred'), how='outer')
             .join(average(human_options, 'human_options'), how='outer')
             .join(average(options, 'options'), how='outer'))
    return frame.reset_index()


def figure_distributions(distributions, models, column, output, title):
    """One panel per task: the human as bars, each model as a marker."""

    tasks = sorted(distributions[models[0]]['experiment'].unique())
    # each readout has its own human baseline: they are defined on different
    # subsets of trials wherever a task mixes single- and multi-token answers
    reference = f'human_{column}'
    cols = 6
    rows = -(-len(tasks) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 1.75 * rows),
                             sharey=True)
    for ax, task in zip(axes.ravel(), tasks):
        base = distributions[models[0]]
        part = base[base['experiment'] == task].sort_values(
            reference, ascending=False).head(TOP_CODES)
        order = list(part['code'])
        x = np.arange(len(order))
        ax.bar(x, part[reference], color=GRID, width=0.78, zorder=1)
        for color, model in zip(SERIES_COLORS, models):
            frame = distributions[model]
            frame = frame[frame['experiment'] == task].set_index('code')
            values = [frame[column].get(code, np.nan) for code in order]
            ax.scatter(x, values, s=9, color=color, zorder=3)
        ax.set_title(short_name(task), fontsize=5.6, pad=2)
        ax.set_xticks(x, order, fontsize=4.6)
        ax.set_ylim(0, 1.03)
        ax.tick_params(labelsize=5)
        style_axis(ax)
        ax.grid(axis='x', visible=False)
    for ax in axes.ravel()[len(tasks):]:
        ax.set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GRID)] + [
        plt.Line2D([], [], color=c, marker='o', lw=0, markersize=4)
        for c in SERIES_COLORS[:len(models)]]
    fig.legend(handles, ['human'] + list(models), frameon=False, fontsize=8,
               ncol=7, loc='lower center')
    fig.suptitle(title, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.022, 1, 0.975))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def figure_distance(tasks, models, output):
    """Per task, how far each model's distribution sits from the human's."""

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 0.19 * tasks['experiment']
                                            .nunique() + 1.4), sharey=True)
    panels = [('tvd_pred', 'null_pred', '(1)  greedy readout'),
              ('tvd_options', 'null_options', '(2)  expected from P(option)')]
    order = list(tasks.groupby('experiment')['tvd_pred'].mean()
                 .sort_values().index)
    for ax, (value, floor, title) in zip(axes, panels):
        wide = tasks.pivot_table(index='experiment', columns='model',
                                 values=value).reindex(order)
        base = tasks.groupby('experiment')[floor].mean().reindex(order)
        y = np.arange(len(order))
        for row, task in enumerate(order):
            values = [v for v in list(wide.loc[task]) + [base[task]]
                      if not np.isnan(v)]
            if values:
                ax.hlines(row, min(values), max(values), color=GRID, lw=0.9,
                          zorder=1)
        ax.scatter(base, y, marker='|', s=90, color=MUTED, zorder=3,
                   label='i.i.d. null')
        for color, model in zip(SERIES_COLORS, models):
            if model in wide:
                ax.scatter(wide[model], y, s=15, color=color, zorder=2,
                           label=model)
        ax.set_yticks(y, [short_name(t) for t in order], fontsize=6.2)
        ax.set_ylim(-1, len(order))
        ax.invert_yaxis()
        ax.tick_params(axis='y', length=0)
        ax.set_xlim(0, 1.06 * max(tasks[value].max(), tasks[floor].max()))
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel('total variation from the human distribution', fontsize=8)
        style_axis(ax)
        ax.grid(axis='y', visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    height = fig.get_size_inches()[1]
    fig.legend(handles, labels, frameon=False, fontsize=8, ncol=7,
               loc='upper center', bbox_to_anchor=(0.5, 1 - 0.42 / height))
    fig.suptitle('A13  Distance from the human answer distribution, per task',
                 fontsize=11, color=INK, y=1 - 0.12 / height)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.78 / height))
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

    models = [label for label, _ in args.run]
    if len(models) > len(SERIES_COLORS):
        raise SystemExit(f"{len(models)} runs but only {len(SERIES_COLORS)} "
                         "validated hues")

    # the key dictionary is a property of the data, not of any model: build it
    # once from the first run and hold every model to it
    first = read_table(args.run[0][1], 'predictions', NEEDED, args.condition)
    first_options = read_table(args.run[0][1], 'pred_options', OPTION_NEEDED,
                               args.condition)
    aligned, report = align_sessions(args.tables, first)
    codes = label_map(aligned, first_options)
    covered = sorted(set(codes['experiment']))
    print(f"{len(report)} experiments have a canonical table; "
          f"{len(covered)} of them aligned and produced a key dictionary")
    print(report[report['note'] != ''].to_string(index=False))
    print(f"\nkey dictionary: {len(codes):,} entries, "
          f"{(codes['source'] == 'elimination').mean():.1%} filled by "
          "elimination")
    # the one-to-one test only bites where the participant used more than one
    # key: a session spent pressing a single key pairs with anything
    tested = aligned.groupby(KEYS + ['block'])['label'].nunique()
    print(f"the one-to-one test has teeth on {(tested > 1).mean():.1%} of "
          f"blocks ({(tested == 1).sum():,} of {len(tested):,} saw one key "
          "only, and are carried by the rest of their task)")

    aligned = aligned[aligned['experiment'].isin(covered)]
    codes = codes.drop(columns='source')
    blocks = aligned[CHOICE + ['block', 'code']]

    distributions, tasks = {}, []
    for label, path in args.run:
        frame = (first if label == models[0]
                 else read_table(path, 'predictions', NEEDED, args.condition))
        options = (first_options if label == models[0]
                   else read_table(path, 'pred_options', OPTION_NEEDED,
                                   args.condition))
        frame = frame[frame['experiment'].isin(covered)]
        options = options[options['experiment'].isin(covered)]

        human_pred, model_pred, shares = readout_pred(frame, codes, blocks)
        human_options, model_options, off = readout_options(
            frame, options, codes, blocks)
        nulls = [resampling_null(human_pred, 'null_pred', two_sided=True),
                  resampling_null(human_options, 'null_options',
                                   two_sided=False)]

        table = by_task(total_variation(human_pred.drop(columns='n_choices'),
                                        model_pred),
                        total_variation(human_options.drop(columns='n_choices'),
                                        model_options),
                        nulls, shares, off, human_pred, human_options,
                        model_pred, model_options)
        table.insert(0, 'model', label)
        tasks.append(table)
        distribution = task_distribution(human_pred, model_pred,
                                         human_options, model_options)
        distribution.insert(0, 'model', label)
        distributions[label] = distribution
        print(f"{label}: {table['n_participants'].sum():,} sessions over "
              f"{len(table)} tasks; (1) covers "
              f"{table['n_pred'].sum() / table['n_participants'].sum():.1%} of "
              f"them, (2) {table['n_options'].sum() / table['n_participants'].sum():.1%}")

    tasks = pd.concat(tasks, ignore_index=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tasks.to_csv(out / 'a13_by_task.csv', index=False)
    pd.concat(distributions.values(), ignore_index=True).to_csv(
        out / 'a13_distribution.csv', index=False)
    report.to_csv(out / 'a13_alignment.csv', index=False)
    with plt.rc_context(STYLE):
        figure_distance(tasks, models, out / 'a13_distance.png')
        figure_distributions(distributions, models, 'pred',
                             out / 'a13_distribution_pred.png',
                             'A13  (1) greedy readout against the human, '
                             'per task')
        figure_distributions(distributions, models, 'options',
                             out / 'a13_distribution_options.png',
                             'A13  (2) expected frequencies against the human, '
                             'per task')
    print(f"\nwrote 3 tables and 3 figures to {out}")

    summary = tasks.groupby('model')[
        ['tvd_pred', 'null_pred', 'tvd_options', 'null_options',
         'off_block', 'off_set', 'off_block_mass', 'human_top_pred',
         'pred_top', 'human_top_options', 'options_top']].mean()
    print("\naveraged over tasks (orientation; the per-task table is the "
          "result):")
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
