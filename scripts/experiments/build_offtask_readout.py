"""A9: what the model actually emitted when its readout was not a legal option.

Zero GPU. `format_ok` is a boolean over a one-token readout, and on its own it
badly overstates "the model was not answering". Two things have to be split off
before the number means anything:

  the human's choice was multi-token
      (1) reads one token and `format_ok` compares it to the whole option
      string, so `77` can never equal `77.37`. Structurally false, every time
      (design section 2.2, section 9 clause 8) -- not a failure to answer.

  the model's chosen option is multi-token
      `tur` for `turquoise`, `te` for `teal`. The model is answering; the
      readout just stops after one token. Detected by testing the emission
      against the prefixes of the session's legal options.

What is left -- ` to`, ` and`, ` button`, `.\n` -- is the real thing: the model
carrying on writing prose instead of answering. Section 2.3's whole argument is
that this is invisible in the option probabilities and obvious in the raw
output, so this analysis reads the strings.

The legal set is the union of the session's own `<<...>>` markers, which is the
same lower bound the scorer used (section 9 clause 5). Because it is a lower
bound, every emission is also tested against the task-wide union, and both
counts are reported: an emission that is off-task for the session but legal for
the task is an option this participant simply never used.

    python scripts/experiments/build_offtask_readout.py \
        --run centaur8b=outputs/runs/<tag>/4bit/centaur8b/e3/full \
        ... \
        --prompts data/psych-101-test/prompts_testing_t1.jsonl \
        --output-dir outputs/analysis/<tag>/a9
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import string

import pandas as pd

from mt.evaluation.aggregate import drop_excluded
from mt.models.llm.supervision import find_target_spans

NEEDED = ['condition', 'experiment', 'participant', 'choice_index',
          'num_tokens', 'raw_generation', 'human_choice', 'format_ok']
EXAMPLES_PER_CELL = 12

# Ordered: the first test that fires wins, so a stripped exact match is never
# also counted as a prefix.
CATEGORIES = [
    'human_multi_token',   # (1) cannot match by construction -- not a failure
    'exact',               # legal already; present only as a cross-check
    'whitespace_variant',  # a legal option with surrounding whitespace
    'case_variant',        # a legal option in different case
    'option_prefix',       # first token of a legal but multi-token option
    'numeric_off_support',  # a number where the human also gave a number
    'whitespace_only',
    'punctuation_only',
    'off_task',            # the real one: prose instead of an answer
]


def is_numeric(text):
    try:
        float(text.strip())
    except (ValueError, AttributeError):
        return False
    return True


def parse_run(text):
    label, _, path = text.partition('=')
    if not label or not path:
        raise argparse.ArgumentTypeError(f"expected label=path, got {text!r}")
    return label, Path(path)


def read_predictions(path):
    if path.is_dir():
        for name in ('predictions.csv', 'predictions.csv.gz'):
            if (path / name).exists():
                path = path / name
                break
        else:
            raise SystemExit(f"No predictions table in {path}")
    frame = pd.read_csv(path, usecols=NEEDED, low_memory=False)
    frame['participant'] = frame['participant'].astype(str)
    frame['raw_generation'] = frame['raw_generation'].fillna('')
    frame['format_ok'] = frame['format_ok'].astype(str).str.lower().eq('true')
    return frame


def option_sets(prompts):
    """The `<<...>>` union per session, and the wider union per task.

    This is the same lower bound the scorer froze as `option_support`; the
    task-wide union is strictly larger and bounds how much of `off_task` is
    really "an option this participant never happened to use".
    """

    per_session, per_task = {}, {}
    with open(prompts) as handle:
        for line in handle:
            row = json.loads(line)
            text = row['text']
            options = {text[start:end] for start, end in find_target_spans(text)}
            key = (row['experiment'], str(row['participant']))
            per_session[key] = options
            per_task.setdefault(row['experiment'], set()).update(options)
    return per_session, per_task


def classify(emission, options, human_choice):
    """Name what one readout is, relative to a legal option set.

    `numeric_off_support` is keyed on the human's own answer at this trial
    rather than on the task: 18 of the 75 experiments mix key presses with
    numeric ratings, so "is this a numeric task" has no answer at the task
    level, while "was a number called for here" always does.
    """

    if emission in options:
        return 'exact'
    stripped = emission.strip()
    if stripped and stripped in options:
        return 'whitespace_variant'
    if stripped and any(stripped.lower() == o.lower() for o in options):
        return 'case_variant'
    if stripped and any(o.startswith(stripped) for o in options):
        return 'option_prefix'
    if is_numeric(stripped) and is_numeric(human_choice):
        return 'numeric_off_support'
    if emission and not stripped:
        return 'whitespace_only'
    if stripped and all(c in string.punctuation for c in stripped):
        return 'punctuation_only'
    return 'off_task'


def classify_frame(frame, per_session, per_task):
    """Label every not-format_ok readout, twice over: session then task union."""

    work = frame[~frame['format_ok']].copy()
    human = work['human_choice'].fillna('').astype(str)
    session_key = list(zip(work['experiment'], work['participant']))
    work['category'] = [
        'human_multi_token' if tokens > 1
        else classify(emission, per_session.get(key, set()), choice)
        for emission, tokens, key, choice
        in zip(work['raw_generation'], work['num_tokens'], session_key, human)
    ]
    # the same call against the task-wide union, to bound the lower-bound
    work['category_task_union'] = [
        'human_multi_token' if tokens > 1
        else classify(emission, per_task.get(experiment, set()), choice)
        for emission, tokens, experiment, choice
        in zip(work['raw_generation'], work['num_tokens'], work['experiment'],
               human)
    ]
    return work


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', action='append', type=parse_run, required=True,
                        metavar='LABEL=PATH')
    parser.add_argument('--prompts', required=True,
                        help='The scored transcripts, for the legal option sets')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--condition', default='full',
                        help="Condition to analyse, or 'all' for every one")
    args = parser.parse_args()

    per_session, per_task = option_sets(args.prompts)
    print(f"legal option sets: {len(per_session):,} sessions, "
          f"{len(per_task)} tasks")

    counts, examples, totals = [], [], []
    for label, path in args.run:
        frame = drop_excluded(read_predictions(path))
        if args.condition != 'all':
            frame = frame[frame['condition'] == args.condition]
        work = classify_frame(frame, per_session, per_task)

        totals.append({
            'model': label,
            'choices': len(frame),
            'not_format_ok': int((~frame['format_ok']).sum()),
            'human_multi_token': int((work['category'] == 'human_multi_token').sum()),
            'off_task': int((work['category'] == 'off_task').sum()),
            'off_task_even_for_task_union':
                int((work['category_task_union'] == 'off_task').sum()),
        })
        grouped = work.groupby(['condition', 'category',
                                'category_task_union']).size()
        table = grouped.rename('n').reset_index()
        table.insert(0, 'model', label)
        counts.append(table)

        for category, part in work.groupby('category'):
            sample = part.sample(min(EXAMPLES_PER_CELL, len(part)),
                                 random_state=0)
            sample = sample[['experiment', 'participant', 'choice_index',
                             'human_choice', 'raw_generation',
                             'category_task_union']].copy()
            sample.insert(0, 'category', category)
            sample.insert(0, 'model', label)
            examples.append(sample)
        print(f"{label}: {len(frame):,} choices, "
              f"{len(work):,} not format_ok")

    counts = pd.concat(counts, ignore_index=True)
    examples = pd.concat(examples, ignore_index=True)
    totals = pd.DataFrame(totals)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out / 'a9_categories.csv', index=False)
    examples.to_csv(out / 'a9_examples.csv', index=False)
    totals.to_csv(out / 'a9_totals.csv', index=False)
    print(f"\nwrote 3 tables to {out}")

    order = [c for c in CATEGORIES if c in set(counts['category'])]
    breakdown = (counts.pivot_table(index='model', columns='category',
                                    values='n', aggfunc='sum')
                 .reindex(columns=order).fillna(0).astype(int))
    print(f"\nnot-format_ok readouts by kind (condition={args.condition}):")
    print(breakdown.to_string())

    totals = totals.set_index('model')
    totals['off_task_rate'] = totals['off_task'] / totals['choices']
    totals['vs_task_union_rate'] = (totals['off_task_even_for_task_union']
                                    / totals['choices'])
    print("\nthe one that means 'was not answering', as a rate over all "
          "scored choices:")
    print(totals[['choices', 'not_format_ok', 'off_task', 'off_task_rate',
                  'off_task_even_for_task_union', 'vs_task_union_rate']]
          .to_string(float_format=lambda v: f"{v:.4%}"))


if __name__ == '__main__':
    main()
