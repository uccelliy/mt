"""Access to the per-task tables that carry Psych-101's canonical choice codes.

Psych-101 transcripts randomize each participant's response keys, so counting
choices across participants in transcript label space is noise.  The per-task
tables published alongside the dataset do not: they record a canonical
`choice` column that means the same thing for everyone, which is what makes
population-level baselines possible at all.

38 of the test split's experiments have such a table; the rest do not, and no
population baseline exists for them.
"""

from __future__ import annotations

import json
from pathlib import Path
import urllib.request

import pandas as pd

API = "https://huggingface.co/api/datasets/marcelbinz"
RESOLVE = "https://huggingface.co/datasets/marcelbinz"

FAMILIES = (
    "badham2017deficits", "bahrami2020four", "collsioo2023is",
    "enkavi2019adaptivenback", "enkavi2019digitspan", "enkavi2019gonogo",
    "enkavi2019recentprobes", "feng2021dynamics", "flesch2018comparing",
    "frey2017cct", "frey2017risk", "gershman2018deconstructing",
    "gershman2020reward", "hebart2023things", "hilbig2014generalized",
    "jansen2021dunningkruger", "kool2016when", "kool2017cost",
    "lefebvre2017behavioural", "levering2020revisiting", "ludwig2023human",
    "peterson2021using", "plonsky2018when", "ruggeri2022globalizability",
    "sadeghiyeh2020temporal", "schulz2020finding", "somerville2017charting",
    "speekenbrink2008learning", "steingroever2015data", "tomov2021multitask",
    "waltz2020differential", "wilson2014humans", "wu2018generalisation",
    "wu2023chunking", "wulff2018description", "wulff2018sampling",
    "xiong2023neural", "zorowitz2023data",
)

# HF repo names that differ from the test-split family names
FAMILY_RENAMES = {"collsioo2023is": "collsiöö2023MCPL"}

# tasks whose canonical response lives outside the standard column
COLUMN_OVERRIDES = {"zorowitz2023data": ["choice_S1", "choice_S2"]}

def download_tables(cache):
    """Fetch any missing parquet tables into `cache`, one directory per task."""

    cache = Path(cache)
    for family in FAMILIES:
        target = cache / family
        if target.exists() and any(target.rglob("*.parquet")):
            continue
        files = fetch_json(f"{API}/{family}/tree/main?recursive=true")
        parquets = [f['path'] for f in files
                    if f['path'].endswith(".parquet")]
        print(f"downloading {family}: {len(parquets)} file(s)")
        for path in parquets:
            destination = target / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(
                f"{RESOLVE}/{family}/resolve/main/{path}", destination)

def fetch_json(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


# Canonical choice codes are never negative; `-1` / `-1.0` / NaN mark a trial
# with no response (timeout, miss). They are not options and must not inflate
# the denominator of a chance level.
def _is_option(value):
    text = str(value).strip()
    if text.lower() in ("", "nan", "none"):
        return False
    try:
        return float(text) >= 0
    except ValueError:
        return True


# A session's transcript rows are a subset of its table rows, and which subset
# is a property of the task: `wilson2014humans` scores only the free choices of
# a game, several tasks drop the trials the participant missed. Three rules
# cover 47 of the 57 experiments that have a table; the rest need task-specific
# knowledge and are reported as unaligned rather than guessed at.
def _candidate_rows(rows, code_column):
    yield 'raw', rows
    if 'forced' in rows.columns:
        yield 'forced==0', rows[rows['forced'] == 0]
    codes = pd.to_numeric(rows[code_column], errors='coerce').fillna(-1)
    yield 'code>=0', rows[codes >= 0]


def _code(value):
    """A canonical code as a stable string, or None where none was recorded.

    Same rule as `option_counts`: `-1` / NaN mark a trial the participant let
    pass, and a missed trial has no answer to compare a model against. Whole
    numbers lose their float tail so that `1` and `1.0` are one category.
    """

    if not _is_option(value):
        return None
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else text


def _table_paths(cache, experiment):
    """Parquet files behind a test-split experiment id, if the task has any."""

    family, _, name = experiment.partition('/')
    reverse = {v: k for k, v in FAMILY_RENAMES.items()}
    family = reverse.get(family, family)
    directory = Path(cache) / family / name.removesuffix('.csv')
    return sorted(directory.rglob("*.parquet"))


def align_sessions(cache, transcript):
    """Pair every transcript choice with the canonical code of that trial.

    The transcript and the published table hold the same trials in the same
    order, but the transcript carries the participant's randomized key letter
    where the table carries the canonical code. Aligning them row by row is
    what turns one into the other.

    Alignment is only accepted when the two agree on the number of rows, and
    the caller should check the result: within a `(participant, block)` the
    label and the code must stand in one-to-one correspondence, because the
    key assignment is fixed there. A misalignment scrambles the pairing and
    destroys that correspondence, which is what makes it a usable test.

    Returns `(aligned, report)`. `aligned` has one row per transcript choice
    -- experiment, participant, choice_index, block, code, label -- and
    `report` one row per experiment saying how much of it aligned and why not.
    """

    frames, report = [], []
    for experiment, sessions in transcript.groupby('experiment'):
        paths = _table_paths(cache, experiment)
        if not paths:
            report.append({'experiment': experiment, 'note': 'no table'})
            continue
        family = experiment.partition('/')[0]
        columns = COLUMN_OVERRIDES.get(
            {v: k for k, v in FAMILY_RENAMES.items()}.get(family, family),
            ['choice'])
        if len(columns) > 1:
            # two canonical responses per table row; the transcript splits them
            # over two choices, so row counts cannot line up as they stand
            report.append({'experiment': experiment,
                           'note': 'canonical response split over '
                                   f'{len(columns)} columns'})
            continue
        code_column = columns[0]
        table = pd.concat([pd.read_parquet(p) for p in paths],
                          ignore_index=True)
        table['participant'] = table['participant'].astype(str)
        by_participant = dict(list(table.groupby('participant')))
        n_aligned, filters = 0, set()
        for participant, session in sessions.groupby('participant'):
            rows = by_participant.get(participant)
            if rows is None:
                continue
            session = session.sort_values('choice_index')
            for name, candidate in _candidate_rows(rows, code_column):
                if len(candidate) == len(session):
                    break
            else:
                continue
            n_aligned += 1
            filters.add(name)
            block = (candidate['task'] if 'task' in candidate.columns
                     else pd.Series(0, index=candidate.index))
            paired = pd.DataFrame({
                'experiment': experiment,
                'participant': participant,
                'choice_index': session['choice_index'].to_numpy(),
                'block': block.astype(str).to_numpy(),
                'code': [_code(v) for v in candidate[code_column]],
                'label': session['human_choice'].astype(str).to_numpy()})
            frames.append(paired[paired['code'].notna()])
        report.append({'experiment': experiment,
                       'n_sessions': sessions['participant'].nunique(),
                       'n_aligned': n_aligned,
                       'filter': '+'.join(sorted(filters)),
                       'note': '' if n_aligned else 'no session aligned'})
    aligned = (pd.concat(frames, ignore_index=True) if frames
               else pd.DataFrame(columns=['experiment', 'participant',
                                          'choice_index', 'block', 'code',
                                          'label']))
    return aligned, pd.DataFrame(report)


def label_map(aligned, options):
    """The key letter -> canonical code dictionary of each `(participant, block)`.

    Built from the labels the participant actually pressed, then completed by
    elimination: where a block has exactly one canonical code nobody pressed
    and the session offers exactly one label no block of it has claimed, that
    pairing is forced by counting. Without this step every binary task loses
    the sessions where the participant simply never used one of the two keys.

    A block whose observed pairing is not one-to-one failed the alignment test
    and is dropped. `options` supplies the labels a session offers at all
    (experiment, participant, option).

    Returns experiment, participant, block, label, code, source.
    """

    keys = ['experiment', 'participant', 'block']
    grouped = aligned.groupby(keys + ['label'])['code']
    pressed = grouped.agg(n_codes='nunique',
                          code=lambda s: s.mode().iat[0]).reset_index()
    per_block = pressed.groupby(keys).agg(pure=('n_codes', lambda s: (s == 1).all()),
                                          n_label=('code', 'size'),
                                          n_code=('code', 'nunique'))
    good = per_block[per_block['pure'] & (per_block['n_label'] == per_block['n_code'])]
    pressed = pressed.merge(good.reset_index()[keys], on=keys)
    pressed = pressed[keys + ['label', 'code']].assign(source='pressed')

    # what the task allows in this block, pooled over participants: canonical
    # codes mean the same thing for everyone, which is the whole point of them
    block_codes = (aligned.groupby(['experiment', 'block'])['code']
                   .agg(set).to_dict())
    offered = (options.groupby(['experiment', 'participant'])['option']
               .agg(set).to_dict())
    used = pressed.groupby(keys).agg(codes=('code', set), labels=('label', set))

    filled = []
    for (experiment, participant, block), row in used.iterrows():
        free = block_codes.get((experiment, block), set()) - row['codes']
        # candidates are the keys this block has not spoken for; a label
        # belonging to another block leaves more than one and the rule stays
        # silent, which is what keeps a multi-block task from being guessed at
        spare = offered.get((experiment, participant), set()) - row['labels']
        if len(free) == 1 and len(spare) == 1:
            filled.append({'experiment': experiment, 'participant': participant,
                           'block': block, 'label': spare.pop(),
                           'code': free.pop(), 'source': 'elimination'})
    if filled:
        pressed = pd.concat([pressed, pd.DataFrame(filled)], ignore_index=True)
    return pressed


def option_counts(cache):
    """How many distinct canonical responses each experiment allows.

    Keyed by the test-split experiment id (`badham2017deficits/exp1.csv`).
    This is the union over every participant in canonical code space, which is
    what makes it usable as a chance level: transcript labels are randomized
    per participant, so a session's own `<<...>>` union both misses options the
    participant never pressed and cannot be compared across people.

    **It is still a union, not a per-trial legal set.** Where a task shows a
    subset of its responses on each trial the two differ enormously --
    `hebart2023things` has 1,823 canonical objects and offers 3 per trial --
    so callers must check the count against the task before using it.
    """

    counts = {}
    for family in (path for path in sorted(Path(cache).iterdir())
                   if path.is_dir()):
        columns = COLUMN_OVERRIDES.get(family.name, ["choice"])
        name = FAMILY_RENAMES.get(family.name, family.name)
        for experiment in sorted(p for p in family.iterdir() if p.is_dir()):
            values = set()
            for path in sorted(experiment.rglob("*.parquet")):
                table = pd.read_parquet(path, columns=columns)
                for column in columns:
                    values |= set(table[column].astype(str).unique())
            options = {v for v in values if _is_option(v)}
            if options:
                counts[f"{name}/{experiment.name}.csv"] = len(options)
    return counts
