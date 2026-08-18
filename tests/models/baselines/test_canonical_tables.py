"""Tests for pairing transcript key letters with canonical choice codes."""

from __future__ import annotations

import pandas as pd
import pytest

from mt.models.baselines import canonical_tables as ct


@pytest.fixture
def table(tmp_path):
    """A two-block task: the same codes, a different key pair per block."""

    def write(rows):
        directory = tmp_path / 'toy2020task' / 'exp1'
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(directory / 'train.parquet')
        return tmp_path

    return write


def transcript(rows):
    return pd.DataFrame(rows, columns=['experiment', 'participant',
                                       'choice_index', 'human_choice'])


def test_alignment_pairs_rows_in_order(table):
    cache = table({'participant': ['0'] * 4, 'task': [0, 0, 1, 1],
                   'choice': [0, 1, 0, 1]})
    aligned, report = ct.align_sessions(cache, transcript([
        ('toy2020task/exp1.csv', '0', 0, 'C'),
        ('toy2020task/exp1.csv', '0', 1, 'N'),
        ('toy2020task/exp1.csv', '0', 2, 'X'),
        ('toy2020task/exp1.csv', '0', 3, 'Y')]))

    assert list(aligned['code']) == ['0', '1', '0', '1']
    assert list(aligned['block']) == ['0', '0', '1', '1']
    assert report.loc[0, 'n_aligned'] == 1


def test_session_of_the_wrong_length_is_reported_not_guessed(table):
    cache = table({'participant': ['0'] * 4, 'task': [0] * 4,
                   'choice': [0, 1, 0, 1]})
    aligned, report = ct.align_sessions(cache, transcript([
        ('toy2020task/exp1.csv', '0', 0, 'C'),
        ('toy2020task/exp1.csv', '0', 1, 'N')]))

    assert aligned.empty
    assert report.loc[0, 'note'] == 'no session aligned'


def test_forced_trials_are_dropped_to_recover_the_length(table):
    cache = table({'participant': ['0'] * 4, 'task': [0] * 4,
                   'forced': [1, 0, 1, 0], 'choice': [0, 1, 0, 1]})
    aligned, report = ct.align_sessions(cache, transcript([
        ('toy2020task/exp1.csv', '0', 0, 'N'),
        ('toy2020task/exp1.csv', '0', 1, 'N')]))

    assert list(aligned['code']) == ['1', '1']
    assert report.loc[0, 'filter'] == 'forced==0'


def test_missed_trials_are_not_options(table):
    """`-1` and NaN mark a trial that got no answer, so there is none to score."""

    cache = table({'participant': ['0'] * 3, 'task': [0] * 3,
                   'choice': [0, -1, 1]})
    aligned, _ = ct.align_sessions(cache, transcript([
        ('toy2020task/exp1.csv', '0', 0, 'C'),
        ('toy2020task/exp1.csv', '0', 1, 'C'),
        ('toy2020task/exp1.csv', '0', 2, 'N')]))

    assert list(aligned['code']) == ['0', '1']
    assert list(aligned['choice_index']) == [0, 2]


def test_whole_number_codes_lose_their_float_tail(table):
    cache = table({'participant': ['0'] * 2, 'task': [0] * 2,
                   'choice': [1.0, 2.0]})
    aligned, _ = ct.align_sessions(cache, transcript([
        ('toy2020task/exp1.csv', '0', 0, 'C'),
        ('toy2020task/exp1.csv', '0', 1, 'N')]))

    assert list(aligned['code']) == ['1', '2']


def options(pairs):
    return pd.DataFrame(pairs, columns=['experiment', 'participant', 'option'])


def aligned_frame(rows):
    return pd.DataFrame(rows, columns=['experiment', 'participant',
                                       'choice_index', 'block', 'code',
                                       'label'])


def test_label_map_reads_the_dictionary_off_the_presses():
    frame = aligned_frame([('t/exp1.csv', '0', 0, '0', '0', 'C'),
                           ('t/exp1.csv', '0', 1, '0', '1', 'N')])
    result = ct.label_map(frame, options([('t/exp1.csv', '0', 'C'),
                                          ('t/exp1.csv', '0', 'N')]))

    assert dict(zip(result['label'], result['code'])) == {'C': '0', 'N': '1'}
    assert set(result['source']) == {'pressed'}


def test_elimination_names_the_key_the_participant_never_pressed():
    """One code free and one label spare is a pairing forced by counting."""

    frame = aligned_frame([('t/exp1.csv', '0', 0, '0', '0', 'C'),
                           ('t/exp1.csv', '0', 1, '0', '0', 'C'),
                           # a second participant shows the block allows '1'
                           ('t/exp1.csv', '1', 0, '0', '1', 'Q')])
    result = ct.label_map(frame, options([('t/exp1.csv', '0', 'C'),
                                          ('t/exp1.csv', '0', 'N'),
                                          ('t/exp1.csv', '1', 'Q')]))

    filled = result[result['source'] == 'elimination']
    assert list(filled['label']) == ['N']
    assert list(filled['code']) == ['1']


def test_elimination_stays_silent_when_more_than_one_label_is_spare():
    """A multi-block session offers keys belonging to its other blocks."""

    frame = aligned_frame([('t/exp1.csv', '0', 0, '0', '0', 'C'),
                           ('t/exp1.csv', '1', 0, '0', '1', 'Q')])
    result = ct.label_map(frame, options([('t/exp1.csv', '0', 'C'),
                                          ('t/exp1.csv', '0', 'N'),
                                          ('t/exp1.csv', '0', 'X'),
                                          ('t/exp1.csv', '1', 'Q')]))

    assert (result['source'] == 'elimination').sum() == 0


def test_a_block_whose_pairing_is_not_one_to_one_is_dropped():
    """Two labels on one code is what a misaligned session looks like."""

    frame = aligned_frame([('t/exp1.csv', '0', 0, '0', '0', 'C'),
                           ('t/exp1.csv', '0', 1, '1', '0', 'C'),
                           ('t/exp1.csv', '0', 2, '1', '1', 'C')])
    result = ct.label_map(frame, options([('t/exp1.csv', '0', 'C')]))

    assert list(result['block']) == ['0']
