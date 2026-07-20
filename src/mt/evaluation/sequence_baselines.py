"""Count-based sequence baselines for marked-transcript choices."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Any

from mt.data._llm_supervision import find_target_spans

SMOOTHING = 0.5

@dataclass(frozen=True)
class BaselineTables:
    """Per-experiment count tables fitted on training sequences."""

    labels: tuple[str, ...]
    label_set: frozenset[str]
    unigram: Counter
    bigram: Counter
    prev_totals: Counter
    n_unigram: int
    n_repeats: int
    n_transitions: int

    def unigram_prob(self, label):
        k = len(self.labels)
        return ((self.unigram[label] + SMOOTHING)
                / (self.n_unigram + SMOOTHING * k))

    def bigram_prob(self, prev, label):
        k = len(self.labels)
        return ((self.bigram[(prev, label)] + SMOOTHING)
                / (self.prev_totals[prev] + SMOOTHING * k))

    def stick_theta(self):
        return ((self.n_repeats + SMOOTHING)
                / (self.n_transitions + 2 * SMOOTHING))

    def sticky_prob(self, prev, label):
        k = len(self.labels)
        if k == 1:
            return 1.0
        theta = self.stick_theta()
        if label == prev:
            return theta
        return (1 - theta) / (k - 1)

class TableBuilder:
    """Streaming accumulator for baseline count tables."""

    def __init__(self):
        self.unigram = Counter()
        self.bigram = Counter()
        self.prev_totals = Counter()
        self.n_repeats = 0
        self.n_transitions = 0

    def observe(self, sequence):
        self.unigram.update(sequence)
        for prev, current in zip(sequence, sequence[1:]):
            self.bigram[(prev, current)] += 1
            self.prev_totals[prev] += 1
            self.n_transitions += 1
            self.n_repeats += prev == current

    def build(self, vocabulary=()) -> BaselineTables:
        labels = tuple(sorted(set(vocabulary) | set(self.unigram)))
        return BaselineTables(labels, frozenset(labels), self.unigram,
                              self.bigram, self.prev_totals,
                              sum(self.unigram.values()), self.n_repeats,
                              self.n_transitions)

def extract_choices(text) -> list[str]:
    """Return the marked choice labels of a transcript, in order."""

    return [text[start:end] for start, end in find_target_spans(text)]

def fit_tables(sequences, vocabulary=()) -> BaselineTables:
    """Fit count tables from training sequences over a fixed vocabulary."""

    builder = TableBuilder()
    for sequence in sequences:
        builder.observe(sequence)
    return builder.build(vocabulary)

def score_sequence_online(choices) -> list[dict[str, Any]]:
    """Score one session with strictly-causal within-session baselines.

    Psych-101 randomizes response-key labels per participant, so counts
    fitted across participants are void in raw label space. Each trial is
    scored from the counts of earlier trials in the same session only; the
    session's label inventory is treated as known task structure.
    """

    labels = sorted(set(choices))
    k = len(labels)
    uniform_nll = math.log(k)
    unigram = Counter()
    bigram = Counter()
    prev_totals = Counter()
    n_repeats = 0
    n_transitions = 0
    prev = None
    records = []
    for index, label in enumerate(choices):
        base_prob = (unigram[label] + SMOOTHING) / (index + SMOOTHING * k)
        if prev is None or k == 1:
            sticky_prob = 1.0 if k == 1 else base_prob
            bigram_prob = 1.0 if k == 1 else base_prob
        else:
            theta = ((n_repeats + SMOOTHING)
                     / (n_transitions + 2 * SMOOTHING))
            sticky_prob = theta if label == prev else (1 - theta) / (k - 1)
            bigram_prob = ((bigram[(prev, label)] + SMOOTHING)
                           / (prev_totals[prev] + SMOOTHING * k))
        records.append({'choice_index': index,
                        'uniform': uniform_nll,
                        'base_rate': -math.log(base_prob),
                        'sticky': -math.log(sticky_prob),
                        'bigram': -math.log(bigram_prob)})
        unigram[label] += 1
        if prev is not None:
            bigram[(prev, label)] += 1
            prev_totals[prev] += 1
            n_transitions += 1
            n_repeats += prev == label
        prev = label
    return records

def score_sequence(tables, choices) -> list[dict[str, Any]]:
    """Return per-trial NLLs of each baseline for one choice sequence."""

    uniform_nll = math.log(len(tables.labels))
    records = []
    prev = None
    for index, label in enumerate(choices):
        if label not in tables.label_set:
            raise ValueError(f"Label {label!r} missing from vocabulary.")
        base_prob = tables.unigram_prob(label)
        if prev is None:
            sticky_prob = base_prob
            bigram_prob = base_prob
        else:
            sticky_prob = tables.sticky_prob(prev, label)
            bigram_prob = tables.bigram_prob(prev, label)
        records.append({'choice_index': index,
                        'uniform': uniform_nll,
                        'base_rate': -math.log(base_prob),
                        'sticky': -math.log(sticky_prob),
                        'bigram': -math.log(bigram_prob)})
        prev = label
    return records
