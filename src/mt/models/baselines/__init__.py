"""Count-based baseline models for behavioral choice sequences.

Two spaces, because Psych-101 randomizes response keys per participant:

- :mod:`mt.models.baselines.sequence` — transcript label space, scored
  strictly within a session (``uniform``, ``base_rate``, ``sticky``, ``bigram``)
- :mod:`mt.models.baselines.population` — canonical choice space, pooled
  across participants (``pop_base_rate``, ``pop_bigram``, ``canonical_uniform``)
"""

from mt.models.baselines.population import (
    POPULATION_BASELINES,
    fit_counts,
    score_codes,
    summarize,
)
from mt.models.baselines.sequence import (
    SEQUENCE_BASELINES,
    SMOOTHING,
    BaselineTables,
    TableBuilder,
    extract_choices,
    fit_tables,
    score_sequence,
    score_sequence_online,
)


__all__ = [
    "POPULATION_BASELINES",
    "SEQUENCE_BASELINES",
    "SMOOTHING",
    "BaselineTables",
    "TableBuilder",
    "extract_choices",
    "fit_counts",
    "fit_tables",
    "score_codes",
    "score_sequence",
    "score_sequence_online",
    "summarize",
]
