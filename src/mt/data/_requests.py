"""Declarative requests used by evaluation code to ask for data views."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from mt.data._contracts import DataContract


DataSource = str | Path | pd.DataFrame


@dataclass(frozen=True)
class FilterSpec:
    """Declarative row filter for selecting or excluding data."""

    column: str
    values: tuple[Any, ...]
    mode: str = "include"


@dataclass(frozen=True)
class TransformSpec:
    """Declarative data transform such as masking or history truncation."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitSpec:
    """Declarative split request owned by the data layer."""

    strategy: str
    column: str
    num_splits: int = 10
    shuffle: bool = False
    random_state: int | None = None


@dataclass(frozen=True)
class DataRequest:
    """Everything evaluation needs to ask the data layer for a prepared view."""

    source: DataSource
    required_columns: tuple[str, ...] = field(default_factory=tuple)
    contract: DataContract | None = None
    split: SplitSpec | None = None
    filters: tuple[FilterSpec, ...] = field(default_factory=tuple)
    transforms: tuple[TransformSpec, ...] = field(default_factory=tuple)


def held_out_participant_split(
    *,
    num_splits: int = 10,
    shuffle: bool = False,
    random_state: int | None = None,
) -> SplitSpec:
    return SplitSpec(
        strategy="held_out_participant",
        column="participant",
        num_splits=num_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def held_out_trial_split(
    *,
    num_splits: int = 10,
    shuffle: bool = False,
    random_state: int | None = None,
) -> SplitSpec:
    return SplitSpec(
        strategy="held_out_trial",
        column="trial",
        num_splits=num_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


def held_out_task_split(
    *,
    num_splits: int = 10,
    shuffle: bool = False,
    random_state: int | None = None,
) -> SplitSpec:
    return SplitSpec(
        strategy="held_out_task",
        column="task",
        num_splits=num_splits,
        shuffle=shuffle,
        random_state=random_state,
    )
