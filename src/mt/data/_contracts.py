"""Tabular data contracts for behavioral and cognitive tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable, Mapping, Sequence

import pandas as pd


DEFAULT_INDEX_COLUMNS = ("participant", "task", "trial")
DEFAULT_COLUMN_GROUPS = {
    "index": DEFAULT_INDEX_COLUMNS,
    "cognitive_model": ("choice", "reward", "ground_truth", "current_state", "forced"),
}


@dataclass(frozen=True)## can not changeable dataclass
class ColumnSpec:
    """Expected column in a behavioral task table."""

    name: str
    dtype: str | None = None
    required: bool = True
    group: str | None = None
    description: str = ""


@dataclass(frozen=True)
class ColumnPatternSpec:
    """Required family of columns selected by a regular-expression pattern."""

    name: str
    pattern: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class DataContract:
    """Schema-level contract for a tabular behavioral dataset."""

    name: str
    columns: tuple[ColumnSpec, ...] = field(default_factory=tuple)
    column_patterns: tuple[ColumnPatternSpec, ...] = field(default_factory=tuple)
    index_columns: tuple[str, ...] = DEFAULT_INDEX_COLUMNS
    column_groups: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    description: str = ""

    @property
    def required_columns(self) -> tuple[str, ...]:
        declared = tuple(column.name for column in self.columns if column.required)
        return tuple(dict.fromkeys((*self.index_columns, *declared))) ## fromkey Deduplicate while preserving order

    def columns_for_groups(self, groups: Sequence[str]) -> tuple[str, ...]:
        """Return columns assigned to one or more semantic groups."""

        columns: list[str] = []
        for group in groups:
            columns.extend(self.column_groups.get(group, ()))
        return tuple(dict.fromkeys(columns))


def missing_required_columns(columns: Iterable[str], contract: DataContract) -> list[str]:
    """Return required contract columns that are absent from a column collection."""

    available = set(columns)
    return [column for column in contract.required_columns if column not in available]


def validate_dataframe(df: pd.DataFrame, contract: DataContract) -> pd.DataFrame:
    """Validate that a dataframe satisfies a contract and return it unchanged."""

    missing = missing_required_columns(df.columns, contract)
    if missing:
        raise KeyError(f"{contract.name} is missing required columns: {missing}")
    missing_patterns = [
        pattern.name
        for pattern in contract.column_patterns
        if pattern.required
        and not any(re.fullmatch(pattern.pattern, str(column)) for column in df.columns)
    ]
    if missing_patterns:
        raise KeyError(
            f"{contract.name} is missing required column patterns: {missing_patterns}"
        )
    return df


def make_contract(
    name: str,
    required_columns: Iterable[str],
    *, ##parameters after `*` must be passed by keyword.
    optional_columns: Iterable[str] = (),
    index_columns: Iterable[str] = DEFAULT_INDEX_COLUMNS,
    column_groups: Mapping[str, Sequence[str]] | None = None,
    column_patterns: Iterable[ColumnPatternSpec] = (),
    description: str = "",
) -> DataContract:
    """Build a simple contract from required column names."""

    required_specs = tuple(ColumnSpec(name=column) for column in required_columns)
    optional_specs = tuple(
        ColumnSpec(name=column, required=False) for column in optional_columns
    )
    return DataContract(
        name=name,
        columns=(*required_specs, *optional_specs),
        column_patterns=tuple(column_patterns),
        index_columns=tuple(index_columns),
        column_groups={
            group: tuple(columns)
            for group, columns in (column_groups or {}).items()
        },
        description=description,
    )


def cognitive_model_contract(
    name: str,
    required_columns: Iterable[str],
    *,
    optional_columns: Iterable[str] = (),
    index_columns: Iterable[str] = DEFAULT_INDEX_COLUMNS,
) -> DataContract:
    """Build a contract for data used by cognitive models."""

    required_columns = tuple(required_columns)
    optional_columns = tuple(optional_columns)
    return make_contract(
        name,
        required_columns,
        optional_columns=optional_columns,
        index_columns=index_columns,
        column_groups=DEFAULT_COLUMN_GROUPS,
    )


def standard_behavior_contract() -> DataContract:
    """Default standardized behavioral table contract."""

    return cognitive_model_contract(
        "standard_behavior",
        required_columns=("choice", "reward"),
        optional_columns=("ground_truth", "current_state", "forced"),
    )
