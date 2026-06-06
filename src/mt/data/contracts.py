"""Tabular data contracts for behavioral and cognitive tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


DEFAULT_INDEX_COLUMNS = ("participant", "task", "trial")


@dataclass(frozen=True)
class ColumnSpec:
    """Expected column in a behavioral task table."""

    name: str
    dtype: str | None = None
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class DataContract:
    """Schema-level contract for a tabular behavioral dataset."""

    name: str
    columns: tuple[ColumnSpec, ...] = field(default_factory=tuple)
    index_columns: tuple[str, ...] = DEFAULT_INDEX_COLUMNS
    description: str = ""

    @property
    def required_columns(self) -> tuple[str, ...]:
        declared = tuple(column.name for column in self.columns if column.required)
        return tuple(dict.fromkeys((*self.index_columns, *declared)))


def validate_dataframe(df: pd.DataFrame, contract: DataContract) -> pd.DataFrame:
    """Validate that a dataframe satisfies a contract and return it unchanged."""

    missing = [column for column in contract.required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"{contract.name} is missing required columns: {missing}")
    return df


def make_contract(
    name: str,
    required_columns: Iterable[str],
    *,
    index_columns: Iterable[str] = DEFAULT_INDEX_COLUMNS,
    description: str = "",
) -> DataContract:
    """Build a simple contract from required column names."""

    return DataContract(
        name=name,
        columns=tuple(ColumnSpec(name=column) for column in required_columns),
        index_columns=tuple(index_columns),
        description=description,
    )
