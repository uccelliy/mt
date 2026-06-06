"""Filtering helpers for behavioral task dataframes."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd

from mt.data.requests import FilterSpec


def filter_dataframe(df: pd.DataFrame, spec: FilterSpec) -> pd.DataFrame:
    """Apply a declarative include/exclude filter."""

    if spec.column not in df.columns:
        raise KeyError(f"Missing filter column: {spec.column}")

    mask = df[spec.column].isin(spec.values)
    if spec.mode == "include":
        return df.loc[mask].copy()
    if spec.mode == "exclude":
        return df.loc[~mask].copy()
    raise ValueError(f"Unknown filter mode {spec.mode!r}. Use 'include' or 'exclude'.")


def filter_by_values(
    df: pd.DataFrame,
    column: str,
    values: Iterable[Any],
    *,
    mode: str = "include",
) -> pd.DataFrame:
    """Convenience wrapper around :class:`FilterSpec`."""

    return filter_dataframe(df, FilterSpec(column=column, values=tuple(values), mode=mode))


def drop_participants(df: pd.DataFrame, participants: Iterable[Any]) -> pd.DataFrame:
    """Remove participants from a dataframe."""

    return filter_by_values(df, "participant", participants, mode="exclude")
