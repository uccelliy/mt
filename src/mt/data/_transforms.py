"""Data transforms that change what information is visible to a model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from mt.data._requests import TransformSpec


def mask_columns(df: pd.DataFrame, columns: Sequence[str], value: Any = np.nan) -> pd.DataFrame:
    """Replace one or more columns with a fixed mask value."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns to mask: {missing}")

    out = df.copy()
    for column in columns:
        out[column] = value
    return out


def replace_values(df: pd.DataFrame, replacements: dict[str, Any]) -> pd.DataFrame:
    """Replace selected columns with fixed values."""

    return mask_columns(df, tuple(replacements), value=np.nan).assign(**replacements)


def clear_history(df: pd.DataFrame, columns: Sequence[str] = ("history",)) -> pd.DataFrame:
    """Remove history/context columns from a dataframe."""

    existing = [column for column in columns if column in df.columns]
    return mask_columns(df, existing, value="")


def apply_transform(df: pd.DataFrame, spec: TransformSpec) -> pd.DataFrame:
    """Apply a declarative transform spec."""

    if spec.name == "mask_columns":
        return mask_columns(df, tuple(spec.params["columns"]), spec.params.get("value", np.nan))
    if spec.name == "replace_values":
        return replace_values(df, dict(spec.params["replacements"]))
    if spec.name == "clear_history":
        return clear_history(df, tuple(spec.params.get("columns", ("history",))))
    raise ValueError(f"Unknown transform {spec.name!r}")
