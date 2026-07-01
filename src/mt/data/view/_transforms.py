"""Data transforms that change what information is visible to a model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

def mask_columns(df, columns: Sequence[str], value: Any = np.nan):
    """Replace one or more columns with a fixed mask value."""

    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns to mask: {missing}")

    out = df.copy()
    for column in columns:
        out[column] = value
    return out

def replace_values(df, replacements: dict[str, Any]):
    """Replace selected columns with fixed values."""

    return mask_columns(
        df, tuple(replacements), value=np.nan
    ).assign(**replacements)

def clear_history(df, columns: Sequence[str] = ("history",)):
    """Remove history/context columns from a dataframe."""

    existing = [column for column in columns if column in df.columns]
    return mask_columns(df, existing, value="")
