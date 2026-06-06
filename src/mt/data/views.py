"""Prepared data views returned by the data layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DataSplit:
    """One train/evaluation split produced by the data layer."""

    train: pd.DataFrame
    eval: pd.DataFrame
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataView:
    """A prepared dataset view, optionally containing multiple splits."""

    splits: list[DataSplit]
    metadata: dict[str, Any] = field(default_factory=dict)
