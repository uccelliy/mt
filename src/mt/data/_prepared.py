"""Prepared data containers returned by the public data API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass
class PreparedSplit(Generic[T]):
    """One train/evaluation split produced by the data layer."""

    train: T
    eval: T
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedData(Generic[T]):
    """Prepared data, always represented as one or more named splits."""

    splits: list[PreparedSplit[T]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full(self) -> T:
        """Return the data payload for a single full split."""

        if len(self.splits) != 1 or self.splits[0].name != "full":
            raise ValueError("PreparedData.full is only available for a single 'full' split.")
        return self.splits[0].train
