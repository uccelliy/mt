"""Evaluation specifications.

Evaluation specs describe what to run; the data layer owns how requested data
is filtered, transformed, and split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mt.data.requests import DataRequest


@dataclass(frozen=True)
class MetricSpec:
    """Declarative metric configuration."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationSpec:
    """Declarative model evaluation setup."""

    name: str
    data: DataRequest
    metrics: tuple[MetricSpec, ...] = field(default_factory=tuple)
    model_name: str | None = None
    seed: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
