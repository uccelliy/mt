from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from mt.data._field_registry import (
    CANONICAL_COORDINATES,
    CANONICAL_PATHS,
    CANONICAL_SLOTS,
    FIELD_REGISTRY,
    CanonicalValue,
)

_SLOT_KEYS = MappingProxyType(
    {
        slot: tuple(
            field.key for field in FIELD_REGISTRY.values()
            if field.slot == slot
        )
        for slot in CANONICAL_SLOTS
    }
)
_CANONICAL_SCALAR_TYPES = (str, int, float, bool, type(None))

@dataclass(eq=False)
class TrialCollection:
    participant_id: np.ndarray
    session_id: np.ndarray
    block_index: np.ndarray
    trial_index: np.ndarray
    task_name: np.ndarray
    condition: np.ndarray
    task: list[dict[str, CanonicalValue]]
    context: list[dict[str, CanonicalValue]]
    stimulus: list[dict[str, CanonicalValue]]
    response: list[dict[str, CanonicalValue]]
    outcome: list[dict[str, CanonicalValue]]

    def __post_init__(self):
        for name in CANONICAL_COORDINATES:
            value = validate_coordinate(name, getattr(self, name))
            setattr(self, name, value)

        n_trials = len(self.participant_id)
        for name in CANONICAL_COORDINATES[1:]:
            actual = len(getattr(self, name))
            if actual != n_trials:
                raise ValueError(
                    f"Coordinate {name!r} has length {actual}; expected "
                    f"{n_trials}."
                )

        for name in CANONICAL_SLOTS:
            value = validate_slot(name, getattr(self, name), n_trials)
            setattr(self, name, value)

    @property
    def n_trials(self):
        return len(self.participant_id)

    def __len__(self):
        return self.n_trials

    def copy(self) -> TrialCollection:
        values = {
            name: getattr(self, name) for name in CANONICAL_COORDINATES
        }
        values.update({name: getattr(self, name) for name in CANONICAL_SLOTS})
        return TrialCollection(**values)

    def select(self, selector) -> TrialCollection:
        positions = normalize_selector(selector, self.n_trials)
        values = {
            name: getattr(self, name)[positions]
            for name in CANONICAL_COORDINATES
        }
        values.update(
            {
                name: [getattr(self, name)[position]
                       for position in positions]
                for name in CANONICAL_SLOTS
            }
        )
        return TrialCollection(**values)

    def to_dataframe(self) -> pd.DataFrame:
        columns: dict[str, object] = {}
        for path in CANONICAL_PATHS:
            field = FIELD_REGISTRY[path]
            if field.is_coordinate:
                columns[path] = getattr(self, path).copy()
                continue
            columns[path] = [
                copy_canonical_value(row[field.key], path, position)
                for position, row in enumerate(getattr(self, field.slot))
            ]
        return pd.DataFrame(columns)

def validate_coordinate(name: str, value) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise TypeError(
            f"Coordinate {name!r} must be a numpy.ndarray, got "
            f"{type(value).__name__}."
        )
    if value.ndim != 1:
        raise ValueError(
            f"Coordinate {name!r} must be one-dimensional, got shape "
            f"{value.shape}."
        )

    result = value.copy()
    for position, item in enumerate(result):
        scalar = normalize_numpy_scalar(item)
        if not isinstance(scalar, _CANONICAL_SCALAR_TYPES):
            raise ValueError(
                f"Coordinate {name!r} at trial position {position} must be "
                f"a canonical scalar, got {type(item).__name__}."
            )
    return result

def validate_slot(name: str, value, n_trials: int):
    if not isinstance(value, list):
        raise TypeError(
            f"Slot {name!r} must be a list, got {type(value).__name__}."
        )
    if len(value) != n_trials:
        raise ValueError(
            f"Slot {name!r} has length {len(value)}; expected {n_trials}."
        )

    expected_keys = _SLOT_KEYS[name]
    expected_set = set(expected_keys)
    result: list[dict[str, CanonicalValue]] = []
    for position, row in enumerate(value):
        if not isinstance(row, dict):
            raise TypeError(
                f"Slot {name!r} at trial position {position} must be a "
                f"dictionary, got {type(row).__name__}."
            )

        missing = tuple(key for key in expected_keys if key not in row)
        unknown = tuple(key for key in row if key not in expected_set)
        if missing or unknown:
            raise ValueError(
                f"Slot {name!r} at trial position {position} has invalid "
                f"keys; missing={missing}, unknown={unknown}."
            )

        copied = {
            key: copy_canonical_value(
                row[key], f"{name}.{key}", position
            )
            for key in expected_keys
        }
        result.append(copied)
    return result

def copy_canonical_value(value, path: str, position: int):
    if isinstance(value, np.ndarray):
        return value.copy()

    scalar = normalize_numpy_scalar(value)
    if isinstance(scalar, _CANONICAL_SCALAR_TYPES):
        return scalar
    raise ValueError(
        f"Canonical path {path!r} at trial position {position} has "
        f"unsupported value type {type(value).__name__}."
    )

def normalize_numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value

def normalize_selector(selector, n_trials: int) -> np.ndarray:
    if isinstance(selector, slice):
        try:
            return np.arange(n_trials)[selector]
        except ValueError as error:
            raise ValueError(f"Invalid trial slice: {error}") from error

    values = np.asarray(selector)
    if values.ndim != 1:
        raise ValueError(
            "Trial selector must be a one-dimensional integer sequence or "
            "boolean mask."
        )
    if values.size == 0:
        return np.asarray([], dtype=int)

    items = (
        values.tolist() if isinstance(selector, np.ndarray)
        else list(selector)
    )
    if all(isinstance(item, (bool, np.bool_)) for item in items):
        if len(items) != n_trials:
            raise ValueError(
                f"Boolean trial selector has length {len(items)}; expected "
                f"{n_trials}."
            )
        return np.flatnonzero(np.asarray(items, dtype=bool))

    integer_types = (int, np.integer)
    if not all(
        isinstance(item, integer_types)
        and not isinstance(item, (bool, np.bool_))
        for item in items
    ):
        raise ValueError(
            "Trial selector values must be all integers or all booleans."
        )

    indices = np.asarray(items, dtype=int)
    try:
        return np.arange(n_trials)[indices]
    except IndexError as error:
        raise ValueError(f"Trial selector index is out of range: {error}") \
            from error
