from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np
import pandas as pd

from mt.data._collection import TrialCollection
from mt.data._field_registry import (
    CANONICAL_COORDINATES,
    CANONICAL_PATHS,
    CANONICAL_SLOTS,
    FIELD_REGISTRY,
)
from mt.data._loading import DataSource, load as load_source
from mt.data._mapping import ColumnMapping, MappingResolution

_TRIAL_IDENTITY = (
    "participant_id",
    "task_name",
    "session_id",
    "block_index",
    "trial_index",
)
_CANONICAL_SCALAR_TYPES = (str, int, float, bool, type(None))
_SLOT_PATHS = MappingProxyType(
    {
        slot: tuple(
            field.path for field in FIELD_REGISTRY.values()
            if field.slot == slot
        )
        for slot in CANONICAL_SLOTS
    }
)

@dataclass(frozen=True)
class AdaptationResult:
    collection: TrialCollection
    mapping: MappingResolution
    defaulted_paths: tuple[str, ...]
    input_rows_nums: int
    output_rows_nums: int

    def __post_init__(self):
        if not isinstance(self.collection, TrialCollection):
            raise TypeError(
                "AdaptationResult collection must be a TrialCollection."
            )
        if not isinstance(self.mapping, MappingResolution):
            raise TypeError(
                "AdaptationResult mapping must be a MappingResolution."
            )

        defaulted_paths = normalize_defaulted_paths(self.defaulted_paths)
        object.__setattr__(self, "defaulted_paths", defaulted_paths)
        validate_row_count("input_rows_nums", self.input_rows_nums)
        validate_row_count("output_rows", self.output_rows_nums)
        if self.output_rows_nums > self.input_rows_nums:
            raise ValueError(
                "AdaptationResult output_rows cannot exceed input_rows_nums."
            )
        if self.output_rows_nums != self.collection.n_trials:
            raise ValueError(
                "AdaptationResult output_rows must equal collection.n_trials."
            )

    def report(self):
        return format_adaptation_report(self)

class DataAdapter:
    def __init__(self, mapping: ColumnMapping | None = None):
        if mapping is None:
            mapping = ColumnMapping()
        if not isinstance(mapping, ColumnMapping):
            raise TypeError(
                "DataAdapter mapping must be a ColumnMapping or None, got "
                f"{type(mapping).__name__}."
            )
        self.mapping = mapping

    def adapt(self, source: DataSource, *, filters=None) -> AdaptationResult:
        raw = load_source(source)
        input_rows_nums = len(raw)
        resolution = self.mapping.resolve(raw)
        mapped = self.mapping.apply(raw)
        defaulted_paths = find_defaulted_paths(mapped)
        canonical = apply_defaults(mapped)
        normalized = normalize_missing_values(canonical)
        if filters is not None:
            normalized = filter_rows(normalized, filters)
        validated = validate_dataframe(normalized)
        collection = assemble_trials(validated)
        return AdaptationResult(
            collection=collection,
            mapping=resolution,
            defaulted_paths=defaulted_paths,
            input_rows_nums=input_rows_nums,
            output_rows_nums=collection.n_trials,
        )

def apply_defaults(df: pd.DataFrame) -> pd.DataFrame:
    validate_canonical_dataframe(df, "apply_defaults")
    result = df.copy(deep=True)
    for path in CANONICAL_PATHS:
        field = FIELD_REGISTRY[path]
        if path not in result.columns and field.has_default:
            values = [copy_default(field.default) for _ in range(len(result))]
            dtype = object if field.default is None or isinstance(
                field.default, np.ndarray
            ) else type(field.default)
            result[path] = pd.Series(values, index=result.index, dtype=dtype)

    columns = [path for path in CANONICAL_PATHS if path in result.columns]
    return result.loc[:, columns].copy(deep=True)

def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    validate_canonical_dataframe(df, "normalize_missing_values")
    result = df.copy(deep=True)
    for path in df.columns:
        values = df[path].tolist()
        if not any(is_missing_scalar(value) for value in values):
            continue
        normalized = [
            None if is_missing_scalar(value) else value for value in values
        ]
        result[path] = pd.Series(
            normalized, index=result.index, dtype=object
        )
    return result

def filter_rows(df: pd.DataFrame,
                criteria: Mapping[str, object]) -> pd.DataFrame:
    validate_canonical_dataframe(df, "filter_rows")
    if not isinstance(criteria, Mapping):
        raise TypeError(
            f"Filter criteria must be a mapping, got "
            f"{type(criteria).__name__}."
        )

    for path in criteria:
        if path not in CANONICAL_COORDINATES:
            raise ValueError(
                f"Filter path must be a canonical coordinate: {path!r}."
            )
        if path not in df.columns:
            raise ValueError(f"Filter coordinate is missing: {path!r}.")

    mask = pd.Series(True, index=df.index, dtype=bool)
    for path in CANONICAL_COORDINATES:
        if path in criteria:
            values = normalize_filter_values(path, criteria[path])
            mask &= df[path].isin(values)
    return df.loc[mask].copy(deep=True)

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    validate_canonical_dataframe(df, "validate_dataframe")
    missing = tuple(path for path in CANONICAL_PATHS if path not in df.columns)
    if missing:
        raise ValueError(f"Canonical DataFrame is missing paths: {missing}.")

    for path in CANONICAL_COORDINATES:
        for position, value in enumerate(df[path].tolist()):
            validate_coordinate_value(path, position, value)

    for path in CANONICAL_PATHS:
        field = FIELD_REGISTRY[path]
        if field.is_coordinate:
            continue
        for position, value in enumerate(df[path].tolist()):
            validate_content_value(path, position, value)

    duplicate_mask = df.loc[:, list(_TRIAL_IDENTITY)].duplicated(keep=False)
    if duplicate_mask.any():
        positions = tuple(np.flatnonzero(duplicate_mask.to_numpy()).tolist())
        raise ValueError(
            "Duplicate logical trial identity at row positions "
            f"{positions}. Multi-row trials are not supported by the "
            "one-row assembly strategy."
        )
    return df.copy(deep=True)

def assemble_trials(df: pd.DataFrame) -> TrialCollection:
    validate_canonical_dataframe(df, "assemble_trials")
    missing = tuple(path for path in CANONICAL_PATHS if path not in df.columns)
    if missing:
        raise ValueError(f"Cannot assemble missing canonical paths: {missing}.")

    coordinates = {
        path: df[path].to_numpy(copy=True)
        for path in CANONICAL_COORDINATES
    }
    slots: dict[str, list[dict[str, object]]] = {
        slot: [] for slot in CANONICAL_SLOTS
    }
    for position in range(len(df)):
        for slot in CANONICAL_SLOTS:
            row = {
                FIELD_REGISTRY[path].key: copy_assembled_value(
                    df[path].iloc[position]
                )
                for path in _SLOT_PATHS[slot]
            }
            slots[slot].append(row)
    return build_collection(coordinates, slots)

def build_collection(coordinates, slots) -> TrialCollection:
    validate_group_keys("coordinates", coordinates, CANONICAL_COORDINATES)
    validate_group_keys("slots", slots, CANONICAL_SLOTS)
    values = {name: coordinates[name] for name in CANONICAL_COORDINATES}
    values.update({name: slots[name] for name in CANONICAL_SLOTS})
    return TrialCollection(**values)

def validate_canonical_dataframe(df, operation: str):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{operation} requires pandas.DataFrame, got "
            f"{type(df).__name__}."
        )

    duplicate_mask = df.columns.duplicated()
    duplicates = tuple(dict.fromkeys(df.columns[duplicate_mask].tolist()))
    if duplicates:
        raise ValueError(
            f"Canonical DataFrame columns must be unique: {duplicates}."
        )

    unknown = tuple(path for path in df.columns if path not in FIELD_REGISTRY)
    if unknown:
        raise ValueError(f"Unknown canonical DataFrame paths: {unknown}.")

def copy_default(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    return value

def is_missing_scalar(value):
    if isinstance(value, np.ndarray) or not pd.api.types.is_scalar(value):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(missing, (bool, np.bool_)) and bool(missing)

def normalize_numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    return value

def normalize_filter_values(path: str, value) -> tuple[object, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise ValueError(
                f"Filter values for {path!r} must be one-dimensional."
            )
        values = value.tolist()
    elif isinstance(value, pd.Series):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    elif isinstance(value, (set, Mapping)) or not is_filter_scalar(value):
        raise ValueError(
            f"Filter values for {path!r} must be a scalar, list, tuple, "
            "one-dimensional numpy.ndarray, or pandas.Series."
        )
    else:
        values = [value]

    return tuple(normalize_filter_scalar(path, item) for item in values)

def is_filter_scalar(value):
    if is_missing_scalar(value):
        return True
    value = normalize_numpy_scalar(value)
    return isinstance(value, _CANONICAL_SCALAR_TYPES)

def normalize_filter_scalar(path: str, value):
    if is_missing_scalar(value):
        return None
    value = normalize_numpy_scalar(value)
    if not isinstance(value, _CANONICAL_SCALAR_TYPES):
        raise ValueError(
            f"Filter value for {path!r} must be a canonical scalar, got "
            f"{type(value).__name__}."
        )
    return value

def validate_coordinate_value(path: str, position: int, value):
    if is_missing_scalar(value):
        raise ValueError(
            f"Coordinate {path!r} is missing at row position {position}."
        )
    value = normalize_numpy_scalar(value)
    if not isinstance(value, _CANONICAL_SCALAR_TYPES):
        raise ValueError(
            f"Coordinate {path!r} at row position {position} must be a "
            f"canonical scalar, got {type(value).__name__}."
        )

def validate_content_value(path: str, position: int, value):
    if isinstance(value, np.ndarray):
        return
    if value is not None and is_missing_scalar(value):
        raise ValueError(
            f"Canonical content {path!r} at row position {position} has an "
            "unnormalized missing scalar. Run normalize_missing_values()."
        )
    value = normalize_numpy_scalar(value)
    if not isinstance(value, _CANONICAL_SCALAR_TYPES):
        raise ValueError(
            f"Canonical content {path!r} at row position {position} has "
            f"unsupported value type {type(value).__name__}."
        )

def copy_assembled_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    return normalize_numpy_scalar(value)

def validate_group_keys(name: str, value, expected: tuple[str, ...]):
    if not isinstance(value, Mapping):
        raise TypeError(
            f"Collection {name} must be a mapping, got "
            f"{type(value).__name__}."
        )
    missing = tuple(key for key in expected if key not in value)
    unknown = tuple(key for key in value if key not in expected)
    if missing or unknown:
        raise ValueError(
            f"Collection {name} keys are invalid; missing={missing}, "
            f"unknown={unknown}."
        )

def find_defaulted_paths(df: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        path
        for path in CANONICAL_PATHS
        if path not in df.columns and FIELD_REGISTRY[path].has_default
    )

def normalize_defaulted_paths(paths) -> tuple[str, ...]:
    try:
        paths = tuple(paths)
    except TypeError as error:
        raise TypeError("defaulted_paths must be an iterable of paths.") \
            from error

    if len(paths) != len(set(paths)):
        raise ValueError("defaulted_paths must not contain duplicates.")
    for path in paths:
        if path not in FIELD_REGISTRY or not FIELD_REGISTRY[path].has_default:
            raise ValueError(f"Invalid defaulted canonical path: {path!r}.")
    path_set = set(paths)
    return tuple(path for path in CANONICAL_PATHS if path in path_set)

def validate_row_count(name: str, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")

def format_adaptation_report(result: AdaptationResult):
    lines = [
        "Adaptation succeeded.",
        f"Rows: {result.input_rows_nums} input -> {result.output_rows_nums} output",
        "Canonical fields:",
    ]
    defaulted = set(result.defaulted_paths)
    for path in CANONICAL_PATHS:
        if path in result.mapping.sources:
            sources = ", ".join(result.mapping.sources[path])
            suffix = " (pattern)" if path in result.mapping.pattern_targets \
                else ""
            lines.append(f"  {path} <- {sources}{suffix}")
        elif path in defaulted:
            value = FIELD_REGISTRY[path].default
            lines.append(f"  {path} <- {value!r} (default)")

    ignored = ", ".join(result.mapping.ignored_columns) or "(none)"
    lines.extend(["Ignored raw columns:", f"  {ignored}"])
    lines.append("Reused raw columns:")
    if result.mapping.reused_columns:
        for source, targets in result.mapping.reused_columns.items():
            lines.append(f"  {source} -> {', '.join(targets)}")
    else:
        lines.append("  (none)")
    lines.append("Scalar missing value: None")
    return "\n".join(lines)
