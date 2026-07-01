from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from types import MappingProxyType
import warnings

import numpy as np
import pandas as pd

from mt.data._field_registry import (
    CANONICAL_PATHS,
    FIELD_REGISTRY,
    get_field_spec,
)


@dataclass(frozen=True)
class MappingResolution:
    sources: Mapping[str, tuple[str, ...]]
    pattern_targets: tuple[str, ...]
    ignored_columns: tuple[str, ...]
    reused_columns: Mapping[str, tuple[str, ...]]

    def __post_init__(self):
        object.__setattr__(self, "sources", freeze_tuple_mapping(self.sources))
        object.__setattr__(self, "pattern_targets", tuple(self.pattern_targets))
        object.__setattr__(self, "ignored_columns", tuple(self.ignored_columns))
        object.__setattr__(
            self,
            "reused_columns",
            freeze_tuple_mapping(self.reused_columns),
        )


@dataclass(frozen=True)
class ColumnMapping:
    mappings: Mapping[str, str] = field(default_factory=dict)
    patterns: Mapping[str, str] = field(default_factory=dict)
    _compiled_patterns: Mapping[str, re.Pattern[str]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        mappings = validate_fixed_mappings(self.mappings)
        patterns, compiled_patterns = validate_patterns(self.patterns)
        overlapping_targets = tuple(
            path
            for path in CANONICAL_PATHS
            if path in mappings and path in patterns
        )
        if overlapping_targets:
            raise ValueError(
                "Multiple columns target one field; use only a pattern for "
                "multi-column mapping. Conflicting canonical paths: "
                f"{overlapping_targets}."
            )

        object.__setattr__(self, "mappings", MappingProxyType(mappings))
        object.__setattr__(self, "patterns", MappingProxyType(patterns))
        object.__setattr__(
            self,
            "_compiled_patterns",
            MappingProxyType(compiled_patterns),
        )

    def resolve(self, df: pd.DataFrame) -> MappingResolution:
        validate_dataframe(df)
        columns = tuple(df.columns)
        sources: dict[str, tuple[str, ...]] = {}
        pattern_targets: list[str] = []

        for target in CANONICAL_PATHS:
            spec = FIELD_REGISTRY[target]
            if target in self.patterns:
                sources[target] = resolve_pattern(
                    target,
                    self.patterns[target],
                    self._compiled_patterns[target],
                    columns,
                )
                pattern_targets.append(target)
                continue
            if target in self.mappings:
                source = self.mappings[target]
                if source not in df.columns:
                    raise ValueError(
                        f"Raw column {source!r} mapped to {target!r} was not "
                        "found."
                    )
                sources[target] = (source,)
                continue

            identity = spec.path if spec.is_coordinate else spec.key
            if identity in df.columns:
                sources[target] = (identity,)
            elif spec.required:
                raise ValueError(
                    f"Required canonical path {target!r} has no mapping and "
                    f"raw column {identity!r} was not found."
                )

        reused_columns = find_reused_columns(columns, sources)
        consumed = {
            source
            for target_sources in sources.values()
            for source in target_sources
        }
        ignored_columns = tuple(
            column for column in columns if column not in consumed
        )
        return MappingResolution(
            sources=sources,
            pattern_targets=tuple(pattern_targets),
            ignored_columns=ignored_columns,
            reused_columns=reused_columns,
        )

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        resolution = self.resolve(df)
        warn_reused_columns(resolution.reused_columns)
        result = pd.DataFrame(index=df.index.copy())

        for target, sources in resolution.sources.items():
            if target in resolution.pattern_targets:
                result[target] = stack_pattern_values(df, target, sources)
            else:
                result[target] = df[sources[0]].copy(deep=True)

        return result


def freeze_tuple_mapping(
    value: Mapping[str, tuple[str, ...]],
) -> Mapping[str, tuple[str, ...]]:
    return MappingProxyType({key: tuple(items) for key, items in value.items()})


def validate_fixed_mappings(value) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"mappings must be a mapping, got {type(value).__name__}."
        )

    result: dict[str, str] = {}
    for target, source in value.items():
        get_field_spec(target)
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"Raw source for canonical path {target!r} must be a "
                "non-empty string."
            )
        result[target] = source
    return result


def validate_patterns(
    value,
) -> tuple[dict[str, str], dict[str, re.Pattern[str]]]:
    if not isinstance(value, Mapping):
        raise TypeError(
            f"patterns must be a mapping, got {type(value).__name__}."
        )

    patterns: dict[str, str] = {}
    compiled_patterns: dict[str, re.Pattern[str]] = {}
    for target, expression in value.items():
        spec = get_field_spec(target)
        if not isinstance(expression, str) or not expression:
            raise ValueError(
                f"Pattern for canonical path {target!r} must be a non-empty "
                "string."
            )
        if not spec.allows_pattern:
            raise ValueError(
                f"Canonical path {target!r} does not allow pattern mapping."
            )
        try:
            compiled = re.compile(expression)
        except re.error as error:
            raise ValueError(
                f"Invalid pattern for canonical path {target!r}: "
                f"{expression!r}. {error}"
            ) from error
        patterns[target] = expression
        compiled_patterns[target] = compiled
    return patterns, compiled_patterns


def validate_dataframe(df) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"ColumnMapping requires pandas.DataFrame, got {type(df).__name__}."
        )

    duplicate_mask = df.columns.duplicated()
    duplicates = tuple(dict.fromkeys(df.columns[duplicate_mask].tolist()))
    if duplicates:
        raise ValueError(
            f"Raw DataFrame column names must be unique: {duplicates}."
        )


def resolve_pattern(
    target: str,
    expression: str,
    pattern: re.Pattern[str],
    columns: tuple[str, ...],
) -> tuple[str, ...]:
    matches: list[tuple[int, str, re.Match[str]]] = []
    for position, column in enumerate(columns):
        if not isinstance(column, str):
            continue
        match = pattern.fullmatch(column)
        if match is not None:
            matches.append((position, column, match))

    if not matches:
        raise ValueError(
            f"Pattern for canonical path {target!r} matched no raw columns: "
            f"{expression!r}."
        )

    if "index" in pattern.groupindex:
        indexed_matches: list[tuple[int, int, str]] = []
        for position, column, match in matches:
            captured = match.group("index")
            try:
                index = int(captured) if captured else None
            except ValueError:
                index = None
            if index is None:
                raise ValueError(
                    f"Pattern for canonical path {target!r} requires a "
                    f"numeric 'index' capture; column {column!r} captured "
                    f"{captured!r}."
                )
            indexed_matches.append((index, position, column))
        indexed_matches.sort(key=lambda item: (item[0], item[1]))
        return tuple(column for _, _, column in indexed_matches)

    return tuple(column for _, column, _ in matches)


def find_reused_columns(
    columns: tuple[str, ...],
    sources: Mapping[str, tuple[str, ...]],
) -> Mapping[str, tuple[str, ...]]:
    targets_by_source: defaultdict[str, list[str]] = defaultdict(list)
    for target, target_sources in sources.items():
        for source in target_sources:
            targets_by_source[source].append(target)

    return MappingProxyType(
        {
            column: tuple(targets_by_source[column])
            for column in columns
            if len(targets_by_source[column]) > 1
        }
    )


def warn_reused_columns(
    reused_columns: Mapping[str, tuple[str, ...]],
) -> None:
    for source, targets in reused_columns.items():
        warnings.warn(
            f"Raw column {source!r} is reused by canonical targets {targets}.",
            UserWarning,
            stacklevel=3,
        )


def stack_pattern_values(
    df: pd.DataFrame,
    target: str,
    sources: tuple[str, ...],
) -> pd.Series:
    if df.empty:
        return pd.Series(index=df.index.copy(), dtype=object)

    values: list[np.ndarray] = []
    for row_position, row in enumerate(
        df.loc[:, list(sources)].itertuples(index=False, name=None)
    ):
        try:
            stacked = np.stack(
                [np.asarray(value) for value in row],
                axis=-1,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Cannot stack pattern values for canonical path {target!r} "
                f"at row position {row_position}: {error}"
            ) from error
        values.append(stacked)

    return pd.Series(values, index=df.index.copy(), dtype=object)
