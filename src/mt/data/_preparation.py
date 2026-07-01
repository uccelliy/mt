"""Prepare dataframe data through the public API."""

from __future__ import annotations

import pandas as pd

from mt.data._loading import load
from mt.data._prepared import PreparedData, PreparedSplit
from mt.data._requests import DataRequest, DataSource, FilterSpec, SplitSpec, TransformSpec
from mt.data.view import apply_transform, filter_dataframe, split_data_from_spec


def prepare_dataframe(
    source: DataSource | DataRequest,
    *,
    required_columns: tuple[str, ...] = (),
    split: SplitSpec | None = None,
    filters: tuple[FilterSpec, ...] = (),
    transforms: tuple[TransformSpec, ...] = (),
) -> PreparedData[pd.DataFrame]:
    """Load, validate, filter, transform, and split dataframe data."""

    request = _coerce_request(
        source,
        required_columns=required_columns,
        split=split,
        filters=filters,
        transforms=transforms,
    )
    df = load(request.source)
    _validate_required_columns(df, request.required_columns)

    for filter_spec in request.filters:
        df = filter_dataframe(df, filter_spec)

    for transform_spec in request.transforms:
        df = apply_transform(df, transform_spec)

    if request.split is None:
        return PreparedData(
            splits=[PreparedSplit(train=df.copy(), eval=df.copy(), name="full")]
        )

    return PreparedData(splits=list(split_data_from_spec(df, request.split)))


def prepare_data_view(request: DataRequest) -> PreparedData[pd.DataFrame]:
    """Backward-compatible alias for dataframe preparation."""

    return prepare_dataframe(request)


def _coerce_request(
    source: DataSource | DataRequest,
    *,
    required_columns: tuple[str, ...],
    split: SplitSpec | None,
    filters: tuple[FilterSpec, ...],
    transforms: tuple[TransformSpec, ...],
) -> DataRequest:
    if isinstance(source, DataRequest):
        return source
    return DataRequest(
        source=source,
        required_columns=required_columns,
        split=split,
        filters=filters,
        transforms=transforms,
    )


def _validate_required_columns(df, required_columns: tuple[str, ...]) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
