"""Prepare dataframe data through the public API."""

from __future__ import annotations

import pandas as pd

from mt.data._contracts import DataContract, validate_dataframe
from mt.data._loading import load_dataframe
from mt.data._prepared import PreparedData, PreparedSplit
from mt.data._requests import DataRequest, DataSource, FilterSpec, SplitSpec, TransformSpec
from mt.data.view import apply_transform, filter_dataframe, split_data_from_spec


def prepare_dataframe(
    source: DataSource | DataRequest,
    *,
    contract: DataContract | None = None,
    required_columns: tuple[str, ...] = (),
    split: SplitSpec | None = None,
    filters: tuple[FilterSpec, ...] = (),
    transforms: tuple[TransformSpec, ...] = (),
) -> PreparedData[pd.DataFrame]:
    """Load, validate, filter, transform, and split dataframe data."""

    request = _coerce_request(
        source,
        contract=contract,
        required_columns=required_columns,
        split=split,
        filters=filters,
        transforms=transforms,
    )
    df = load_dataframe(request.source, _columns_to_load(request))

    if request.contract is not None:
        validate_dataframe(df, request.contract)

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
    contract: DataContract | None,
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
        contract=contract,
        split=split,
        filters=filters,
        transforms=transforms,
    )


def _columns_to_load(request: DataRequest) -> list[str] | None:
    if request.contract is not None and request.contract.column_patterns:
        # Pattern-based requirements (for example x1, x2, ...) cannot be pushed
        # down as an exact column projection before the file schema is inspected.
        return None

    columns: list[str] = []
    if request.contract is not None:
        columns.extend(request.contract.required_columns)
    columns.extend(request.required_columns)
    return list(dict.fromkeys(columns)) or None
