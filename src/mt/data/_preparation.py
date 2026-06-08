"""Prepare dataframe and tensor data through the public API."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from mt.data._contracts import DataContract, validate_dataframe
from mt.data._filtering import filter_dataframe
from mt.data._loading import load_dataframe
from mt.data._prepared import PreparedData, PreparedSplit
from mt.data._requests import DataRequest, DataSource, FilterSpec, SplitSpec, TransformSpec
from mt.data._splitting import split_data_from_spec
from mt.data._tensors import dataframe_to_tensors
from mt.data._transforms import apply_transform


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


def prepare_tensors(
    source: DataSource | DataRequest,
    *,
    contract: DataContract | None = None,
    required_columns: tuple[str, ...] = (),
    split: SplitSpec | None = None,
    filters: tuple[FilterSpec, ...] = (),
    transforms: tuple[TransformSpec, ...] = (),
) -> PreparedData[dict[str, object]]:
    """Prepare data and materialize each split as model-ready tensor dictionaries."""

    request = _coerce_request(
        source,
        contract=contract,
        required_columns=required_columns,
        split=split,
        filters=filters,
        transforms=transforms,
    )
    contract = contract or request.contract
    if contract is None:
        raise ValueError("prepare_tensors requires a DataContract.")

    dataframe_data = prepare_dataframe(replace(request, contract=contract))
    return PreparedData(
        splits=[
            PreparedSplit(
                train=dataframe_to_tensors(split.train, contract),
                eval=dataframe_to_tensors(split.eval, contract),
                name=split.name,
                metadata=split.metadata,
            )
            for split in dataframe_data.splits
        ],
        metadata={**dataframe_data.metadata, "contract": contract.name},
    )


def prepare_data_view(request: DataRequest) -> PreparedData[pd.DataFrame]:
    """Backward-compatible alias for dataframe preparation."""

    return prepare_dataframe(request)


def prepare_tensor_data_view(
    request: DataRequest,
    contract: DataContract | None = None,
) -> PreparedData[dict[str, object]]:
    """Backward-compatible alias for tensor preparation."""

    return prepare_tensors(request, contract=contract)


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
    columns: list[str] = []
    if request.contract is not None:
        columns.extend(request.contract.required_columns)
    columns.extend(request.required_columns)
    return list(dict.fromkeys(columns)) or None
