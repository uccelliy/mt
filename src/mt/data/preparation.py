"""Prepare data views from declarative requests."""

from __future__ import annotations

from mt.data.filtering import filter_dataframe
from mt.data.loading import load_dataframe
from mt.data.requests import DataRequest
from mt.data.splitting import split_data_from_spec
from mt.data.transforms import apply_transform
from mt.data.views import DataSplit, DataView


def prepare_data_view(request: DataRequest) -> DataView:
    """Load, filter, transform, and split data according to a request."""

    df = load_dataframe(request.source, list(request.required_columns) or None)

    for filter_spec in request.filters:
        df = filter_dataframe(df, filter_spec)

    for transform_spec in request.transforms:
        df = apply_transform(df, transform_spec)

    if request.split is None:
        return DataView(splits=[DataSplit(train=df.copy(), eval=df.copy(), name="full")])

    return DataView(splits=list(split_data_from_spec(df, request.split)))
