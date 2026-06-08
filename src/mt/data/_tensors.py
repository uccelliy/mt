"""Materialize contracted dataframes as model-ready tensor dictionaries."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from mt.data._contracts import DataContract, TensorSpec, validate_dataframe


def dataframe_to_tensors(
    df: pd.DataFrame,
    contract: DataContract,
    *,
    tensor_specs: Iterable[TensorSpec] | None = None,
) -> dict[str, object]:
    """Map dataframe columns declared by a contract onto tensor keys."""

    validate_dataframe(df, contract)
    specs = contract.tensors if tensor_specs is None else tuple(tensor_specs)
    tensors = {}

    for spec in specs:
        missing = [column for column in spec.columns if column not in df.columns]
        if missing:
            if spec.required:
                raise KeyError(f"{contract.name} tensor {spec.key!r} missing columns: {missing}")
            continue

        arrays = [_wide_array(df, column, contract.index_columns) for column in spec.columns]
        data = arrays[0] if len(arrays) == 1 else np.stack(arrays, axis=-1)
        tensors[spec.key] = _to_torch_tensor(data, spec.dtype)

    return tensors


def dataframe_pair_to_tensors(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    contract: DataContract,
) -> tuple[dict[str, object], dict[str, object]]:
    """Convert matching train/eval dataframes to tensor dictionaries."""

    return dataframe_to_tensors(train_df, contract), dataframe_to_tensors(eval_df, contract)


def _wide_array(df: pd.DataFrame, value_column: str, index_columns: tuple[str, ...]) -> np.ndarray:
    if not (
        pd.api.types.is_numeric_dtype(df[value_column])
        or pd.api.types.is_bool_dtype(df[value_column])
    ):
        raise TypeError(
            f"Tensor column {value_column!r} must be numeric or bool; "
            f"got {df[value_column].dtype}"
        )

    arr = df.loc[:, [*index_columns, value_column]].to_numpy()
    dims = [np.unique(arr[:, i], return_inverse=True) for i in range(len(index_columns))]
    wide = np.full([len(dim[0]) for dim in dims], np.nan)
    index = tuple(dim[1] for dim in dims)
    wide[index] = arr[:, -1]
    return wide.reshape(-1, wide.shape[-1])


def _to_torch_tensor(array: np.ndarray, dtype: str | None):
    import torch

    tensor = torch.from_numpy(array)
    if dtype is None:
        return tensor

    dtype_map = {
        "float": torch.float32,
        "float32": torch.float32,
        "float64": torch.float64,
        "long": torch.long,
        "int64": torch.int64,
        "bool": torch.bool,
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unknown tensor dtype {dtype!r}")
    return tensor.to(dtype_map[dtype])
