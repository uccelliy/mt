from __future__ import annotations

from pathlib import Path

from datasets import Dataset, DatasetDict, IterableDataset
import pandas as pd

DataSource = str | Path | pd.DataFrame | Dataset


def load(source: DataSource) -> pd.DataFrame:
    """Load a supported raw source into an independent DataFrame."""
    if isinstance(source, pd.DataFrame):
        frame = source.copy(deep=True)
    elif isinstance(source, Dataset):
        frame = source.to_pandas()
        if not isinstance(frame, pd.DataFrame):
            raise RuntimeError(
                "Dataset.to_pandas() returned batched output unexpectedly."
            )
    elif isinstance(source, DatasetDict):
        raise TypeError(
            "HuggingFace DatasetDict is not a single data source. Select one "
            "split and pass its Dataset."
        )
    elif isinstance(source, IterableDataset):
        raise TypeError(
            "HuggingFace IterableDataset is not supported. Materialize it as "
            "a Dataset or pandas.DataFrame."
        )
    elif isinstance(source, (str, Path)):
        frame = load_path(source)
    else:
        raise TypeError(
            f"Unsupported data source type: {type(source).__name__}. Expected "
            "str, Path, pandas.DataFrame, or datasets.Dataset."
        )

    return normalize_columns(frame)


def load_path(source: str | Path) -> pd.DataFrame:
    suffix = Path(str(source)).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    raise ValueError(
        f"Unsupported data source format for {str(source)!r}. Supported file "
        "extensions are '.csv' and '.parquet'."
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = tuple(normalize_column_name(column) for column in df.columns)
    column_index = pd.Index(columns)
    duplicates = tuple(
        dict.fromkeys(column_index[column_index.duplicated()].tolist())
    )
    if duplicates:
        raise ValueError(
            "Raw column names must be unique after string normalization; "
            f"duplicates: {duplicates}."
        )
    df.columns = columns
    return df


def normalize_column_name(column) -> str:
    if pd.api.types.is_scalar(column) and bool(pd.isna(column)):
        return "None"
    return str(column)
