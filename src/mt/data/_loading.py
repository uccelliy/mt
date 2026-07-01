from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DEFAULT_COLUMNS = ["participant", "task", "trial"]


def load_dataframe(path, columns: list[str] | None = None):
    columns = _with_default_columns(columns)

    if isinstance(path, pd.DataFrame):
        df = path.copy()
    else:
        path = str(path)
        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            if columns is not None:
                available = pd.read_csv(path, nrows=0).columns.tolist()
                _validate_columns(available, columns)
                return pd.read_csv(path, usecols=columns).loc[:, columns]
            df = pd.read_csv(path)
        elif suffix == ".parquet" or path.startswith("hf://"):
            df = pd.read_parquet(path, columns=columns)
        elif suffix == ".jsonl":
            df = pd.read_json(path, lines=True)
        elif suffix == ".json":
            df = pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    if columns is not None:
        _validate_columns(df.columns, columns)
        return df.loc[:, columns]

    return df


def load_hf_dataset(source: str, split: str, columns: list[str] | None, **kwargs):
    from datasets import load_dataset

    ds = load_dataset(source, split=split, **kwargs)
    if columns is not None:
        missing = [col for col in columns if col not in ds.column_names]
        if missing:
            raise KeyError(f"Missing columns: {missing}")

        ds = ds.select_columns(columns)

    return ds


def _with_default_columns(columns: Iterable[str] | None) -> list[str] | None:
    if columns is None:
        return None
    return list(dict.fromkeys([*DEFAULT_COLUMNS, *columns]))


def _validate_columns(available: Iterable[str], columns: Iterable[str]) -> None:
    available = set(available)
    missing = [column for column in columns if column not in available]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
