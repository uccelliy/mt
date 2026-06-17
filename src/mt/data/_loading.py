import logging
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

import pandas as pd

from mt.data._contracts import DataContract, validate_dataframe


LOGGER = logging.getLogger(__name__)


DEFAULT_COLUMNS = ["participant", "task", "trial"]
DataSourceT = TypeVar("DataSourceT", str, Path, pd.DataFrame)


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


def iter_contract_dataframes(
    sources: Iterable[DataSourceT],
    contract: DataContract,
    *,
    columns: Iterable[str] = (),
    logger: logging.Logger | None = None,
) -> Iterator[tuple[DataSourceT, pd.DataFrame]]:
    """Yield dataframes that satisfy a contract; log and skip invalid sources."""

    log = logger or LOGGER
    requested_columns = list(dict.fromkeys((*contract.required_columns, *columns)))
    for source in sources:
        try:
            df = load_dataframe(source, requested_columns)
            validate_dataframe(df, contract)
        except (KeyError, TypeError, ValueError) as exc:
            log.error("Skipping %s: %s", _source_name(source), exc)
            continue
        yield source, df


def iter_data_directory(
    root: str | Path,
    contract: DataContract,
    *,
    pattern: str = "*.csv",
    columns: Iterable[str] = (),
    logger: logging.Logger | None = None,
) -> Iterator[tuple[Path, pd.DataFrame]]:
    """Yield valid contracted dataframes from a directory."""

    paths = sorted(Path(root).glob(pattern))
    yield from iter_contract_dataframes(
        paths,
        contract,
        columns=columns,
        logger=logger,
    )


def _with_default_columns(columns: Iterable[str] | None) -> list[str] | None:
    if columns is None:
        return None
    return list(dict.fromkeys([*DEFAULT_COLUMNS, *columns]))


def _validate_columns(available: Iterable[str], columns: Iterable[str]) -> None:
    available = set(available)
    missing = [column for column in columns if column not in available]
    if missing:
        raise KeyError(f"Missing columns: {missing}")


def _source_name(source: str | Path | pd.DataFrame) -> str:
    if isinstance(source, pd.DataFrame):
        return "<dataframe>"
    return Path(source).name

