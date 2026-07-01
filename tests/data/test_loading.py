from pathlib import Path

from datasets import Dataset, DatasetDict, IterableDataset
import pandas as pd
import pytest

import mt.data as data
from mt.data import DataSource, load


def test_dataframe_source_returns_independent_copy():
    source = pd.DataFrame(
        {"second": [1, 2], "first": ["a", "b"]},
        index=pd.Index([4, 2], name="raw_index"),
    )

    result = load(source)
    result.loc[4, "second"] = 99

    assert result is not source
    assert source.loc[4, "second"] == 1
    assert list(result.columns) == ["second", "first"]
    assert result.index.equals(source.index)
    assert result.dtypes.equals(source.dtypes)


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA, pd.NaT])
def test_column_names_normalize_to_strings(missing):
    source = pd.DataFrame([[1, 2, 3]], columns=[1, missing, ("feature", 2)])

    result = load(source)

    assert list(result.columns) == ["1", "None", "('feature', 2)"]
    assert source.columns[0] == 1
    assert pd.isna(source.columns[1])
    assert source.columns[2] == ("feature", 2)


@pytest.mark.parametrize(
    "columns, duplicate",
    [
        (["choice", "choice"], "choice"),
        ([1, "1"], "1"),
        ([None, "None"], "None"),
    ],
)
def test_duplicate_column_names_after_normalization_fail(columns, duplicate):
    source = pd.DataFrame([[1, 2]], columns=columns)

    with pytest.raises(
        ValueError, match="unique after string normalization"
    ) as error:
        load(source)

    assert duplicate in str(error.value)


@pytest.mark.parametrize("as_string", [False, True])
def test_csv_loads_from_path_or_string(tmp_path, as_string):
    expected = pd.DataFrame({"raw_choice": [1, 2], "raw_label": ["a", "b"]})
    path = tmp_path / "trials.CSV"
    expected.to_csv(path, index=False)
    source = str(path) if as_string else path

    result = load(source)

    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("as_string", [False, True])
def test_parquet_loads_from_path_or_string(tmp_path, as_string):
    expected = pd.DataFrame({"raw_choice": [1, 2], "raw_label": ["a", "b"]})
    path = tmp_path / "trials.PARQUET"
    expected.to_parquet(path, index=False)
    source = str(path) if as_string else path

    result = load(source)

    pd.testing.assert_frame_equal(result, expected)


def test_huggingface_dataset_converts_to_dataframe():
    source = Dataset.from_dict({"raw_choice": [1, 2], "raw_label": ["a", "b"]})

    result = load(source)

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"raw_choice": [1, 2], "raw_label": ["a", "b"]}),
    )


def test_huggingface_dataset_rejects_unexpected_batched_output(monkeypatch):
    source = Dataset.from_dict({"raw_choice": [1]})
    monkeypatch.setattr(Dataset, "to_pandas", lambda self: iter(()))

    with pytest.raises(RuntimeError, match="batched output unexpectedly"):
        load(source)


def test_empty_noncanonical_dataframe_loads_without_validation():
    source = pd.DataFrame(columns=["unknown_raw_field"])

    result = load(source)

    pd.testing.assert_frame_equal(result, source)


@pytest.mark.parametrize(
    "name", ["trials.json", "trials.jsonl", "trials.csv.gz", "owner/dataset"]
)
def test_unsupported_file_format_names_source_and_formats(name):
    with pytest.raises(ValueError) as error:
        load(name)

    message = str(error.value)
    assert name in message
    assert ".csv" in message
    assert ".parquet" in message


def test_unsupported_source_type_names_received_type():
    with pytest.raises(TypeError, match="Unsupported data source type: list"):
        load([])  # type: ignore[arg-type]


def test_dataset_dict_requires_one_split():
    source = DatasetDict({"train": Dataset.from_dict({"raw_choice": [1]})})

    with pytest.raises(TypeError, match="Select one split"):
        load(source)  # type: ignore[arg-type]


def test_iterable_dataset_requires_materialization():
    source = object.__new__(IterableDataset)

    with pytest.raises(TypeError, match="Materialize"):
        load(source)  # type: ignore[arg-type]


def test_missing_supported_file_preserves_file_not_found(tmp_path):
    source = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load(source)


def test_loading_does_not_add_legacy_defaults_or_select_columns():
    source = pd.DataFrame({"raw_a": [1], "raw_b": [2]})

    result = load(source)

    assert list(result.columns) == ["raw_a", "raw_b"]


def test_data_source_alias_accepts_supported_sources():
    source: DataSource = Path("trials.csv")

    assert source == Path("trials.csv")


def test_legacy_loading_api_is_not_exported():
    assert not hasattr(data, "load_dataframe")
    assert not hasattr(data, "load_hf_dataset")
