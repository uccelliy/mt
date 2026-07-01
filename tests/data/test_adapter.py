from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import mt.data._adapter as adapter_module
from mt.data import (
    CANONICAL_COORDINATES,
    CANONICAL_PATHS,
    CANONICAL_SLOTS,
    AdaptationResult,
    ColumnMapping,
    DataAdapter,
    TrialCollection,
)
from mt.data._adapter import (
    apply_defaults,
    assemble_trials,
    build_collection,
    copy_default,
    filter_rows,
    normalize_missing_values,
    validate_dataframe,
)

def mapped_frame():
    return pd.DataFrame(
        {
            "participant_id": ["p1", "p1", "p2"],
            "trial_index": [1, 2, 1],
            "response.choice": [0, np.nan, 1],
        },
        index=[10, 11, 12],
    )

def canonical_frame():
    frame = apply_defaults(mapped_frame())
    frame["task_name"] = ["task", "task", "task"]
    frame["condition"] = ["a", "b", "a"]
    return normalize_missing_values(frame)

def raw_frame():
    return pd.DataFrame(
        {
            "subject": ["p1", "p1", "p2"],
            "trial": [1, 2, 1],
            "choice": [0, np.nan, 1],
            "unused": ["x", "y", "z"],
        },
        index=[10, 11, 12],
    )

def raw_mapping():
    return ColumnMapping(
        mappings={
            "participant_id": "subject",
            "trial_index": "trial",
            "response.choice": "choice",
        }
    )

def collection_groups(frame=None):
    if frame is None:
        frame = canonical_frame()
    collection = assemble_trials(frame)
    coordinates = {
        name: getattr(collection, name) for name in CANONICAL_COORDINATES
    }
    slots = {name: getattr(collection, name) for name in CANONICAL_SLOTS}
    return coordinates, slots

def test_apply_defaults_adds_every_registered_default_in_order():
    result = apply_defaults(mapped_frame())

    assert tuple(result.columns) == CANONICAL_PATHS
    assert result["session_id"].tolist() == [1, 1, 1]
    assert result["task.instructions"].tolist() == [None, None, None]

def test_apply_defaults_does_not_fill_existing_missing_cells():
    frame = mapped_frame()
    frame["session_id"] = [1, np.nan, 2]

    result = apply_defaults(frame)

    assert pd.isna(result["session_id"].iloc[1])

def test_apply_defaults_leaves_missing_required_paths_for_validation():
    result = apply_defaults(pd.DataFrame({"participant_id": ["p1"]}))

    assert "trial_index" not in result.columns
    assert "response.choice" not in result.columns

def test_apply_defaults_does_not_mutate_input():
    frame = mapped_frame()
    result = apply_defaults(frame)

    result.loc[10, "participant_id"] = "changed"

    assert tuple(frame.columns) == (
        "participant_id",
        "trial_index",
        "response.choice",
    )
    assert frame.loc[10, "participant_id"] == "p1"

def test_apply_defaults_supports_empty_dataframes():
    result = apply_defaults(mapped_frame().iloc[0:0])

    assert tuple(result.columns) == CANONICAL_PATHS
    assert result.empty

def test_copy_default_copies_mutable_numpy_values():
    value = np.asarray([1.0, 2.0])
    first = copy_default(value)
    second = copy_default(value)

    first[0] = -1

    assert value.tolist() == [1.0, 2.0]
    assert second.tolist() == [1.0, 2.0]

@pytest.mark.parametrize("function", [apply_defaults, normalize_missing_values])
def test_canonical_transforms_reject_unknown_paths(function):
    with pytest.raises(ValueError, match="Unknown canonical"):
        function(pd.DataFrame({"raw": [1]}))

def test_normalize_missing_values_uses_none_for_all_scalar_sentinels():
    frame = pd.DataFrame(
        {
            "response.choice": pd.Series(
                [
                    None,
                    float("nan"),
                    np.float32("nan"),
                    pd.NA,
                    pd.NaT,
                    np.datetime64("NaT"),
                    np.timedelta64("NaT"),
                ],
                dtype=object,
            )
        }
    )

    result = normalize_missing_values(frame)

    assert result["response.choice"].tolist() == [None] * 7
    assert result["response.choice"].dtype == object

def test_normalize_missing_values_preserves_non_missing_values():
    frame = pd.DataFrame(
        {"response.choice": [0.0, np.inf, -np.inf, -999.0, ""]}
    )

    result = normalize_missing_values(frame)

    assert result["response.choice"].tolist() == [
        0.0,
        np.inf,
        -np.inf,
        -999.0,
        "",
    ]

def test_normalize_missing_values_leaves_array_contents_unchanged():
    features = np.asarray([1.0, np.nan, 3.0])
    frame = pd.DataFrame(
        {"stimulus.features": pd.Series([features], dtype=object)}
    )

    result = normalize_missing_values(frame)

    assert result["stimulus.features"].iloc[0] is features
    assert np.isnan(result["stimulus.features"].iloc[0][1])
    assert np.isnan(frame["stimulus.features"].iloc[0][1])

def test_normalize_missing_values_does_not_mutate_input():
    frame = pd.DataFrame({"response.choice": [np.nan]})
    result = normalize_missing_values(frame)

    assert result["response.choice"].iloc[0] is None
    assert pd.isna(frame["response.choice"].iloc[0])

@pytest.mark.parametrize(
    ("criterion", "expected"),
    [
        ("p2", [12]),
        (["p1"], [10, 11]),
        (("p1",), [10, 11]),
        (np.asarray(["p2"]), [12]),
        (pd.Series(["p1"]), [10, 11]),
    ],
)
def test_filter_rows_accepts_deterministic_value_forms(criterion, expected):
    result = filter_rows(canonical_frame(), {"participant_id": criterion})

    assert result.index.tolist() == expected

def test_filter_rows_combines_coordinates_with_and():
    result = filter_rows(
        canonical_frame(),
        {"participant_id": ["p1"], "condition": "b"},
    )

    assert result.index.tolist() == [11]

def test_filter_rows_empty_criteria_returns_independent_copy():
    frame = canonical_frame()
    result = filter_rows(frame, {})

    result.loc[10, "participant_id"] = "changed"

    assert frame.loc[10, "participant_id"] == "p1"

def test_filter_rows_empty_values_selects_zero_rows():
    result = filter_rows(canonical_frame(), {"participant_id": []})

    assert result.empty

@pytest.mark.parametrize(
    "criterion",
    [
        {"p1"},
        {"participant": "p1"},
        (value for value in ["p1"]),
        np.asarray([["p1"]]),
    ],
)
def test_filter_rows_rejects_nondeterministic_or_shaped_values(criterion):
    with pytest.raises(ValueError, match="Filter values"):
        filter_rows(canonical_frame(), {"participant_id": criterion})

def test_filter_rows_rejects_non_coordinate_paths():
    with pytest.raises(ValueError, match="canonical coordinate"):
        filter_rows(canonical_frame(), {"response.choice": [0]})

def test_filter_rows_rejects_missing_coordinates():
    frame = canonical_frame().drop(columns="condition")

    with pytest.raises(ValueError, match="Filter coordinate is missing"):
        filter_rows(frame, {"condition": "a"})

def test_filter_rows_rejects_non_mapping_criteria():
    with pytest.raises(TypeError, match="must be a mapping"):
        filter_rows(canonical_frame(), ["p1"])

def test_repeated_filter_rows_calls_compose():
    frame = filter_rows(canonical_frame(), {"participant_id": "p1"})
    result = filter_rows(frame, {"condition": "a"})

    assert result.index.tolist() == [10]

def test_validate_dataframe_returns_independent_valid_copy():
    frame = canonical_frame()
    result = validate_dataframe(frame)

    result.loc[10, "participant_id"] = "changed"

    assert frame.loc[10, "participant_id"] == "p1"

def test_validate_dataframe_rejects_missing_paths():
    frame = canonical_frame().drop(columns="response.rt")

    with pytest.raises(ValueError, match="missing paths.*response.rt"):
        validate_dataframe(frame)

def test_validate_dataframe_rejects_duplicate_columns():
    frame = canonical_frame()
    frame.columns = [*frame.columns[:-1], frame.columns[-2]]

    with pytest.raises(ValueError, match="columns must be unique"):
        validate_dataframe(frame)

@pytest.mark.parametrize("path", CANONICAL_COORDINATES)
def test_validate_dataframe_rejects_missing_coordinate_cells(path):
    frame = canonical_frame()
    frame[path] = frame[path].astype(object)
    frame.at[10, path] = None

    with pytest.raises(ValueError, match=f"Coordinate {path!r} is missing"):
        validate_dataframe(frame)

def test_validate_dataframe_allows_missing_content_and_arrays():
    frame = canonical_frame()
    features = np.asarray([1.0, np.nan])
    frame["stimulus.features"] = pd.Series(
        [features, None, None], index=frame.index, dtype=object
    )

    result = validate_dataframe(frame)

    assert np.isnan(result["stimulus.features"].iloc[0][1])
    assert result["response.choice"].iloc[1] is None

def test_validate_dataframe_rejects_unnormalized_missing_content():
    frame = canonical_frame()
    frame["response.rt"] = [1.0, np.nan, 2.0]

    with pytest.raises(ValueError, match="unnormalized missing scalar"):
        validate_dataframe(frame)

def test_validate_dataframe_rejects_unsupported_content_values():
    frame = canonical_frame()
    frame["response.choice"] = pd.Series(
        [[0], 1, 0], index=frame.index, dtype=object
    )

    with pytest.raises(ValueError, match="response.choice.*unsupported"):
        validate_dataframe(frame)

def test_validate_dataframe_rejects_duplicate_identity_without_condition():
    frame = canonical_frame().iloc[:2].copy()
    frame["trial_index"] = [1, 1]
    frame["condition"] = ["a", "b"]

    with pytest.raises(ValueError, match=r"row positions \(0, 1\).*"):
        validate_dataframe(frame)

def test_validate_dataframe_allows_empty_canonical_data():
    result = validate_dataframe(canonical_frame().iloc[0:0])

    assert result.empty

def test_assemble_trials_preserves_alignment_and_discards_index():
    collection = assemble_trials(validate_dataframe(canonical_frame()))

    assert collection.participant_id.tolist() == ["p1", "p1", "p2"]
    assert collection.trial_index.tolist() == [1, 2, 1]
    assert [row["choice"] for row in collection.response] == [0, None, 1]
    assert collection.context == [{}, {}, {}]

def test_assemble_trials_normalizes_numpy_scalars_and_copies_arrays():
    frame = canonical_frame()
    features = np.asarray([1.0, np.nan])
    frame["response.choice"] = pd.Series(
        [np.int64(0), None, np.int64(1)], index=frame.index, dtype=object
    )
    frame["stimulus.features"] = pd.Series(
        [features, None, None], index=frame.index, dtype=object
    )

    collection = assemble_trials(validate_dataframe(frame))
    collection.stimulus[0]["features"][0] = -1

    assert isinstance(collection.response[0]["choice"], int)
    assert features.tolist()[0] == 1.0
    assert np.isnan(collection.stimulus[0]["features"][1])

def test_assemble_trials_rejects_missing_paths():
    with pytest.raises(ValueError, match="Cannot assemble missing"):
        assemble_trials(canonical_frame().drop(columns="response.rt"))

@pytest.mark.parametrize("group", ["coordinates", "slots"])
def test_build_collection_rejects_invalid_group_keys(group):
    coordinates, slots = collection_groups()
    values = coordinates if group == "coordinates" else slots
    values.pop(next(iter(values)))

    with pytest.raises(ValueError, match=f"Collection {group} keys"):
        build_collection(coordinates, slots)

def test_data_adapter_runs_complete_pipeline_and_returns_report():
    source = raw_frame()
    result = DataAdapter(raw_mapping()).adapt(
        source, filters={"participant_id": ["p1"]}
    )

    assert isinstance(result, AdaptationResult)
    assert isinstance(result.collection, TrialCollection)
    assert result.input_rows == 3
    assert result.output_rows == 2
    assert result.collection.response[1]["choice"] is None
    assert "session_id" in result.defaulted_paths
    assert result.mapping.ignored_columns == ("unused",)
    assert source.loc[11, "choice"] is np.nan or pd.isna(
        source.loc[11, "choice"]
    )

def test_data_adapter_can_be_reused_without_retaining_run_state():
    adapter = DataAdapter(raw_mapping())
    first = adapter.adapt(raw_frame(), filters={"participant_id": "p1"})
    second = adapter.adapt(raw_frame(), filters={"participant_id": "p2"})

    assert first.collection.participant_id.tolist() == ["p1", "p1"]
    assert second.collection.participant_id.tolist() == ["p2"]

def test_data_adapter_propagates_stage_errors_without_result():
    with pytest.raises(ValueError, match="Required canonical path"):
        DataAdapter().adapt(pd.DataFrame({"participant_id": ["p1"]}))

def test_data_adapter_stops_after_first_unexpected_stage_error(monkeypatch):
    later_stage_called = False

    def fail_defaults(df):
        raise RuntimeError("stage bug")

    def record_normalization(df):
        nonlocal later_stage_called
        later_stage_called = True
        return df

    monkeypatch.setattr(adapter_module, "apply_defaults", fail_defaults)
    monkeypatch.setattr(
        adapter_module, "normalize_missing_values", record_normalization
    )

    with pytest.raises(RuntimeError, match="stage bug"):
        DataAdapter(raw_mapping()).adapt(raw_frame())

    assert not later_stage_called

def test_data_adapter_exposes_no_mutable_stage_methods():
    adapter = DataAdapter()

    for name in ("load", "map", "defaults", "filter", "validate", "assemble"):
        assert not hasattr(adapter, name)

def test_adaptation_result_report_is_deterministic_success_metadata():
    result = DataAdapter(raw_mapping()).adapt(raw_frame())

    report = result.report()

    assert report == result.report()
    assert "Adaptation succeeded." in report
    assert "Rows: 3 input -> 3 output" in report
    assert "participant_id <- subject" in report
    assert "session_id <- 1 (default)" in report
    assert "unused" in report
    assert "Scalar missing value: None" in report

def test_adaptation_result_rejects_inconsistent_row_counts():
    result = DataAdapter(raw_mapping()).adapt(raw_frame())

    with pytest.raises(ValueError, match="collection.n_trials"):
        AdaptationResult(
            collection=result.collection,
            mapping=result.mapping,
            defaulted_paths=result.defaulted_paths,
            input_rows=3,
            output_rows=2,
        )
