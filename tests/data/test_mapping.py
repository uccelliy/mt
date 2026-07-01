from dataclasses import FrozenInstanceError
import warnings

import numpy as np
import pandas as pd
import pytest

import mt.data as data
from mt.data import ColumnMapping, MappingResolution


def raw_frame(**columns) -> pd.DataFrame:
    values = {
        "participant_id": ["p1", "p2"],
        "trial_index": [1, 2],
        "choice": [0, 1],
    }
    values.update(columns)
    return pd.DataFrame(values)


def test_empty_mapping_resolves_identity_columns_in_registry_order():
    df = raw_frame(notes=["a", "b"])
    mapping = ColumnMapping()

    resolution = mapping.resolve(df)
    result = mapping.apply(df)

    assert tuple(resolution.sources) == (
        "participant_id",
        "trial_index",
        "response.choice",
    )
    assert list(result.columns) == list(resolution.sources)
    assert resolution.ignored_columns == ("notes",)


def test_mapping_configuration_and_resolution_are_immutable():
    fixed = {"response.choice": "resp"}
    mapping = ColumnMapping(mappings=fixed)
    fixed["response.choice"] = "changed"
    resolution = mapping.resolve(
        pd.DataFrame(
            {
                "participant_id": ["p1"],
                "trial_index": [1],
                "resp": [0],
            }
        )
    )

    assert mapping.mappings["response.choice"] == "resp"
    with pytest.raises(TypeError):
        mapping.mappings["response.choice"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        resolution.sources["response.choice"] = ("changed",)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        resolution.ignored_columns = ()  # type: ignore[misc]


@pytest.mark.parametrize("argument", [[], "not-a-mapping", None])
def test_configuration_requires_mapping_objects(argument):
    with pytest.raises(TypeError, match="mappings must be a mapping"):
        ColumnMapping(mappings=argument)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="patterns must be a mapping"):
        ColumnMapping(patterns=argument)  # type: ignore[arg-type]


def test_unknown_target_uses_registry_error():
    with pytest.raises(
        ValueError, match="Unknown canonical path: 'context.cue'"
    ):
        ColumnMapping(mappings={"context.cue": "cue"})


@pytest.mark.parametrize("source", ["", 1, None])
def test_fixed_source_must_be_non_empty_string(source):
    with pytest.raises(ValueError, match="response.choice"):
        ColumnMapping(mappings={"response.choice": source})  # type: ignore[dict-item]


@pytest.mark.parametrize("expression", ["", 1, None])
def test_pattern_must_be_non_empty_string(expression):
    with pytest.raises(ValueError, match="stimulus.features"):
        ColumnMapping(patterns={"stimulus.features": expression})  # type: ignore[dict-item]


def test_invalid_and_disallowed_patterns_fail_at_construction():
    with pytest.raises(ValueError, match="Invalid pattern.*stimulus.features"):
        ColumnMapping(patterns={"stimulus.features": "["})

    with pytest.raises(ValueError, match="participant_id.*does not allow"):
        ColumnMapping(patterns={"participant_id": r"participant_\d+"})

    with pytest.raises(ValueError, match="response.rt.*does not allow"):
        ColumnMapping(patterns={"response.rt": r"rt_\d+"})


def test_one_target_cannot_have_fixed_and_pattern_mapping():
    with pytest.raises(ValueError, match="use only a pattern"):
        ColumnMapping(
            mappings={"stimulus.features": "feature"},
            patterns={"stimulus.features": r"feature_\d+"},
        )


def test_explicit_coordinate_and_content_mappings_use_canonical_paths():
    df = pd.DataFrame(
        {"subject": ["p1"], "trial_no": [1], "resp": [2], "points": [5]}
    )
    mapping = ColumnMapping(
        mappings={
            "participant_id": "subject",
            "trial_index": "trial_no",
            "response.choice": "resp",
            "outcome.reward": "points",
        }
    )

    result = mapping.apply(df)

    assert list(result.columns) == [
        "participant_id",
        "trial_index",
        "response.choice",
        "outcome.reward",
    ]
    assert result.iloc[0].tolist() == ["p1", 1, 2, 5]


def test_full_content_path_is_not_an_implicit_raw_identity():
    df = pd.DataFrame(
        {
            "participant_id": ["p1"],
            "trial_index": [1],
            "response.choice": [2],
        }
    )

    with pytest.raises(ValueError, match="raw column 'choice' was not found"):
        ColumnMapping().apply(df)

    result = ColumnMapping(
        mappings={"response.choice": "response.choice"}
    ).apply(df)
    assert result["response.choice"].tolist() == [2]


def test_explicit_mapping_overrides_identity_source():
    df = raw_frame(resp=[2, 3])
    mapping = ColumnMapping(mappings={"response.choice": "resp"})

    resolution = mapping.resolve(df)
    result = mapping.apply(df)

    assert result["response.choice"].tolist() == [2, 3]
    assert "choice" in resolution.ignored_columns


def test_missing_explicit_source_fails_even_for_optional_target():
    with pytest.raises(
        ValueError, match="'points'.*'outcome.reward'.*not found"
    ):
        ColumnMapping(mappings={"outcome.reward": "points"}).apply(raw_frame())


@pytest.mark.parametrize(
    "df, target, identity",
    [
        (
            pd.DataFrame({"trial_index": [1], "choice": [0]}),
            "participant_id",
            "participant_id",
        ),
        (
            pd.DataFrame({"participant_id": ["p1"], "trial_index": [1]}),
            "response.choice",
            "choice",
        ),
    ],
)
def test_missing_required_identity_names_target_and_raw_name(
    df, target, identity
):
    with pytest.raises(ValueError) as error:
        ColumnMapping().resolve(df)

    assert target in str(error.value)
    assert identity in str(error.value)


def test_optional_fields_are_omitted_and_defaults_are_not_applied():
    result = ColumnMapping().apply(raw_frame())

    assert "session_id" not in result
    assert "task.instructions" not in result
    assert "outcome.reward" not in result


def test_resolution_records_ignored_columns_and_output_preserves_index():
    df = raw_frame(notes=["a", "b"])
    df.index = pd.Index([5, 3], name="raw_index")
    source = df.copy(deep=True)
    mapping = ColumnMapping()

    resolution = mapping.resolve(df)
    result = mapping.apply(df)

    assert resolution.ignored_columns == ("notes",)
    assert "notes" not in result
    assert result.index.equals(df.index)
    pd.testing.assert_frame_equal(df, source)


def test_duplicate_raw_column_names_fail_at_mapping_entry():
    df = pd.DataFrame(
        [["p1", 1, 0, 1]],
        columns=[
            "participant_id",
            "trial_index",
            "choice",
            "choice",
        ],
    )

    with pytest.raises(ValueError, match="must be unique.*choice"):
        ColumnMapping().apply(df)


def test_non_dataframe_input_fails_at_mapping_entry():
    with pytest.raises(TypeError, match="requires pandas.DataFrame, got list"):
        ColumnMapping().apply([])  # type: ignore[arg-type]


def test_empty_dataframe_maps_required_and_pattern_columns():
    df = pd.DataFrame(
        columns=["participant_id", "trial_index", "choice", "feature_1"]
    )
    mapping = ColumnMapping(patterns={"stimulus.features": r"feature_\d+"})

    result = mapping.apply(df)

    assert list(result.columns) == [
        "participant_id",
        "trial_index",
        "stimulus.features",
        "response.choice",
    ]
    assert result.empty
    assert result["stimulus.features"].dtype == object


def test_pattern_uses_fullmatch_and_records_ignored_partial_matches():
    df = raw_frame(feature_1=[10, 11], prefix_feature_2=[20, 21])
    mapping = ColumnMapping(patterns={"stimulus.features": r"feature_\d+"})

    resolution = mapping.resolve(df)
    result = mapping.apply(df)

    assert resolution.sources["stimulus.features"] == ("feature_1",)
    assert "prefix_feature_2" in resolution.ignored_columns
    np.testing.assert_array_equal(result["stimulus.features"].iloc[0], [10])


def test_numeric_index_capture_sorts_numerically():
    df = raw_frame(feature_10=[10, 11], feature_2=[2, 3])
    mapping = ColumnMapping(
        patterns={"stimulus.features": r"feature_(?P<index>\d+)"}
    )

    resolution = mapping.resolve(df)
    result = mapping.apply(df)

    assert resolution.sources["stimulus.features"] == (
        "feature_2",
        "feature_10",
    )
    np.testing.assert_array_equal(result["stimulus.features"].iloc[0], [2, 10])


def test_equal_numeric_indices_preserve_source_order():
    df = raw_frame(feature_1_b=[2, 3], feature_1_a=[1, 2])
    mapping = ColumnMapping(
        patterns={
            "stimulus.features": r"feature_(?P<index>\d+)_[ab]",
        }
    )

    resolution = mapping.resolve(df)

    assert resolution.sources["stimulus.features"] == (
        "feature_1_b",
        "feature_1_a",
    )


def test_pattern_without_index_preserves_dataframe_order():
    df = raw_frame(feature_b=[2, 3], feature_a=[1, 2])
    mapping = ColumnMapping(patterns={"stimulus.features": r"feature_[ab]"})

    resolution = mapping.resolve(df)
    result = mapping.apply(df)

    assert resolution.sources["stimulus.features"] == (
        "feature_b",
        "feature_a",
    )
    np.testing.assert_array_equal(result["stimulus.features"].iloc[0], [2, 1])


@pytest.mark.parametrize(
    "pattern, column, capture",
    [
        (r"feature_(?P<index>[a-z]+)", "feature_first", "first"),
        (r"(?:feature_(?P<index>\d+)|other)", "other", None),
    ],
)
def test_index_capture_must_be_present_and_numeric(pattern, column, capture):
    df = raw_frame(**{column: [1, 2]})
    mapping = ColumnMapping(patterns={"stimulus.features": pattern})

    with pytest.raises(ValueError) as error:
        mapping.resolve(df)

    assert "numeric 'index' capture" in str(error.value)
    assert column in str(error.value)
    assert repr(capture) in str(error.value)


def test_explicit_pattern_with_no_matches_fails():
    mapping = ColumnMapping(patterns={"stimulus.features": r"feature_\d+"})

    with pytest.raises(ValueError, match="matched no raw columns"):
        mapping.resolve(raw_frame())


def test_scalar_pattern_values_stack_per_row_and_one_match_keeps_axis():
    df = raw_frame(feature_1=[10, 11], feature_2=[20, 21])
    mapping = ColumnMapping(patterns={"stimulus.features": r"feature_\d+"})

    result = mapping.apply(df)

    first = result["stimulus.features"].iloc[0]
    second = result["stimulus.features"].iloc[1]
    np.testing.assert_array_equal(first, [10, 20])
    np.testing.assert_array_equal(second, [11, 21])
    assert first.shape == (2,)

    one = ColumnMapping(patterns={"stimulus.features": "feature_1"}).apply(df)
    assert one["stimulus.features"].iloc[0].shape == (1,)


def test_array_valued_pattern_cells_stack_along_last_axis():
    feature_1 = [np.array([1, 2, 3]), np.array([4, 5, 6])]
    feature_2 = [np.array([10, 20, 30]), np.array([40, 50, 60])]
    df = raw_frame(feature_1=feature_1, feature_2=feature_2)
    mapping = ColumnMapping(patterns={"stimulus.features": r"feature_\d+"})

    result = mapping.apply(df)

    np.testing.assert_array_equal(
        result["stimulus.features"].iloc[0],
        [[1, 10], [2, 20], [3, 30]],
    )
    assert result["stimulus.features"].iloc[0].shape == (3, 2)


def test_pattern_arrays_are_independent_from_source_and_other_rows():
    source_array = np.array([1, 2])
    df = raw_frame(
        feature_1=[source_array, np.array([3, 4])],
        feature_2=[np.array([5, 6]), np.array([7, 8])],
    )
    mapping = ColumnMapping(patterns={"stimulus.features": r"feature_\d+"})

    result = mapping.apply(df)
    result["stimulus.features"].iloc[0][0, 0] = 99

    assert source_array[0] == 1
    assert result["stimulus.features"].iloc[1][0, 0] == 3


def test_incompatible_pattern_shapes_name_target_and_row():
    df = raw_frame(
        feature_1=[np.array([1, 2]), np.array([3, 4])],
        feature_2=[np.array([5, 6, 7]), np.array([8, 9])],
    )
    mapping = ColumnMapping(patterns={"stimulus.features": r"feature_\d+"})

    with pytest.raises(ValueError) as error:
        mapping.apply(df)

    assert "stimulus.features" in str(error.value)
    assert "row position 0" in str(error.value)


def test_fixed_source_reuse_warns_once_and_copies_values():
    df = pd.DataFrame(
        {"participant_id": ["p1"], "trial_index": [1], "answer": [2]}
    )
    mapping = ColumnMapping(
        mappings={
            "stimulus.ground_truth": "answer",
            "response.choice": "answer",
        }
    )

    resolution = mapping.resolve(df)
    with pytest.warns(UserWarning, match="answer") as warning_records:
        result = mapping.apply(df)

    assert len(warning_records) == 1
    assert resolution.reused_columns == {
        "answer": ("stimulus.ground_truth", "response.choice")
    }
    assert result["stimulus.ground_truth"].tolist() == [2]
    assert result["response.choice"].tolist() == [2]


def test_resolution_records_reuse_without_warning():
    df = pd.DataFrame(
        {"participant_id": ["p1"], "trial_index": [1], "answer": [2]}
    )
    mapping = ColumnMapping(
        mappings={
            "stimulus.ground_truth": "answer",
            "response.choice": "answer",
        }
    )

    with warnings.catch_warnings(record=True) as warning_records:
        resolution = mapping.resolve(df)

    assert not warning_records
    assert tuple(resolution.reused_columns) == ("answer",)


def test_fixed_pattern_overlap_warns_and_pattern_remains_array():
    df = pd.DataFrame(
        {"participant_id": ["p1"], "trial_index": [1], "answer": [2]}
    )
    mapping = ColumnMapping(
        mappings={"response.choice": "answer"},
        patterns={"stimulus.features": "answer"},
    )

    with pytest.warns(UserWarning, match="answer"):
        result = mapping.apply(df)

    np.testing.assert_array_equal(result["stimulus.features"].iloc[0], [2])
    assert result["response.choice"].tolist() == [2]


def test_pattern_overlap_warns_in_raw_column_order():
    df = raw_frame(x_2=[2, 3], x_1=[1, 2])
    mapping = ColumnMapping(
        patterns={
            "stimulus.features": r"x_\d",
            "outcome.feedback": r"x_\d",
        }
    )

    with pytest.warns(UserWarning) as warning_records:
        resolution = mapping.resolve(df)
        mapping.apply(df)

    assert not resolution.ignored_columns
    assert ["x_2" in str(record.message) for record in warning_records] == [
        True,
        False,
    ]
    assert "x_1" in str(warning_records[1].message)


def test_public_mapping_types_are_exported():
    assert data.ColumnMapping is ColumnMapping
    assert data.MappingResolution is MappingResolution
