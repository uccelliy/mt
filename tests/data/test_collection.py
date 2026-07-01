from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

import mt.data._collection as collection_module
from mt.data import CANONICAL_PATHS, TrialCollection

def collection_values(n_trials=3):
    return {
        "participant_id": np.asarray(["p1", "p1", "p2"][:n_trials]),
        "session_id": np.asarray([1, 1, 1][:n_trials]),
        "block_index": np.asarray([1, 1, 2][:n_trials]),
        "trial_index": np.asarray([1, 2, 1][:n_trials]),
        "task_name": np.asarray(["task", "task", "task"][:n_trials]),
        "condition": np.asarray(["a", "b", "a"][:n_trials]),
        "task": [
            {"instructions": "choose"},
            {"instructions": None},
            {"instructions": "choose"},
        ][:n_trials],
        "context": [{}, {}, {}][:n_trials],
        "stimulus": [
            {"ground_truth": 0, "features": np.asarray([1.0, 2.0])},
            {"ground_truth": 1, "features": np.asarray([3.0, 4.0])},
            {"ground_truth": None, "features": None},
        ][:n_trials],
        "response": [
            {"choice": np.int64(0), "rt": 300.0},
            {"choice": 1, "rt": 450.0},
            {"choice": None, "rt": None},
        ][:n_trials],
        "outcome": [
            {"reward": 1.0, "feedback": "correct"},
            {"reward": 0.0, "feedback": "incorrect"},
            {"reward": None, "feedback": None},
        ][:n_trials],
    }

def make_collection(n_trials=3):
    return TrialCollection(**collection_values(n_trials))

def assert_collection_positions(collection, positions):
    source = make_collection()
    assert collection.participant_id.tolist() == [
        source.participant_id[position] for position in positions
    ]
    assert collection.trial_index.tolist() == [
        source.trial_index[position] for position in positions
    ]
    assert [row["choice"] for row in collection.response] == [
        source.response[position]["choice"] for position in positions
    ]

def test_valid_collection_has_all_coordinates_and_slots():
    collection = make_collection()

    assert collection.n_trials == 3
    assert len(collection) == 3
    assert collection.context == [{}, {}, {}]
    assert set(collection.stimulus[0]) == {"ground_truth", "features"}
    assert isinstance(collection.response[0]["choice"], int)

def test_empty_collection_is_valid():
    collection = make_collection(0)

    assert collection.n_trials == 0
    assert collection.to_dataframe().shape == (0, len(CANONICAL_PATHS))

def test_coordinate_must_be_numpy_array():
    values = collection_values()
    values["participant_id"] = ["p1", "p1", "p2"]

    with pytest.raises(TypeError, match="participant_id.*numpy.ndarray"):
        TrialCollection(**values)

def test_coordinate_must_be_one_dimensional():
    values = collection_values()
    values["participant_id"] = np.asarray([["p1"], ["p1"], ["p2"]])

    with pytest.raises(ValueError, match="participant_id.*one-dimensional"):
        TrialCollection(**values)

def test_coordinate_lengths_must_agree():
    values = collection_values()
    values["session_id"] = np.asarray([1, 1])

    with pytest.raises(ValueError, match="session_id.*length 2.*expected 3"):
        TrialCollection(**values)

def test_coordinate_values_must_be_scalars():
    values = collection_values()
    values["participant_id"] = np.empty(3, dtype=object)
    values["participant_id"][:] = [[1], [2], [3]]

    with pytest.raises(ValueError, match="participant_id.*canonical scalar"):
        TrialCollection(**values)

def test_slot_must_be_list():
    values = collection_values()
    values["context"] = ({}, {}, {})

    with pytest.raises(TypeError, match="context.*must be a list"):
        TrialCollection(**values)

def test_slot_length_must_match_coordinates():
    values = collection_values()
    values["context"] = [{}, {}]

    with pytest.raises(ValueError, match="context.*length 2.*expected 3"):
        TrialCollection(**values)

def test_slot_items_must_be_dictionaries():
    values = collection_values()
    values["context"][1] = None

    with pytest.raises(TypeError, match="context.*position 1.*dictionary"):
        TrialCollection(**values)

@pytest.mark.parametrize(
    ("slot", "row"),
    [
        ("task", {}),
        ("context", {"cue": 1}),
        ("stimulus", {"ground_truth": 0}),
        ("response", {"choice": 0, "rt": None, "reward": 1}),
    ],
)
def test_slot_keys_must_match_registry(slot, row):
    values = collection_values()
    values[slot][0] = row

    with pytest.raises(ValueError, match=f"Slot {slot!r}.*invalid keys"):
        TrialCollection(**values)

@pytest.mark.parametrize("value", [[1, 2], {"nested": 1}, object()])
def test_slot_values_reject_non_canonical_types(value):
    values = collection_values()
    values["response"][0]["choice"] = value

    with pytest.raises(ValueError, match="response.choice.*position 0"):
        TrialCollection(**values)

def test_constructor_copies_owned_runtime_values():
    values = collection_values()
    participant_id = values["participant_id"]
    task = values["task"]
    features = values["stimulus"][0]["features"]
    collection = TrialCollection(**values)

    participant_id[0] = "changed"
    task[0]["instructions"] = "changed"
    features[0] = -1

    assert collection.participant_id[0] == "p1"
    assert collection.task[0]["instructions"] == "choose"
    assert collection.stimulus[0]["features"].tolist() == [1.0, 2.0]

def test_copy_has_independent_runtime_values():
    source = make_collection()
    copied = source.copy()

    copied.participant_id[0] = "changed"
    copied.task[0]["instructions"] = "changed"
    copied.stimulus[0]["features"][0] = -1

    assert source.participant_id[0] == "p1"
    assert source.task[0]["instructions"] == "choose"
    assert source.stimulus[0]["features"].tolist() == [1.0, 2.0]

@pytest.mark.parametrize(
    ("selector", "positions"),
    [
        (slice(1, None), [1, 2]),
        ([2, 0, 2], [2, 0, 2]),
        ([-1, 0], [2, 0]),
        ([True, False, True], [0, 2]),
    ],
)
def test_select_keeps_every_field_aligned(selector, positions):
    selected = make_collection().select(selector)

    assert_collection_positions(selected, positions)

def test_empty_selection_returns_empty_collection():
    selected = make_collection().select([])

    assert selected.n_trials == 0
    assert selected.context == []

@pytest.mark.parametrize(
    ("selector", "message"),
    [
        ([[0, 1]], "one-dimensional"),
        ([True, False], "length 2; expected 3"),
        ([True, 1, False], "all integers or all booleans"),
        ([0.0, 1.0], "all integers or all booleans"),
        ([3], "out of range"),
    ],
)
def test_select_rejects_invalid_selectors(selector, message):
    with pytest.raises(ValueError, match=message):
        make_collection().select(selector)

def test_selected_values_are_independent():
    source = make_collection()
    selected = source.select([0])

    selected.participant_id[0] = "changed"
    selected.task[0]["instructions"] = "changed"
    selected.stimulus[0]["features"][0] = -1

    assert source.participant_id[0] == "p1"
    assert source.task[0]["instructions"] == "choose"
    assert source.stimulus[0]["features"].tolist() == [1.0, 2.0]

def test_to_dataframe_uses_canonical_order_and_independent_arrays():
    source = make_collection()
    frame = source.to_dataframe()

    assert isinstance(frame, pd.DataFrame)
    assert tuple(frame.columns) == CANONICAL_PATHS
    assert isinstance(frame.index, pd.RangeIndex)

    frame.loc[0, "participant_id"] = "changed"
    frame.at[0, "stimulus.features"][0] = -1

    assert source.participant_id[0] == "p1"
    assert source.stimulus[0]["features"].tolist() == [1.0, 2.0]

def test_collection_is_public_and_does_not_import_models():
    assert TrialCollection is collection_module.TrialCollection
    assert "mt.models" not in inspect.getsource(collection_module)
