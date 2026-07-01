from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from mt.data import (
    CANONICAL_COORDINATES,
    CANONICAL_PATHS,
    CANONICAL_SLOTS,
    FIELD_REGISTRY,
    REQUIRED_PATHS,
    FieldSpec,
    get_field_spec,
    is_registered_path,
)

EXPECTED_COORDINATES = (
    "participant_id",
    "session_id",
    "block_index",
    "trial_index",
    "task_name",
    "condition",
)
EXPECTED_SLOTS = ("task", "context", "stimulus", "response", "outcome")
EXPECTED_PATHS = (
    *EXPECTED_COORDINATES,
    "task.instructions",
    "stimulus.ground_truth",
    "stimulus.features",
    "response.choice",
    "response.rt",
    "outcome.reward",
    "outcome.feedback",
)
PATTERN_PATHS = {
    "task.instructions",
    "stimulus.ground_truth",
    "stimulus.features",
    "response.choice",
    "outcome.feedback",
}

def test_field_registry_has_confirmed_order():
    assert CANONICAL_COORDINATES == EXPECTED_COORDINATES
    assert CANONICAL_SLOTS == EXPECTED_SLOTS
    assert CANONICAL_PATHS == EXPECTED_PATHS
    assert tuple(FIELD_REGISTRY) == EXPECTED_PATHS

def test_required_paths_are_task_independent():
    assert REQUIRED_PATHS == (
        "participant_id",
        "trial_index",
        "response.choice",
    )

def test_field_registry_defaults_are_explicit():
    expected_defaults = {
        "session_id": 1,
        "block_index": 1,
        "task_name": 1,
        "condition": 1,
        "task.instructions": None,
        "stimulus.ground_truth": None,
        "stimulus.features": None,
        "response.rt": None,
        "outcome.reward": None,
        "outcome.feedback": None,
    }
    actual_defaults = {
        field.path: field.default
        for field in FIELD_REGISTRY.values()
        if field.has_default
    }

    assert actual_defaults == expected_defaults

def test_field_registry_pattern_targets_are_explicit():
    actual = {
        field.path for field in FIELD_REGISTRY.values() if field.allows_pattern
    }

    assert actual == PATTERN_PATHS

def test_context_is_an_empty_canonical_slot():
    assert "context" in CANONICAL_SLOTS
    assert all(field.slot != "context" for field in FIELD_REGISTRY.values())

def test_field_spec_exposes_path_parts():
    content = get_field_spec("outcome.reward")
    coordinate = get_field_spec("participant_id")

    assert content.slot == "outcome"
    assert content.key == "reward"
    assert not content.is_coordinate
    assert coordinate.slot is None
    assert coordinate.key == "participant_id"
    assert coordinate.is_coordinate

@pytest.mark.parametrize("path", ["context.cue", "outcome.reward.extra", ""])
def test_get_field_spec_rejects_unknown_or_malformed_path(path):
    with pytest.raises(ValueError, match=repr(path)):
        get_field_spec(path)

def test_get_field_spec_rejects_non_string_path():
    with pytest.raises(ValueError, match="must be a string"):
        get_field_spec(None)

def test_is_registered_path_only_accepts_exact_declared_paths():
    assert is_registered_path("response.choice")
    assert not is_registered_path("choice")
    assert not is_registered_path("response.choice.extra")
    assert not is_registered_path(None)

def test_registered_paths_are_unique():
    assert len(CANONICAL_PATHS) == len(set(CANONICAL_PATHS))

def test_field_spec_is_frozen():
    field = get_field_spec("response.choice")

    with pytest.raises(FrozenInstanceError):
        field.required = False

def test_field_registry_is_read_only():
    with pytest.raises(TypeError):
        FIELD_REGISTRY["new.path"] = get_field_spec("response.choice")

def test_every_field_spec_has_a_description():
    assert all(field.description for field in FIELD_REGISTRY.values())

@pytest.mark.parametrize(
    ("path", "description", "kwargs"),
    [
        ("", "description", {}),
        ("unknown", "description", {}),
        ("unknown.key", "description", {}),
        (
            "participant_id",
            "description",
            {"required": True, "has_default": True},
        ),
        ("participant_id", "description", {"allows_pattern": True}),
        ("participant_id", "", {}),
    ],
)
def test_invalid_field_spec_fails(path, description, kwargs):
    with pytest.raises(RuntimeError):
        FieldSpec(path, description, **kwargs)
