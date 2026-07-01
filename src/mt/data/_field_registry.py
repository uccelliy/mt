"""Canonical field registry for cognitive trial data."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

CanonicalScalar = str | int | float | bool | None
CanonicalValue = CanonicalScalar | np.ndarray

CANONICAL_COORDINATES = (
    "participant_id",
    "session_id",
    "block_index",
    "trial_index",
    "task_name",
    "condition",
)
CANONICAL_SLOTS = ("task", "context", "stimulus", "response", "outcome")

@dataclass(frozen=True)
class FieldSpec:
    path: str
    description: str
    required: bool = False
    has_default: bool = False
    default: CanonicalValue = None
    allows_pattern: bool = False

    def __post_init__(self):
        if not self.path:
            raise RuntimeError("FieldSpec path must not be empty.")
        if not self.description:
            raise RuntimeError(
                f"FieldSpec {self.path!r} must have a description."
            )
        if self.is_coordinate:
            if self.path not in CANONICAL_COORDINATES:
                raise RuntimeError(
                    f"Unknown canonical coordinate: {self.path!r}."
                )
        elif self.path.count(".") != 1 or not self.key:
            raise RuntimeError(
                f"Content field path must use 'slot.key': {self.path!r}."
            )
        elif self.slot not in CANONICAL_SLOTS:
            raise RuntimeError(
                f"Unknown canonical slot in path: {self.path!r}."
            )

        if self.required and self.has_default:
            raise RuntimeError(
                "Required registered field cannot have a default: "
                f"{self.path!r}."
            )
        if self.allows_pattern and self.is_coordinate:
            raise RuntimeError(
                f"Coordinate cannot allow pattern mapping: {self.path!r}."
            )

    @property
    def is_coordinate(self):
        return "." not in self.path

    @property
    def slot(self) -> str | None:
        if self.is_coordinate:
            return None
        return self.path.split(".", 1)[0]

    @property
    def key(self):
        if self.is_coordinate:
            return self.path
        return self.path.split(".", 1)[1]

_FIELD_SPECS = (
    FieldSpec(
        "participant_id",
        "Participant identifier.",
        required=True,
    ),
    FieldSpec(
        "session_id",
        "Session identifier.",
        has_default=True,
        default=1,
    ),
    FieldSpec(
        "block_index",
        "Block index within a session.",
        has_default=True,
        default=1,
    ),
    FieldSpec(
        "trial_index",
        "Trial index within a block.",
        required=True,
    ),
    FieldSpec(
        "task_name",
        "Task identifier.",
        has_default=True,
        default=1,
    ),
    FieldSpec(
        "condition",
        "Condition identifier.",
        has_default=True,
        default=1,
    ),
    FieldSpec(
        "task.instructions",
        "Natural-language task instructions.",
        has_default=True,
        allows_pattern=True,
    ),
    FieldSpec(
        "stimulus.ground_truth",
        "Correct label for the current stimulus.",
        has_default=True,
        allows_pattern=True,
    ),
    FieldSpec(
        "stimulus.features",
        "Generic stimulus feature vector.",
        has_default=True,
        allows_pattern=True,
    ),
    FieldSpec(
        "response.choice",
        "Participant choice.",
        required=True,
        allows_pattern=True,
    ),
    FieldSpec(
        "response.rt",
        "Response time in milliseconds.",
        has_default=True,
    ),
    FieldSpec(
        "outcome.reward",
        "Objective consequence received after a response.",
        has_default=True,
    ),
    FieldSpec(
        "outcome.feedback",
        "Feedback presented to the participant.",
        has_default=True,
        allows_pattern=True,
    ),
)
CANONICAL_PATHS = tuple(field.path for field in _FIELD_SPECS)
REQUIRED_PATHS = tuple(field.path for field in _FIELD_SPECS if field.required)
_DUPLICATE_PATHS = tuple(
    path for path in CANONICAL_PATHS if CANONICAL_PATHS.count(path) > 1
)
if _DUPLICATE_PATHS:
    raise RuntimeError(
        f"Duplicate canonical field paths: {_DUPLICATE_PATHS}."
    )
FIELD_REGISTRY: Mapping[str, FieldSpec] = MappingProxyType(
    {field.path: field for field in _FIELD_SPECS}
)

def get_field_spec(path) -> FieldSpec:
    if not isinstance(path, str):
        raise ValueError(
            f"Canonical path must be a string, got {type(path).__name__}."
        )
    try:
        return FIELD_REGISTRY[path]
    except KeyError:
        raise ValueError(
            f"Unknown canonical path: {path!r}. Add it to the canonical "
            "field registry before mapping it."
        ) from None

def is_registered_path(path):
    return isinstance(path, str) and path in FIELD_REGISTRY
