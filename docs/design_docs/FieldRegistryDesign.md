# Code Design: Canonical Field Registry

**Status:** Implemented and verified
**Target:** `src/mt/data/_field_registry.py`
**Date:** 2026-07-01

---

## Purpose

`_field_registry.py` is the single source of truth for canonical coordinate
names, content slots, content paths, required fields, defaults, and pattern
targets.

It does not inspect datasets, apply mappings, construct `TrialCollection`, or
know which fields a model requires. Those responsibilities belong to later
modules.

---

## Registered Fields

### Coordinates

| Path | Required | Default |
|---|---|---|
| `participant_id` | yes | none |
| `session_id` | no | `1` |
| `block_index` | no | `1` |
| `trial_index` | yes | none |
| `task_name` | no | `1` |
| `condition` | no | `1` |

### Content

| Path | Required | Default | Pattern target |
|---|---|---|---|
| `task.instructions` | no | `None` | yes |
| `stimulus.ground_truth` | no | `None` | yes |
| `stimulus.features` | no | `None` | yes |
| `response.choice` | yes | none | yes |
| `response.rt` | no | `None` | no |
| `outcome.reward` | no | `None` | no |
| `outcome.feedback` | no | `None` | yes |

The canonical slots are `task`, `context`, `stimulus`, `response`, and
`outcome`. `context` is a valid empty slot but currently has no canonical key.
A mapping cannot target an undeclared path such as `context.cue`.

`stimulus` and `context` may both be empty for datasets that do not expose an
input field, such as some reinforcement-learning or bandit datasets. A model
that needs `stimulus.ground_truth` declares that requirement in its model-side
contract.

New paths are added to this file only when an implemented model needs them.
Names must be generic enough to be reused across paradigms.

---

## Module Responsibilities

`_field_registry.py` owns:

- Ordered canonical slots and fields
- Required/default metadata
- Whether a field accepts regex pattern mapping
- Exact canonical-path lookup
- Validation of the field-registry definition itself

It does not own:

- Raw column lookup or renaming
- Regex matching, ordering, or stacking
- DataFrame validation
- Per-trial value validation
- Default application
- Model contracts

The registry describes rules. `_mapping.py`, `_collection.py`, and
`_adapter.py` apply them later.

---

## Value Types

The module declares type aliases for documentation and downstream type hints:

```python
CanonicalScalar = str | int | float | bool | None
CanonicalValue = CanonicalScalar | np.ndarray
```

Runtime validation of actual slot values belongs to `_collection.py`, not this
module. Registry defaults are currently only `1` and `None`; no mutable numpy
array is stored as a default.

---

## Data Structure

Use one immutable dataclass rather than parallel dictionaries that can drift
out of sync.

```python
@dataclass(frozen=True)
class FieldSpec:
    path: str
    description: str
    required: bool = False
    has_default: bool = False
    default: CanonicalValue = None
    allows_pattern: bool = False

    @property
    def is_coordinate(self): ...

    @property
    def slot(self) -> str | None: ...

    @property
    def key(self): ...
```

`path` is the identity:

- Coordinates use a bare key, such as `participant_id`.
- Content fields use exactly `slot.key`, such as `response.choice`.
- Keys cannot contain another dot.

Constructor invariants:

- `path` is non-empty.
- `description` is non-empty and states the field's canonical meaning.
- A bare path belongs to `CANONICAL_COORDINATES`.
- A content path contains exactly one dot.
- Its slot belongs to `CANONICAL_SLOTS`.
- `required=True` cannot be combined with `has_default=True`.
- `allows_pattern=True` is valid only for a content field.

`has_default` is separate from `default` so the design can distinguish
"no default" from "the default is `None`".

---

## Constants

Declare fields once in deterministic order:

```python
CANONICAL_COORDINATES: tuple[str, ...]
CANONICAL_SLOTS: tuple[str, ...]
FIELD_REGISTRY: Mapping[str, FieldSpec]
CANONICAL_PATHS: tuple[str, ...]
REQUIRED_PATHS: tuple[str, ...]
```

`FIELD_REGISTRY` is wrapped with `MappingProxyType`, preserves declaration
order, and cannot be mutated by callers. Construction fails immediately if two
field specifications have the same path.

Tuples and registry insertion order preserve report and test order. Sets may be
derived locally for fast membership checks but are not authoritative.

---

## Functions

### `get_field_spec`

```python
def get_field_spec(path) -> FieldSpec:
    ...
```

Returns metadata for an exact canonical path.

Raises:

```text
ValueError: Unknown canonical path: 'context.cue'. Add it to the canonical
field registry before mapping it.
```

This is the function `_mapping.py` will use to validate mapping targets.

### `is_registered_path`

```python
def is_registered_path(path):
    ...
```

Returns whether `path` is an exact coordinate or content path. It never raises
for an unknown string.

No DataFrame-oriented validation function belongs in this module.

---

## Public API

Export these names from `mt.data`:

```python
FieldSpec
CANONICAL_COORDINATES
CANONICAL_SLOTS
FIELD_REGISTRY
CANONICAL_PATHS
REQUIRED_PATHS
get_field_spec
is_registered_path
```

Do not export `_FIELD_SPECS` or registry-construction helpers.

---

## Failure Semantics

There are two failure categories:

1. An invalid field declaration is a developer error and fails at module
   import with `RuntimeError`, naming the duplicate or invalid declaration.
2. An unknown path supplied by another component or user raises `ValueError`
   from `get_field_spec()`, naming that path.

Missing DataFrame columns are not registry errors. `_mapping.py` handles
them later.

---

## Tests

Create `tests/data/test_field_registry.py` with these behaviors:

1. Coordinates and slots have the exact confirmed order.
2. All coordinate and content paths are present in deterministic order.
3. Required paths are exactly `participant_id`, `trial_index`, and
   `response.choice`.
4. Coordinate defaults are correct.
5. Optional content defaults, including `stimulus.ground_truth`, are `None`.
6. Pattern mapping is allowed for `task.instructions`,
   `stimulus.ground_truth`, `stimulus.features`, `response.choice`, and
   `outcome.feedback`, but not for `response.rt` or `outcome.reward`.
7. `context` exists as a slot but has no field.
8. `get_field_spec()` returns the declared immutable object.
9. Unknown and malformed paths raise an error naming the path.
10. `is_registered_path()` returns `True` only for exact declared paths.
11. Declared paths are unique; index construction cannot silently overwrite.
12. A frozen `FieldSpec` cannot be mutated.
13. The public registry is read-only and preserves declaration order.
14. Every field has a non-empty semantic description.

No pandas fixture is needed. These are pure metadata tests.

---

## Implementation Result

The module is implemented in `src/mt/data/_field_registry.py`, exported through
`mt.data`, and covered by `tests/data/test_field_registry.py`. Focused tests
and lint pass, and the full test suite passes.

---

## Acceptance Criteria

The module is ready when:

- The public constants and functions above exist.
- No mapping or DataFrame logic has entered the module.
- All focused field-registry tests pass.
- `ruff check` passes for the module and test file.
- Existing unrelated behavior has not been changed.

After that, stop and review the result before designing `_loading.py`.
