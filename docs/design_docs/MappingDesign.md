# Code Design: Canonical Column Mapping

**Status:** Implemented and verified
**Target:** `src/mt/data/_mapping.py`
**Date:** 2026-07-01

---

## Purpose

`_mapping.py` translates raw DataFrame column names into canonical field paths.
It is the only stage that knows both raw column names and canonical names.

The stage boundary is:

```text
independent raw DataFrame
  + dataset-specific ColumnMapping
  -> resolve raw columns against the canonical field registry
  -> canonical-column DataFrame
```

Mapping is dataset-facing and model-independent. It does not inspect a model,
read a model contract, apply defaults, filter rows, validate trial values,
assemble trials, or construct a `TrialCollection`.

---

## Confirmed High-Level Rules

The detailed design preserves these decisions from `DataDesign.md`:

- Mapping runs immediately after loading and before defaults or filtering.
- Mapping targets must already exist in the canonical field registry.
- Coordinates use their canonical path as their identity raw name.
- Content fields use their exact canonical key as their identity raw name.
- Explicit fixed mappings override identity lookup for the same target.
- Regex patterns use `fullmatch`, never substring matching.
- A named numeric `index` capture determines numeric source ordering.
- Patterns without an `index` capture preserve raw DataFrame column order.
- Pattern values stack along the last axis.
- An explicitly declared pattern with no matches is an error.
- One raw column may feed multiple canonical targets, with a warning.
- Multi-column mapping is declared only through `patterns`.
- No heuristic name inference is allowed.

---

## Canonical and Raw Names

Mapping configuration is written as canonical target to raw source:

```python
mapping = ColumnMapping(
    mappings={
        "participant_id": "subject",
        "trial_index": "trial_no",
        "response.choice": "resp",
        "outcome.reward": "points",
    },
    patterns={
        "stimulus.features": r"feature_(?P<index>\d+)",
    },
)
```

Canonical output labels always use full registry paths:

| Field kind | Canonical output | Identity raw name |
|---|---|---|
| Coordinate | `participant_id` | `participant_id` |
| Coordinate | `trial_index` | `trial_index` |
| Content | `response.choice` | `choice` |
| Content | `outcome.reward` | `reward` |
| Content | `task.instructions` | `instructions` |

A full content path such as `response.choice` is an output label, not the
implicit raw identity name. A raw DataFrame that already uses full content
paths can map them explicitly:

```python
ColumnMapping(mappings={"response.choice": "response.choice"})
```

This keeps identity lookup aligned with the field registry's `key` property
and avoids treating dots in arbitrary raw names as canonical declarations.

---

## Public API

The module exposes two public immutable classes:

```python
@dataclass(frozen=True)
class MappingResolution:
    sources: Mapping[str, tuple[str, ...]]
    pattern_targets: tuple[str, ...]
    ignored_columns: tuple[str, ...]
    reused_columns: Mapping[str, tuple[str, ...]]

@dataclass(frozen=True)
class ColumnMapping:
    mappings: Mapping[str, str] = field(default_factory=dict)
    patterns: Mapping[str, str] = field(default_factory=dict)

    def resolve(self, df: pd.DataFrame) -> MappingResolution: ...
    def apply(self, df: pd.DataFrame) -> pd.DataFrame: ...
```

Omitted mappings and patterns default to empty immutable mappings:

```python
mapping = ColumnMapping()
mapping = ColumnMapping(mappings={"response.choice": "resp"})
mapping = ColumnMapping(patterns={"stimulus.features": r"x_\d+"})
```

Export `ColumnMapping` and `MappingResolution` from `mt.data`. Internal regex
objects and transformation helpers are not exported.

---

## Immutability

The constructor copies caller-provided mappings and patterns. Public mappings
are exposed through `MappingProxyType`, so later mutation of either the input
dictionary or the `ColumnMapping` instance cannot change behavior.

`MappingResolution` also copies and freezes its mapping fields. Tuple and
mapping insertion order are authoritative and deterministic.

Compiled regex objects are private implementation state. They are constructed
once when `ColumnMapping` is created and cannot be replaced by callers.

---

## Constructor Validation

Construction validates configuration that does not require a DataFrame:

1. Every target is a string and exists in `FIELD_REGISTRY`.
2. Every fixed raw source is a non-empty string.
3. Every regex expression is a non-empty string and compiles successfully.
4. A target cannot appear in both `mappings` and `patterns`.
5. Every pattern target has `FieldSpec.allows_pattern=True`.

Unknown targets delegate to `get_field_spec()` so registry errors remain the
single authoritative wording for unknown canonical paths.

A fixed mapping target may refer to any registered coordinate or content
field. A pattern cannot target a coordinate or a field whose registry entry
disallows patterns.

---

## Raw DataFrame Preconditions

`resolve()` and `apply()` consume the DataFrame produced by `load()`. Loading
has already converted every column label to a string, converted missing labels
to `"None"`, and rejected collisions created by that normalization.

Mapping verifies that column names remain unique but does not normalize them.
A caller using `ColumnMapping` directly should first pass its DataFrame through
`load()` so the same boundary invariant applies. Column-label conversion must
not be duplicated in mapping.

The DataFrame may have zero rows, a non-default index, null values, object
dtype columns, or columns unrelated to the canonical registry. Those are not
mapping errors.

---

## Resolution Algorithm

Resolution iterates `CANONICAL_PATHS` in registry order. For each canonical
target, exactly one of these routes is selected:

1. **Explicit pattern:** if the target appears in `patterns`, resolve all
   matching raw columns.
2. **Explicit fixed mapping:** if the target appears in `mappings`, resolve the
   named raw column.
3. **Identity lookup:** otherwise, look for the field's identity raw name.
4. **Unresolved optional field:** omit an optional target that has no source.
5. **Unresolved required field:** raise a missing-required mapping error.

The constructor rejects a target appearing in both explicit routes, so route
precedence never resolves an ambiguous user declaration.

Explicit fixed mappings and patterns must resolve successfully even when the
target is optional. A declaration is a user assertion; silently falling back
to identity or a default would hide a misspelled raw name or pattern.

Defaults are not applied by resolution. For example, an absent `session_id`
is omitted here and added later by the defaults stage.

---

## `MappingResolution`

`resolve(df)` returns an immutable, inspectable description without modifying
the DataFrame and without emitting warnings.

### `sources`

Maps each resolved canonical target to one or more raw source names:

```python
{
    "participant_id": ("subject",),
    "trial_index": ("trial_no",),
    "stimulus.features": ("feature_1", "feature_2"),
    "response.choice": ("resp",),
}
```

Targets use registry order. Sources for a pattern use resolved pattern order.

### `pattern_targets`

Contains the resolved targets that produce numpy-array values. It uses registry
order and distinguishes a one-column pattern from a fixed mapping.

### `ignored_columns`

Contains raw columns not used by any resolved target, preserving original raw
column order. The later `DataAdapter` report can use this field without asking
mapping logic to run again.

### `reused_columns`

Maps every raw column used by more than one canonical target to those targets.
Raw columns use original DataFrame order; target tuples use registry order.
Reusing a source is permitted and is not represented as an error.

---

## Fixed Mapping Behavior

A fixed mapping copies one raw Series to one full canonical output path.

```text
raw column "resp" -> output column "response.choice"
```

An explicit fixed mapping replaces identity lookup for that target. If both
`resp` and the identity raw name `choice` exist, only `resp` supplies
`response.choice`; `choice` is ignored unless another target uses it.

Fixed mappings preserve values and dtype as far as pandas permits. They do not
coerce strings, numeric values, booleans, nulls, or object-dtype cells.

---

## Pattern Matching

Patterns are compiled with Python's `re` module and applied to every raw column
using `fullmatch`.

For example:

```python
patterns={
    "stimulus.features": r"feature_(?P<index>\d+)",
}
```

matches `feature_1` and `feature_10`, but not `prefix_feature_1` or
`feature_1_suffix`.

### Source ordering

If the regex defines a named `index` capture:

- Every match must produce a non-empty capture accepted by `int()`.
- Matches sort by the resulting integer in ascending order.
- Equal numeric indices retain their original DataFrame order.

This makes `feature_2` sort before `feature_10`. Other capture groups do not
affect ordering.

If no named `index` capture exists, matches preserve raw DataFrame order.

### Row values

For every row, values from the ordered matched columns are converted with
`numpy.asarray()` and combined with `numpy.stack(..., axis=-1)`.

- Scalar source cells produce shape `(n_matches,)`.
- Array source cells stack along their last axis.
- A one-column pattern still produces an array with a final axis of length 1.
- Every row receives a newly created numpy array.
- An empty DataFrame produces an empty object-dtype output Series without
  attempting row stacking.

If matched values in one row have incompatible shapes, mapping raises a
`ValueError` naming the target and row position. Mapping never sums, averages,
flattens, pads, or coerces incompatible shapes.

---

## Reuse and Collision Rules

### Reusing one raw column

One raw source may feed multiple canonical targets. `apply()` emits one
`UserWarning` per reused raw column and copies the values to every target.
The warning names the raw source and all affected canonical targets.

Overlap between a fixed mapping and a pattern, or between two patterns, follows
the same rule. Multiple matches within one pattern target are intentional and
do not count as reuse.

Warnings are emitted in raw DataFrame column order. `resolve()` records reuse
but does not warn, keeping inspection side-effect free.

### Multiple sources for one target

The fixed `mappings` API accepts exactly one raw source per target. A target
cannot also have a pattern. Attempting to declare both routes raises:

```text
Multiple columns target one field; use only a pattern for multi-column mapping.
```

Multi-column values must use `patterns`, and only registry-approved pattern
targets can receive them.

### Ignored columns

Unused raw columns are recorded in `MappingResolution.ignored_columns` but are
not copied into the mapped DataFrame. The `DataAdapter` retains the raw frame
for reporting, while every stage after mapping receives canonical labels only.

---

## Output Invariants

`apply(df)` returns a new DataFrame with these guarantees:

1. The input DataFrame is not mutated.
2. The output index equals the input index.
3. Every output column is an exact registered canonical path.
4. Output columns follow `CANONICAL_PATHS` registry order.
5. Only successfully resolved fields are present.
6. Optional unresolved fields are absent until the defaults stage.
7. Ignored raw columns and consumed raw names are absent.
8. Fixed values are copied without mapping-time coercion.
9. Pattern cells contain fresh numpy arrays stacked along the last axis.

Pandas deep-copy semantics apply to fixed object-dtype cells: arbitrary nested
Python objects are not recursively cloned. Later `TrialCollection`
construction remains responsible for copying numpy-array slot values.

---

## Failure Semantics

Configuration errors fail during `ColumnMapping` construction. DataFrame and
resolution errors fail from `resolve()` and therefore also from `apply()`.

### Configuration failures

- Unknown canonical target: `ValueError` from `get_field_spec()`.
- Empty or non-string raw source: `ValueError` naming the target.
- Empty or invalid regex: `ValueError` naming the target and expression.
- Pattern on a disallowed field: `ValueError` naming the target.
- Fixed and pattern declarations for one target: `ValueError` with the
  multi-column mapping guidance.

### DataFrame failures

- Non-DataFrame input: `TypeError` naming the received type.
- Duplicate raw column labels: `ValueError` naming them.
- Explicit fixed source absent: `ValueError` naming source and target.
- Required identity source absent: `ValueError` naming the canonical target
  and expected raw identity name.
- Explicit pattern matches nothing: `ValueError` naming target and pattern.
- Non-numeric `index` capture: `ValueError` naming target, column, and capture.
- Incompatible pattern cell shapes: `ValueError` naming target and row.

Missing optional identity fields are not errors. Null values and semantic value
types are validated later, not during mapping.

The future `DataAdapter` may catch mapping errors and include them in an
`AdaptationResult`; that facade behavior does not change mapping's local error
types.

---

## Module Structure

The module keeps public declarations small and resolution helpers private:

```text
MappingResolution          immutable inspection result
ColumnMapping              immutable mapping configuration and public methods
validate dataframe columns private entry validation
resolve fixed source       private exact-name resolution
resolve pattern            private fullmatch and ordering
find reuse                 private deterministic collision analysis
stack pattern values       private row-wise numpy construction
```

No model, model contract, `TrialCollection`, or `DataAdapter` import belongs in
this module. Registry access is through `FIELD_REGISTRY`, `CANONICAL_PATHS`, and
`get_field_spec()`.

---

## Public Exports

Export from `mt.data`:

```python
ColumnMapping
MappingResolution
```

Do not export compiled regex state or private resolution helpers.

---

## Tests

Create `tests/data/test_mapping.py` with these behaviors:

### Construction and registry validation

1. Empty construction supports identity-only mapping.
2. Caller dictionaries are copied and public mappings are read-only.
3. Unknown canonical targets fail through `get_field_spec()`.
4. Empty raw source names and regex expressions fail clearly.
5. Invalid regex syntax names its canonical target.
6. Patterns are rejected for coordinates and non-pattern content fields.
7. One target cannot appear in both fixed mappings and patterns.

### Fixed and identity mapping

8. Explicit coordinate and content mappings use full canonical output paths.
9. Unmapped coordinates use exact coordinate-name identity lookup.
10. Unmapped content uses exact key identity lookup.
11. A full content path is not an implicit raw identity name.
12. Explicit mapping overrides an available identity source.
13. Missing explicit sources fail even for optional targets.
14. Missing required identity sources fail with target and expected raw name.
15. Missing optional identity sources are omitted.
16. Defaults are not applied by mapping.

### Resolution and output

17. Resolution and output targets follow registry order.
18. Ignored columns preserve raw order in `MappingResolution` and are absent
    from output.
19. Output preserves the input index and does not mutate input.
20. Duplicate raw labels fail at mapping entry; loading owns string
    normalization.
21. Empty DataFrames map successfully when required columns exist.

### Patterns

22. Regex matching uses `fullmatch`.
23. Numeric `index` captures sort numerically, not lexicographically.
24. Equal numeric indices preserve source order.
25. Patterns without `index` preserve raw column order.
26. Non-numeric or missing `index` captures fail clearly.
27. An explicit pattern with no matches fails clearly.
28. A one-column pattern produces arrays with final size one.
29. Scalar pattern values stack to `(n_matches,)` per row.
30. Array-valued cells stack along the last axis.
31. Pattern arrays are independent between rows and from source values.
32. Incompatible row shapes fail with target and row context.
33. Empty DataFrames produce a valid empty pattern column.

### Reuse and independence

34. One raw column reused by fixed targets warns once and copies values.
35. Fixed-pattern and pattern-pattern overlap warn deterministically.
36. Multiple columns for one target require a pattern.
37. Resolution records reused sources without emitting warnings.
38. Mapping construction and application require no model or model contract.
39. `ColumnMapping` and `MappingResolution` are exported from `mt.data`.

Warnings are tested with `pytest.warns`. Tests use in-memory DataFrames and do
not require files, network access, models, or later adapter modules.

---

## Out of Scope

- Loading files or HuggingFace datasets
- Adding canonical defaults
- Filtering rows
- Validating nulls, value types, or trial identity
- Trial assembly or `TrialCollection` construction
- Model contracts or model-specific requirements
- Heuristic aliases, fuzzy matching, or case-insensitive matching
- Renaming raw values or coercing dtypes
- Arbitrary transform callables in mapping configuration
- List-valued fixed mapping sources
- Summing, averaging, padding, or flattening pattern values
- Persisting or fitting mapping state
- Formatting the final `AdaptationResult` report

---

## Approved Detailed Decisions

The implementation follows these approved decisions:

1. `apply()` returns only resolved canonical columns; ignored raw columns live
   in `MappingResolution`, not in the mapped frame.
2. Missing registry-required fields fail during mapping resolution, while
   optional missing fields wait for the defaults stage.
3. Loading normalizes raw column labels to strings and rejects resulting
   collisions; mapping only verifies uniqueness and never renames raw labels.
4. `MappingResolution` is a public immutable object so the later adapter can
   build reports without duplicating mapping logic.
5. Pattern stacking is row-wise `np.stack(..., axis=-1)`, including array-valued
   raw cells and one-column patterns.

---

## Implementation Result

The module is implemented in `src/mt/data/_mapping.py`, exported through
`mt.data`, and covered by `tests/data/test_mapping.py`.

The implementation includes immutable mapping configuration and resolution,
registry-order identity and explicit resolution, full-match regex ordering,
row-wise last-axis stacking, deterministic ignored/reused-column inspection,
and one warning per reused raw source.

Focused mapping tests, all data-layer tests, and the full repository test suite
pass. Scoped ruff and formatting checks also pass.

---

## Acceptance Criteria

The module is ready for review when:

- The public API and five detailed decisions above are implemented.
- Every mapping target is registry-validated and model-independent.
- Identity, explicit fixed, and pattern routes have unambiguous precedence.
- Resolution order, output order, warnings, and errors are deterministic.
- Mapping never mutates caller-owned data or applies defaults.
- Pattern behavior is fully specified for ordering, shape, and empty data.
- The test list covers every public behavior and failure category.
- Focused mapping tests, the full test suite, and scoped lint pass.
