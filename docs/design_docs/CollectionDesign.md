# Code Design: Canonical Trial Collection

**Status:** Implemented and verified
**Target:** `src/mt/data/_collection.py`
**Date:** 2026-07-01

---

## Purpose

`_collection.py` defines the runtime container for one adapted cognitive
dataset. `TrialCollection` holds actual trial values after raw columns have
been mapped, defaults have been applied, rows have been filtered, and the
canonical DataFrame has been validated.

The canonical field registry and `TrialCollection` have different jobs:

- `FIELD_REGISTRY` is static schema metadata. It declares which canonical
  paths exist and which rules apply to them.
- `TrialCollection` is one dataset instance. It owns coordinate arrays and
  per-trial slot values for a concrete number of logical trials.

The registry is consulted when a collection is built and checked. It does not
hold data, select trials, or copy runtime values.

The stage boundary is:

```text
validated canonical rows
  -> one row assembled as one logical trial
  -> TrialCollection
```

`_collection.py` does not load or map data, apply defaults, filter rows,
detect duplicate raw trial identities, group multi-row trials, split data, or
construct model tensors.

---

## Public Structure

The public class keeps coordinates and content slots as separate attributes:

```python
@dataclass
class TrialCollection:
    participant_id: np.ndarray
    session_id: np.ndarray
    block_index: np.ndarray
    trial_index: np.ndarray
    task_name: np.ndarray
    condition: np.ndarray

    task: list[dict[str, CanonicalValue]]
    context: list[dict[str, CanonicalValue]]
    stimulus: list[dict[str, CanonicalValue]]
    response: list[dict[str, CanonicalValue]]
    outcome: list[dict[str, CanonicalValue]]

    @property
    def n_trials(self) -> int: ...

    def __len__(self) -> int: ...
    def copy(self) -> TrialCollection: ...
    def select(self, selector) -> TrialCollection: ...
    def to_dataframe(self) -> pd.DataFrame: ...
```

Export `TrialCollection` from `mt.data`. Construction helpers and selector
normalization helpers remain private.

All eleven constructor fields are explicit. Canonical defaults do not belong
in the constructor; `_adapter.py` applies them before assembly. The normal
construction path is `_adapter.build_collection()`, although direct
construction remains supported and validates the same runtime invariants.

---

## Coordinate Invariants

The six coordinate attributes correspond exactly to
`CANONICAL_COORDINATES`, in registry order:

```text
participant_id
session_id
block_index
trial_index
task_name
condition
```

Every coordinate must be a one-dimensional `numpy.ndarray` with shape
`(n_trials,)`. Constructor inputs are copied so the collection never aliases a
caller-owned coordinate array.

All coordinate arrays must have the same length. `participant_id` determines
`n_trials`; zero trials are valid when every coordinate and slot is empty.

Coordinates contain scalar values, not lists, dictionaries, or nested arrays.
The adapter performs canonical null and duplicate-identity validation before
construction. The collection repeats structural value checks so direct
construction cannot create a malformed coordinate/content split, but it does
not apply defaults or infer missing coordinates.

---

## Slot Invariants

The five slot attributes correspond exactly to `CANONICAL_SLOTS`, in registry
order:

```text
task
context
stimulus
response
outcome
```

Each slot is a list with length `n_trials`. Each item is a dictionary holding
the registered keys for that slot. Registered keys are derived from
`FIELD_REGISTRY` through `FieldSpec.slot` and `FieldSpec.key`; there is no
second hand-maintained slot-key registry.

For the current registry, every row has this shape:

```python
task[i] = {"instructions": ...}
context[i] = {}
stimulus[i] = {"ground_truth": ..., "features": ...}
response[i] = {"choice": ..., "rt": ...}
outcome[i] = {"reward": ..., "feedback": ...}
```

Every registered key must be present, including optional keys whose value is
`None`. Unknown keys are rejected. `context` is present for every trial but
its dictionary is empty until the canonical registry defines a context key.

A slot value may be:

- `str`, `int`, `float`, `bool`, or `None`
- a NumPy scalar that can be normalized to the corresponding Python scalar
- a `numpy.ndarray` of any shape

Lists, tuples, dictionaries, pandas objects, tensors, and arbitrary class
instances are not canonical slot values. NumPy array values are copied.
Immutable scalar values may be shared safely.

The collection validates runtime value shape and type, not model semantics.
For example, it does not decide whether a particular model can consume a
two-dimensional `stimulus.features` value or whether a nullable response is
acceptable to that model.

---

## Ownership and Mutation

`TrialCollection` owns its stored data:

- coordinate arrays are copied at construction;
- each slot list is rebuilt;
- each per-trial dictionary is rebuilt;
- every NumPy array stored in a slot is copied;
- immutable scalar values may be reused.

The class is not advertised as deeply immutable because NumPy arrays and
dictionaries are mutable Python objects. Instead, every operation that returns
a collection returns independent storage. Mutating a returned copy or
selection must not mutate its source.

Arbitrary mutable Python objects are rejected rather than recursively copied.
This keeps the ownership guarantee exact and testable.

---

## `copy()`

```python
def copy(self) -> TrialCollection:
    ...
```

Returns an independent collection with the same trial order and values.
Coordinates, slot lists, slot dictionaries, and NumPy slot values are all
independent of the source.

`copy()` does not re-map, re-default, revalidate trial identity, or alter
types. Constructor structural checks may run again on the copied values.

---

## `select()`

```python
def select(self, selector) -> TrialCollection:
    ...
```

Selection is positional because a `TrialCollection` deliberately does not
retain a pandas index. Supported selectors are:

- a `slice`;
- a one-dimensional integer sequence or NumPy array;
- a one-dimensional boolean sequence or NumPy array of length `n_trials`.

Integer selection follows NumPy positional indexing, including negative
indices and explicit output ordering. Repeated positions are allowed; this
keeps selection usable for later resampling strategies. An empty sequence
selects zero trials.

Multidimensional selectors, mixed boolean/integer selectors, non-integral
positions, out-of-range positions, and boolean masks of the wrong length raise
clear errors.

Every coordinate and slot uses the same normalized positions. The returned
collection owns independent arrays and dictionaries, including copied NumPy
slot values.

---

## `to_dataframe()`

```python
def to_dataframe(self) -> pd.DataFrame:
    ...
```

`to_dataframe()` is a debugging and inspection tool. It flattens coordinates
to their canonical coordinate names and slot keys to full `slot.key` paths.
Columns follow `CANONICAL_PATHS` order and the result uses a new default
`RangeIndex`.

Array-valued cells are copied so editing the returned DataFrame cannot mutate
the collection. Raw column names, ignored columns, and the original pandas
index cannot be reconstructed and are not included.

No production pipeline component may convert a collection back to a
DataFrame merely to perform filtering, splitting, or model adaptation.

---

## Registry Relationship

`_collection.py` imports these registry declarations:

```python
CANONICAL_COORDINATES
CANONICAL_SLOTS
FIELD_REGISTRY
CanonicalValue
```

Slot keys are derived once in registry order at module import. The derived
mapping is private and read-only. It is a view of `FIELD_REGISTRY`, not a new
source of truth.

`TrialCollection` does not interpret `FieldSpec.required`,
`FieldSpec.has_default`, or `FieldSpec.allows_pattern`:

- required-path validation belongs to `_adapter.py`;
- default application belongs to `_adapter.py`;
- pattern behavior belongs to `_mapping.py`.

The collection only uses path ownership to validate its runtime shape.

---

## Failure Semantics

Direct construction fails at the collection boundary when runtime structure
is malformed:

- a coordinate is not a one-dimensional NumPy array;
- coordinate lengths disagree;
- a slot is not a list;
- a slot length differs from `n_trials`;
- a slot item is not a dictionary;
- a slot dictionary is missing a registered key;
- a slot dictionary contains an unknown key;
- a coordinate or slot contains a disallowed runtime value type.

Use `TypeError` when the received representation has the wrong kind and
`ValueError` when shapes, lengths, keys, or values violate a collection
invariant. Messages name the coordinate or slot and, for per-trial failures,
the trial position.

Missing raw columns, duplicate logical trial identities, and invalid mappings
are not collection failures. They are detected before construction by the
stage that owns them.

---

## Module Structure

```text
TrialCollection             public runtime container
derive slot keys            private registry-derived metadata
copy canonical value        private scalar/array ownership helper
validate coordinate         private one-dimensional shape/type check
validate slot               private length/key/value check
normalize selector          private positional-selection validation
```

No loading, mapping, filtering, splitting, model, tensor, or training import
belongs in this module.

---

## Tests

Create `tests/data/test_collection.py` with these behaviors:

### Construction and shape

1. A valid collection stores all six coordinates and five slots.
2. `n_trials` and `len(collection)` agree.
3. Empty collections are valid when every field is empty.
4. Coordinate inputs are one-dimensional NumPy arrays.
5. Coordinate lengths must agree.
6. Every slot must be a list with `n_trials` dictionaries.
7. All registered keys are required in their correct slot.
8. Unknown and cross-slot keys are rejected.
9. `context` is present and contains one empty dictionary per trial.
10. Scalar, NumPy-scalar, `None`, and NumPy-array values are accepted.
11. Lists, nested dictionaries, tensors, and arbitrary objects are rejected.

### Ownership

12. Constructor coordinate arrays do not alias caller arrays.
13. Constructor slot lists and dictionaries do not alias caller values.
14. Constructor NumPy slot values do not alias caller arrays.
15. `copy()` returns equal values in independent storage.
16. Mutating a copy does not mutate its source.

### Selection

17. Slice, integer, and boolean selection keep every field aligned.
18. Integer selection preserves requested order and permits repeats.
19. Empty selection returns a valid zero-trial collection.
20. Negative integer positions follow NumPy semantics.
21. Invalid selector shape, dtype, mask length, and bounds fail clearly.
22. Selected coordinates, dictionaries, and NumPy values are independent.

### Inspection and public API

23. `to_dataframe()` uses canonical paths in registry order.
24. Editing an inspection DataFrame does not mutate the collection.
25. `TrialCollection` is exported from `mt.data`.
26. Collection construction imports no model or model contract.

Default behavior is tested in `test_adapter.py`, not here. The collection
constructor receives already-complete canonical runtime fields.

---

## Out of Scope

- Raw source loading or column normalization
- Raw-to-canonical mapping
- Canonical default application
- Row filtering
- Duplicate logical-trial detection
- Multi-row trial grouping
- Participant or train/evaluation splitting
- Model-side required-field checks
- Padding, encoding, batching, or tensor conversion
- HuggingFace Dataset export
- Preserving the raw DataFrame index

---

## Implementation Result

The module is implemented in `src/mt/data/_collection.py`, exported through
`mt.data`, and covered by `tests/data/test_collection.py`.

The implementation derives slot keys from `FIELD_REGISTRY`, validates the
coordinate/slot runtime boundary, copies all owned mutable values, supports
positional copy-preserving selection, and exposes `to_dataframe()` only for
inspection.

Focused collection tests, the full repository test suite, scoped ruff, the
80-character convention check, and `git diff --check` pass.

---

## Acceptance Criteria

The module is ready when:

- the coordinate/slot representation and all runtime invariants are explicit;
- slot keys are derived from `FIELD_REGISTRY`, not duplicated manually;
- construction, copying, and selection have exact ownership guarantees;
- default and duplicate-identity behavior remain outside the collection;
- `to_dataframe()` is inspection-only;
- every failure category and public behavior has a focused test;
- no model-specific representation decision enters the class.

The next step is to review `AdapterDesign.md` before implementing
`_adapter.py`.
