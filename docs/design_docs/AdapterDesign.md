# Code Design: Canonical Data Adapter

**Status:** Implemented and verified
**Target:** `src/mt/data/_adapter.py`
**Date:** 2026-07-01

---

## Purpose

`_adapter.py` owns the dataset-level pipeline from a supported raw source to a
canonical `TrialCollection`:

```text
source
  -> load
  -> map
  -> defaults
  -> normalize scalar missing values
  -> optional coordinate filtering
  -> validate canonical rows
  -> assemble one row as one logical trial
  -> AdaptationResult
```

The module contains two layers:

1. Small pure functions implement defaults, missing-scalar normalization,
   filtering, validation, assembly, and collection construction. Each function
   accepts plain values, returns a new value, and is tested directly.
2. `DataAdapter` is a reusable one-shot facade. `adapt()` calls the pure
   components in their required order and returns an `AdaptationResult` after
   every stage succeeds. A stage error propagates immediately.

This follows the same boundary as `_mapping.py`: `ColumnMapping` is a thin
public facade over independently testable resolution and transformation
functions. `DataAdapter` does not turn the entire pipeline into one large
transform.

`_adapter.py` does not know about a model, model contract, split, fitted
encoding, tensor shape, optimizer, or training loop.

---

## Dependencies and Stage Inputs

The adapter composes the already-approved boundaries:

```python
from mt.data._collection import TrialCollection
from mt.data._field_registry import ...
from mt.data._loading import DataSource, load
from mt.data._mapping import ColumnMapping, MappingResolution
```

Loading owns source dispatch and raw column-label normalization. Mapping owns
raw-to-canonical resolution, pattern stacking, collision checks, and reused
source warnings. The adapter must not duplicate either implementation.

After the internal mapping stage, the staging DataFrame contains only full
canonical paths. Every later helper is raw-name blind.

---

## Public API

Export these names from `mt.data`:

```python
DataAdapter
AdaptationResult
```

The facade API is intentionally small:

```python
class DataAdapter:
    def __init__(self, mapping: ColumnMapping | None = None): ...

    def adapt(self, source: DataSource, *,
              filters=None) -> AdaptationResult: ...
```

`mapping=None` means `ColumnMapping()` and therefore exact identity lookup.
The immutable `ColumnMapping` instance is safe to retain.

The normal path is one call:

```python
adapter = DataAdapter(mapping)
result = adapter.adapt(
    source,
    filters={"participant_id": ["p01", "p02"]},
)
```

`filters=None` means no row filtering. The adapter always executes loading,
mapping, defaults, missing-scalar normalization, validation, and assembly.
Filtering runs between normalization and validation when criteria are
supplied.

The required stage order remains an internal correctness invariant. Users do
not manually repeat that fixed order through a mutable fluent state machine.
One adapter may safely adapt multiple sources because it retains only the
immutable mapping configuration, not a staging DataFrame.

---

## Pure Stage Functions

The module-level functions are low-level stage APIs. They are not exported
from `mt.data`, but tests and advanced callers may import them from the private
`mt.data._adapter` module:

```python
def apply_defaults(df: pd.DataFrame) -> pd.DataFrame: ...
def normalize_missing_values(df: pd.DataFrame) -> pd.DataFrame: ...
def filter_rows(df: pd.DataFrame,
                criteria: Mapping[str, object]) -> pd.DataFrame: ...
def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame: ...
def assemble_trials(df: pd.DataFrame) -> TrialCollection: ...
def build_collection(coordinates, slots) -> TrialCollection: ...
```

They do not accept a `DataAdapter`, mutate facade state, catch their own
errors, or produce reports. Direct calls use strict exceptions and validate
the assumptions observable at that function's boundary. They cannot prove
that another function ran previously without introducing wrapper types or
hidden state. For equal inputs they produce equal outputs without external
side effects.

`ColumnMapping.resolve()` and `ColumnMapping.apply()` remain the mapping-stage
operations; they are not copied into this module.

An advanced caller may inspect or resume the pipeline explicitly:

```python
raw = load(source)
mapped = mapping.apply(raw)
canonical = apply_defaults(mapped)
normalized = normalize_missing_values(canonical)
filtered = filter_rows(normalized, criteria)
validated = validate_dataframe(filtered)
collection = assemble_trials(validated)
```

This is intentional composability, not a second stateful adapter interface.
Each function fails on malformed input it can observe and never silently
repairs it. A direct caller owns cross-stage prerequisites that cannot be
inferred from an ordinary DataFrame.

---

## Defaults Stage

`apply_defaults()` consumes a mapped canonical DataFrame and returns a deep
DataFrame copy.

For each `FieldSpec` in registry order:

1. If the path is already a column, preserve every value exactly.
2. If the path is absent and `has_default=True`, add the declared default.
3. If the path is absent and has no default, leave it absent for validation.

Defaults apply to an absent field, never to a null cell. A mapped column that
contains `None`, `NaN`, or another accepted missing scalar is not filled.
Replacing trial-level missing values would be imputation and belongs to a
later model-specific adapter if a model needs it.

Coordinate defaults also apply only when the whole canonical path is absent.
For example, an absent `session_id` column becomes all `1`, but an existing
`session_id` column containing one missing cell is not filled. Per-cell
defaulting could silently merge an unknown session, block, task, or condition
into the documented degenerate case.

Current defaults add:

```text
session_id = 1
block_index = 1
task_name = 1
condition = 1
task.instructions = None
stimulus.ground_truth = None
stimulus.features = None
response.rt = None
outcome.reward = None
outcome.feedback = None
```

`participant_id`, `trial_index`, and `response.choice` have no defaults.

The output columns follow `CANONICAL_PATHS` order. Unknown columns and
duplicate labels are rejected because this stage consumes mapping output, not
raw data. Empty DataFrames receive the same canonical columns with zero rows.

Registry defaults are currently immutable. The implementation must still copy
a future mutable default independently per row rather than aliasing it across
trials.

---

## Missing-scalar Normalization

`normalize_missing_values()` consumes a fully defaulted canonical DataFrame
and returns an independent DataFrame in which every missing scalar uses the
single canonical value `None`.

Recognized missing scalars are:

- `None`;
- Python or NumPy floating-point `NaN`;
- `pandas.NA` and `pandas.NaT`;
- scalar NumPy datetime or timedelta `NaT` values.

Positive and negative infinity, empty strings, zero, and dataset-specific
codes such as `-999` are ordinary values. The adapter does not infer custom
sentinels.

Normalization applies to scalar DataFrame cells only. If a canonical cell is
a NumPy array, the array is left unchanged, including any `NaN` or `NaT`
inside it. Replacing array-internal values with `None` would change a numeric
array to object dtype and make a representational decision that belongs to
later preprocessing.

The result uses object dtype where pandas would otherwise coerce `None` back
to `NaN`. Non-missing values and row/column order are preserved. The function
does not delete rows, fill values, compute statistics, or mutate its input.

After normalization:

- any coordinate cell equal to `None` fails validation;
- content cells may carry `None` into `TrialCollection`;
- `response.choice` must exist as a path, but an individual trial may have
  value `None` until a model-side contract decides whether it is usable.

---

## Filtering Stage

`filter_rows()` performs optional inclusion filtering on canonical
coordinates. It never inspects slot content.

`DataAdapter.adapt(..., filters=...)` accepts a mapping from coordinate names
to criteria:

```python
adapter.adapt(
    source,
    filters={
        "participant_id": ["p01", "p02"],
        "condition": "test",
    },
)
```

Each criterion accepts one scalar or a one-dimensional collection of scalar
values. A string is one scalar, not an iterable of characters. Values within
one criterion use membership/OR semantics; multiple coordinates combine with
AND semantics.

Only names in `CANONICAL_COORDINATES` are accepted. An unknown path, content
path, missing coordinate column, malformed value collection, or non-scalar
criterion fails clearly. An empty value collection selects zero rows.

Filtering preserves source row order and pandas index values and returns an
independent DataFrame. Calling `filter_rows(df, {})` returns a copy unchanged.
Advanced callers may call `filter_rows()` repeatedly; repeated calls compose
sequentially. The one-shot facade accepts one combined criteria mapping.

The first implementation supports inclusion only. Exclusion expressions,
callables, arbitrary pandas queries, and content filtering are deferred until
a concrete use case requires them.

---

## Validation Stage

`validate_dataframe()` checks the fully defaulted, missing-normalized, and
optionally filtered canonical DataFrame and returns an independent copy on
success.

It validates:

1. The input is a DataFrame with unique column labels.
2. Every column is an exact registered canonical path.
3. Every path in `CANONICAL_PATHS` is present after defaults.
4. Every coordinate cell is scalar and non-missing.
5. Every content cell is a canonical scalar or a NumPy array; normalized
   `None` is the only missing scalar.
6. One-row trial identities are unique.

The stage does not coerce values, fill null content, sort rows, reset the
index, or enforce a model contract.

### Required paths and nulls

Registry `required=True` means that the dataset-level path must exist. Mapping
already fails early for a missing required source; validation repeats the
presence check so the helper is correct when called independently.

Coordinates are infrastructure addresses and therefore cannot be null on an
individual row. Content requirements remain dataset-level in this stage:
`response.choice` must be a column, but a null choice on one trial is not
silently dropped or imputed. A later model contract may reject it for a
specific model.

Missing scalars have already been normalized to `None`. Missing values inside
a NumPy array are not recursively changed.

### One-row trial identity

Stage one identifies a logical trial by:

```python
(
    "participant_id",
    "task_name",
    "session_id",
    "block_index",
    "trial_index",
)
```

Two rows with the same identity raise:

```text
Duplicate logical trial identity at row positions (...). Multi-row trials are
not supported by the one-row assembly strategy.
```

`condition` is a trial category, not part of identity. Filtering occurs before
validation by design, so only rows selected for adaptation participate in the
duplicate check.

Zero-row DataFrames are valid when all canonical columns are present.

---

## Assembly and Collection Construction

`assemble_trials()` converts a validated canonical DataFrame directly into a
`TrialCollection`. It does not group or aggregate rows: row position `i`
becomes logical trial `i`.

Assembly performs these mechanical operations:

- each coordinate column becomes a one-dimensional copied NumPy array;
- registered content paths are grouped by `FieldSpec.slot` and
  `FieldSpec.key`;
- each row becomes one dictionary in each of the five slots;
- NumPy scalar content values become the corresponding Python scalar;
- NumPy array content values are copied;
- row order is preserved and the pandas index is discarded.

`assemble_trials()` derives slot membership from `FIELD_REGISTRY`. It does not
use a hand-maintained `CONTENT_KEYS_BY_SLOT` constant.

`build_collection(coordinates, slots)` is the final small pure boundary. It
checks that the coordinate and slot mappings have exactly the registry names,
then passes their values to `TrialCollection`. It contains no DataFrame logic.
`assemble_trials()` calls `build_collection()` after extracting the two
runtime groups.

`DataAdapter.adapt()` places the resulting collection directly in
`AdaptationResult`. `TrialCollection` construction already establishes
independent ownership, and the adapter retains no collection after returning.

---

## One-shot Orchestration

`DataAdapter.adapt()` owns one local execution. It runs:

```text
load(source)
  -> mapping.resolve(raw) and mapping.apply(raw)
  -> apply_defaults(mapped)
  -> normalize_missing_values(canonical)
  -> filter_rows(normalized, filters) when filters are supplied
  -> validate_dataframe(filtered_or_normalized)
  -> assemble_trials(validated)
  -> AdaptationResult
```

Intermediate values and report metadata are local variables, not persistent
facade state. A stage executes only if the previous stage succeeded. The first
stage exception propagates immediately and stops the run.

The adapter does not expose `.load()`, `.map()`, `.defaults()`, `.filter()`,
`.validate()`, or `.assemble()` methods. Those names would create a mutable
state machine whose only valid path repeats an order already fixed by the data
contract.

One `DataAdapter` may be reused sequentially for multiple sources. Every
`adapt()` call has independent local state and returns independent runtime
data. Thread-safety requires no mutable adapter state beyond the immutable
`ColumnMapping`; no explicit concurrent-use guarantee is needed in stage one.

### Already-loaded DataFrames

A pandas DataFrame is a supported `DataSource`:

```python
result = DataAdapter(mapping).adapt(frame)
```

The internal `load(frame)` call is still meaningful: it copies caller-owned
data, normalizes column labels to strings, and rejects normalization
collisions. It performs no file I/O.

A caller that needs mapping only may use the low-level boundaries directly:

```python
mapped = mapping.apply(load(frame))
```

If `frame` is already the output of `mt.data.load()`,
`mapping.apply(frame)` is valid directly. The adapter does not accept a “skip
loading” flag because that would weaken its entry invariant for a small copy
optimization.

### Access and enforcement

Pipeline ordering is a correctness rule, not an access-control boundary.
Python callers can import names from a private module if they deliberately
choose to do so. Leading underscores and omitted top-level exports communicate
support level; they do not make functions inaccessible.

The design therefore does not hide pure helpers in closures or add branded
DataFrame wrapper types merely to prevent direct calls. Guarantees have two
levels:

- the normal `DataAdapter.adapt()` entry guarantees the complete order;
- a low-level stage validates only the local invariants it can observe and
  raises at the source of an invalid direct call.

Direct low-level composition is supported for inspection and research work,
but callers own the explicit ordering and therefore opt out of the facade's
complete-pipeline guarantee when they choose that route.

### Mapping execution

`adapt()` stores a local `MappingResolution` for reporting and calls
`ColumnMapping.apply()` for transformation. With the approved mapping API,
this means resolution validation runs once explicitly and once inside
`apply()`. The adapter does not duplicate mapping transformation logic merely
to avoid that small repeated check. Reused-source warnings are emitted only by
`apply()`.

---

## Failure Semantics

Pure stage functions and the previously implemented loading/mapping components
remain strict: they raise at the boundary that detects an invalid input.

`DataAdapter.adapt()` does not catch or convert stage exceptions. The first
error propagates with its original type and message and stops execution before
the next stage. An `AdaptationResult` exists only after successful assembly.

This keeps programming errors, reader failures, invalid mappings, malformed
filters, validation failures, and collection-construction failures visible at
their source. A future soft-validation mode may introduce a separate API; it
must not weaken the strict default behavior.

---

## `AdaptationResult`

`AdaptationResult` is an immutable top-level result object:

```python
@dataclass(frozen=True)
class AdaptationResult:
    collection: TrialCollection
    mapping: MappingResolution
    defaulted_paths: tuple[str, ...]
    input_rows: int
    output_rows: int

    def report(self) -> str: ...
```

A zero-trial collection is a successful result. There is no `complete` flag,
optional collection, stored exception, or `raise_for_error()` method because
the strict facade never constructs a failure result.

The top-level result is frozen, but its returned collection is ordinary owned
runtime data. Each `.adapt()` call receives an independent collection.

---

## Report

`result.report()` returns a deterministic human-readable string and never
prints. It includes successful adaptation metadata:

- input and output row counts;
- canonical targets and their raw mapping sources;
- fields supplied by registry defaults;
- ignored raw columns;
- reused raw columns;
- filter row reduction;
- the canonical scalar-missing representation.

Canonical fields use registry order. Raw columns use original source order as
already recorded by `MappingResolution`.

Reports do not include tracebacks or dump trial values. Failures are ordinary
exceptions and therefore do not produce an adaptation report.

---

## Ownership and Side Effects

- A caller-owned DataFrame is never mutated.
- Every pure DataFrame stage returns a new DataFrame.
- Mapping warnings remain the only expected stage side effect.
- Assembly and result construction copy coordinate arrays, slot dictionaries,
  and NumPy slot values.
- A `DataAdapter` may be reused for sources that share one mapping.
- Repeated `.adapt()` calls return independent results and collections.

---

## Module Structure

The file follows the project order:

```text
trial identity constant          private module metadata
AdaptationResult                public result object
DataAdapter                     public reusable one-shot facade
apply_defaults                  pure DataFrame transform
normalize_missing_values        pure scalar normalization transform
filter_rows                     pure DataFrame transform
validate_dataframe              pure DataFrame validator/copy
assemble_trials                 pure DataFrame-to-collection transform
build_collection                pure runtime-construction boundary
report formatting helpers       private deterministic rendering
value/filter/orchestration helpers private utilities
```

`AdaptationResult` appears before `DataAdapter` because the facade returns it.
No split, model, tensor, evaluation, training, CLI, or legacy preparation
module is imported.

---

## Tests

Create `tests/data/test_adapter.py`. Test pure functions directly before
testing one-shot facade orchestration.

### Defaults

1. Every absent registry default is added.
2. Existing columns and null cells are never overwritten.
3. Required fields without defaults remain absent for validation to reject.
4. Columns follow registry order.
5. Empty DataFrames receive zero-length default columns.
6. Input frames and future mutable defaults are not aliased.

### Missing-scalar normalization

7. `None`, Python/NumPy `NaN`, `pandas.NA`, and `NaT` become `None`.
8. Infinity, empty strings, zero, and custom numeric sentinels are preserved.
9. NumPy arrays and their internal missing values are unchanged.
10. Pandas does not coerce normalized `None` values back to `NaN`.
11. Input values, order, and array objects are not mutated.

### Filtering

12. Scalar, list, tuple, one-dimensional NumPy array, and Series criteria use
    inclusion semantics.
13. Set, dictionary, generator, and multidimensional criteria fail clearly.
14. Multiple coordinate criteria combine with AND semantics.
15. Row order and pandas index are preserved.
16. Empty criteria return an independent unchanged copy.
17. Empty value collections return zero rows.
18. Unknown, content, missing-coordinate, and malformed criteria fail clearly.
19. Repeated direct `filter_rows()` calls compose.

### Validation

20. Fully canonical one-row data validates without coercion.
21. Missing and unknown paths fail clearly.
22. Duplicate column labels fail clearly.
23. Coordinate values must be scalar and non-missing.
24. Content accepts canonical scalars, `None`, and NumPy arrays.
25. Unsupported content value types name path and row position.
26. Duplicate logical identities name the conflicting positions and the
    unsupported multi-row limitation.
27. `condition` differences do not make duplicate identities unique.
28. Empty canonical DataFrames validate successfully.

### Assembly

29. One DataFrame row becomes one logical trial.
30. Coordinates and all five slots preserve row alignment.
31. Slot keys are derived from registry paths.
32. NumPy scalar content normalizes to Python scalars.
33. Coordinate arrays, dictionaries, and NumPy slot values are copied.
34. `build_collection()` rejects missing or unknown coordinate/slot groups.
35. Assembly does not sort, group, aggregate, or preserve the pandas index.

### Facade and result

36. One `adapt()` call returns an `AdaptationResult` and `TrialCollection`.
37. DataFrame sources are copied and normalized by the internal loading stage.
38. Stages run in the documented order, including missing normalization.
39. One adapter can adapt multiple sources without retaining run state.
40. Loading, mapping, defaults, normalization, filtering, validation, and
    assembly errors propagate with their original exception type.
41. Stages after a failure do not execute and no result is constructed.
42. A zero-row adaptation succeeds.
43. Reports are deterministic and contain successful adaptation metadata.
44. Repeated `.adapt()` calls return independent results and collections.
45. Caller-owned DataFrames are never mutated.
46. DataAdapter and AdaptationResult are exported from `mt.data`.
47. No adapter component imports a model or model contract.
48. The facade exposes no mutable per-stage chaining methods.
49. Low-level stage functions remain directly testable with strict errors.

Tests use in-memory DataFrames. File-format behavior remains in
`test_loading.py`; pattern details remain in `test_mapping.py`; collection
selection details remain in `test_collection.py`.

---

## Out of Scope

- Heuristic raw-name inference
- Deleting missing trials or filling missing values
- Model-specific imputation
- Filtering on slot content or arbitrary expressions
- Row transforms unrelated to canonical adaptation
- Multi-row trial grouping or aggregation
- Assembly strategies in the stage-one public API
- Train/evaluation splitting
- Model contract validation
- Fitted encodings or tensor construction
- Cross-validation, batching, or streaming
- CLI and script migration

---

## Deferred Decisions

### Grouped multi-row trials

Stage two adds `_assembly.py` with one-row and grouped assembly strategies.
That change may move `assemble_trials()` behind a strategy object, but it must
not change mapping, defaults, filtering, validation of canonical paths,
`TrialCollection`, or `AdaptationResult`.

### Fitted missing-value treatment

Canonicalizing scalar missing sentinels to `None` is representation cleanup,
not statistical missing-value treatment. Stage one deliberately does not
offer strategy strings such as `"drop"` or `"mean"`.

Any strategy that learns a value, including mean imputation, must run after
train/evaluation splitting. It fits statistics on the training collection
only and applies the fitted state to both train and evaluation data. Computing
a mean inside `DataAdapter` would use the unsplit dataset and leak evaluation
information.

Deletion also requires a later explicit design. A caller must name the
canonical paths whose missing values trigger deletion; dropping every trial
with any missing optional field would remove valid datasets. Deleting a trial
can also change sequence history, so its placement and history semantics must
be stated rather than hidden behind a generic default.

The later design must cover:

- a fitted `fit(train)` / `transform(collection)` boundary;
- numeric versus categorical fields;
- scalar cells versus missing values inside NumPy arrays;
- the axis and grouping level for statistics, such as global, task, or
  participant;
- explicit paths for deletion and its effect on trial history;
- whether the implementation belongs to a reusable preprocessing transform or
  a model-specific `ModelAdapter`.

`TrialCollection` only carries canonical `None` values and unchanged arrays.
Loading, mapping, and collection construction never own imputation state.

### Rich machine-readable reports

The first report is deterministic text plus public result metadata. A separate
serializable report dataclass is added only if scripts or external callers
need a stable JSON schema.

### Top-level stage-function exports

Low-level stage functions remain importable from `mt.data._adapter` but are not
exported from `mt.data` in stage one. Promote one only when external callers
need a stable supported API for that specific intermediate representation.

---

## Implementation Result

The module is implemented in `src/mt/data/_adapter.py`, exported through
`mt.data`, and covered by `tests/data/test_adapter.py`.

The implementation provides a reusable one-shot `DataAdapter`, a success-only
`AdaptationResult`, independently testable pure stages, scalar missing-value
normalization, deterministic coordinate filtering, strict validation, and
one-row assembly into `TrialCollection`. Stage errors propagate without a
failure-result mode.

After verification, the superseded `_preparation.py`, `_prepared.py`,
`_requests.py`, and `_reports.py` modules and their public exports were removed
in a separate approved cleanup. Spec-bound view entrypoints and legacy tests
that depended on those modules were removed at the same boundary.

Focused adapter tests, the full repository test suite, scoped ruff, the
80-character convention check, and `git diff --check` pass.

---

## Acceptance Criteria

The module is ready when:

- every pipeline stage has one explicit responsibility;
- defaults, missing normalization, filters, validation, assembly, and
  construction are independently testable pure functions;
- registry metadata is applied rather than duplicated;
- strict component errors propagate without a failure-result mode;
- the public facade has one operation and no mutable stage state;
- one-row identity, null, filtering, and ownership rules are exact;
- no model, split, tensor, or legacy behavior enters the replacement API;
- the test plan covers every success, failure, and orchestration boundary.

The first-stage registry, loading, mapping, collection, and adapter modules are
now implemented. Grouped multi-row assembly remains a separately approved
second-stage task.
