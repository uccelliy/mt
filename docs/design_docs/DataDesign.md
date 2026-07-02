# Design Doc: Canonical Data and Adapter System

**Status:** Proposed
**Author:** Ruochen Yin
**Date:** 2026-06-30

---

## Problem

Cognitive tasks are structurally different from each other. Their datasets
use different column names, different formats, and different conventions.
Cognitive models also need different inputs. There is currently no standard
way to connect a dataset to a model — every combination requires custom
preprocessing code.

This creates three compounding problems:

- Adding a new model requires writing new preprocessing from scratch
- Adding a new dataset requires understanding every model's input format
- Comparing models across datasets is fragile and not reproducible

---

## Goal

A system where:

1. The **canonical field registry** defines shared slot and field names
2. The **user declares** how their raw dataset maps to canonical field paths
   without selecting a model
3. The **DataAdapter** produces a `TrialCollection` — raises immediately at
   the stage that detects invalid data
4. The **model declares** which canonical field paths it needs in a
   model-side contract
5. The **ModelAdapter** fits on training data, transforms `TrialCollection`
   → tensors for both splits
6. The **Model** only sees tensors and computes logits

No model ever touches raw data. No dataset ever needs to know what model
will consume it. No eval statistics leak into training.

---

## Inspiration

**HuggingFace:** tokenizer and model are separate objects. The tokenizer
fails early if input is wrong before the model ever sees the data. The
tokenizer vocabulary is fitted on training data only.

**Tidymodels:** recipe and model are separate objects. The recipe is
composable — built from explicit ordered steps. It is fitted on training
data and applied to test data. The model is never aware of preprocessing.

The key shared insight: **the transformation boundary must be explicit,
independently testable, and fail early. Fitting always happens on training
data only.**

---

## Canonical Field Registry

| Slot | Key | Description |
|---|---|---|
| `task` | `instructions` | Task instructions; `None` if absent |
| `context` | *(none yet)* | Reserved for future generic keys; empty for now |
| `stimulus` | `ground_truth` | Correct label; `None` if absent |
| `stimulus` | `features` | Generic variable-length feature vector |
| `response` | `choice` | Participant choice; required key |
| `response` | `rt` | Response time in milliseconds; `None` if absent |
| `outcome` | `reward` | Objective consequence; `None` if absent |
| `outcome` | `feedback` | Presented feedback; `None` if absent |

Pattern mapping is allowed for `task.instructions`,
`stimulus.ground_truth`, `stimulus.features`, `response.choice`, and
`outcome.feedback`. It is not allowed for `response.rt` or `outcome.reward`.

The five content slots are `task`, `context`, `stimulus`, `response`, and
`outcome`. A canonical content field is identified by its full `slot.key`
path, for example `response.choice` or `stimulus.features`. Key names do
not need to be globally unique across slots.

The canonical field registry is data-side and model-independent. There is no
data-side contract derived from a model. Each model has a separate model-side
contract that refers to canonical field paths. No raw column name is ever
used directly as a slot key — every key in the registry table above is the
complete set of names a `ColumnMapping` may target.

**Growing the registry:** new keys are added only when an actual model
needs them, using a generic name that does not presuppose any one paradigm
or theory — e.g. `features` rather than `gcm_stimulus`, so the same key can
later be reused by an unrelated model on an unrelated dataset (for example,
mapping an N-back stimulus stream into `stimulus.features` so
Generalized Context Model can be fit against it, even without theoretical
support for that pairing). Do not pre-add keys for models or paradigms that do
not exist yet.

### Dataset-level registry rules

Vocabulary validation checks keys across the whole dataset, not whether every
individual trial has a non-null value.

- `participant_id`, `trial_index`, and `response.choice` must exist after
  mapping.
- All other fixed content keys are optional at the registry level; whether
  a given model requires them is determined by that model's own contract,
  not by the canonical field registry.
- `context` currently has no defined keys and contributes nothing to any
  `TrialCollection` until a key is added here.
- `stimulus` and `context` may both be empty. Models that require a stimulus
  field, including `ground_truth`, declare it in their model-side contract.

---

## TrialCollection — Principle

The DataAdapter does not produce tensors or a flat DataFrame. It produces
a `TrialCollection` — a typed class that holds validated trial data in a
form that any model, including deep neural networks, can consume without
the canonical structure having made representational decisions on its behalf.
Detailed runtime invariants are specified in
`docs/design_docs/CollectionDesign.md`.

A flat DataFrame is a representational commitment: it says stimulus is a
fixed set of named scalars. `TrialCollection` defers that decision to each
model's adapter. The canonical structure is an interface, not a
representation.

**Why not HuggingFace Datasets:** HF Datasets solves the storage problem
(mixed scalars and nested features, Arrow-backed, efficient batching) and
is the right export target for foundation model training at scale.
It is the wrong internal representation for this pipeline because it has
no concept of the canonical field registry, participant-aware splitting, or
`ModelAdapter.fit()`. HF Datasets export is a future addition, not a
replacement for `TrialCollection`.

### Structure

`TrialCollection` has two fundamentally different kinds of fields:

**Coordinates** — scalar numpy arrays, shape `(n_trials,)`. Used for
splitting and filtering before any model is involved. These are addresses,
not content. Making them scalar is a database design decision, not a
modeling assumption.

**Content slots** — lists of dicts, length `n_trials`. Each dict holds
canonical keys within one of the five canonical slots. Slot values may be
strings, integers, floats, booleans, `None`, or numpy arrays. The canonical
structure makes no claim about how values should be represented for a model —
that decision belongs to its model-specific adapter.

```python
@dataclass
class TrialCollection:
    # Coordinates — scalar, universal, used by split() and filter()
    participant_id: np.ndarray    # shape (n_trials,)
    session_id: np.ndarray
    block_index: np.ndarray
    trial_index: np.ndarray
    task_name: np.ndarray         # routing key — which task applies
    condition: np.ndarray         # trial category; default 1 when absent

    # Content slots — list of dicts using canonical keys
    task: list[dict]        # instructions text only — no routing fields
    context: list[dict]     # environmental and situational framing
    stimulus: list[dict]    # direct object presented to the participant
    response: list[dict]    # participant behavior
    outcome: list[dict]     # consequences and feedback
```

**Why `task_name` and `condition` are coordinates, not slot content:**
the pipeline needs both before any model is involved — `task_name` for
task routing, `condition` for filtering practice or forced-choice
trials. Infrastructure must never unpack slot content to do routing.
Coordinates are addresses; slot content is model input.

**Coordinate presence rules:**

| Coordinate | If absent in raw data |
|---|---|
| `participant_id` | Assert — raise `ValueError`. No default. |
| `trial_index` | Assert — raise `ValueError`. No default. |
| `session_id` | Default `1` — "single session" |
| `block_index` | Default `1` — "single block" |
| `condition` | Default `1` — "one condition, undivided" |
| `task_name` | Default `1` — "unnamed single task" |

These are canonical adaptation rules. `_adapter.apply_defaults()` applies
them before assembly; the `TrialCollection` constructor receives all six
coordinates explicitly and does not perform DataFrame-level defaulting.

`participant_id` and `trial_index` are the minimum identity required to
detect leakage and preserve trial order. Without them the pipeline cannot
make correctness guarantees, so they are hard requirements. All other
coordinates have a natural degenerate-case default that requires no
special-casing downstream.

**`task` slot — instructions only:** the task slot key is `"instructions"`.
It holds the natural language instructions shown to participants as a
string. When a dataset has no instructions column, the key is present with
value `None` — the slot is never absent, only its value is null.
Classical cognitive models may omit this path from their model contract.
Foundation and language models can consume `"instructions"` as a free-text
input with no taxonomy imposed.

**Slot key convention:** raw names are translated by `ColumnMapping` to full
canonical `slot.key` paths before slot construction. A supplied mapping first
renames the raw key to its canonical key; every later pipeline step operates
only on canonical names. `ColumnMapping` does not read a model contract. Model
contracts later refer to the same canonical paths.

```
Raw Dataset                  TrialCollection
  "resp"       ──→           response[i]["choice"]
  "points"     ──→           outcome[i]["reward"]
  "label"      ──→           stimulus[i]["ground_truth"]
  "subject_id" ──→           participant_id[i]       (required — assert)
  "trial_no"   ──→           trial_index[i]          (required — assert)
  "cond"       ──→           condition[i]            (coordinate)
  "task"       ──→           task_name[i]            (coordinate)
  (absent)     ──→           session_id[i]   = 1     (default)
  (absent)     ──→           block_index[i]  = 1     (default)
  (absent)     ──→           condition[i]    = 1     (default)
  (absent)     ──→           task_name[i]    = 1     (default)
  (absent)     ──→           task[i]["instructions"] = None  (default)
```

The first implementation supports exactly one raw row per logical trial. The
combination of `participant_id`, `task_name`, `session_id`, `block_index`, and
`trial_index` must therefore be unique; duplicates fail with a message that
multi-row trials are not supported yet. A second implementation stage will add
multi-row grouping behind a trial-assembly strategy without changing mapping,
filtering, validation, or `TrialCollection`.

Pipeline operations never mutate user input. DataFrames, coordinate arrays,
slot dictionaries, and numpy-array slot values are copied when a new view or
collection is returned. Immutable scalar values do not need special copying.

**Inspectability:** `TrialCollection.to_dataframe()` is available as a
developer tool for exploration and debugging. It is not a pipeline step
and must never appear in the critical path.

---

## Full Pipeline

```
Raw Dataset  (CSV, parquet, HuggingFace Dataset, DataFrame)
     ↓
DataAdapter(mapping).adapt(source, filters=...)
  internal load          any format → pd.DataFrame; normalize column labels
  internal map           raw column names → canonical paths
  internal defaults      add documented canonical defaults
  internal normalize     scalar missing sentinels → None
  internal filter        optional row filtering on coordinates
  internal validate      required canonical keys and row identity are valid
  internal assemble      one canonical row → one TrialCollection trial
     ↓
AdaptationResult         successful collection and adaptation metadata
  result.collection      TrialCollection
  result.report()        what mapped, defaulted, filtered, and ignored
     ↓
split(result.collection)  split on participant_id BEFORE ModelAdapter fits
                           to prevent eval statistics leaking into training
     ↓
SplitResult
  .train                 TrialCollection
  .eval                  TrialCollection
     ↓              ↓
ModelAdapter             public facade; dispatches by model type
  .fit(train)            learn encodings from train slot keys
  .transform(tc)         TrialCollection slots → tensors directly
                         no DataFrame intermediate — mechanical only
     ↓              ↓
train_tensors       eval_tensors
     ↓              ↓
Trainer
  .fit(train_tensors)
  .evaluate(eval_tensors)
     ↓
Model
  .compute_logits()      parameters and formula only
```

**Critical boundary:** `ModelAdapter.fit()` is called on `split_result.train`
only. `ModelAdapter.transform()` is then called separately on
`split_result.train` and `split_result.eval`. This mirrors the tidymodels
recipe pattern exactly.

**Split boundary:** `split()` operates on coordinates (`participant_id`)
only, never on slot content. Splitting is infrastructure, not modeling.

---

## Component Responsibilities

| Component | Owns | Does not own |
|---|---|---|
| `DataAdapter` | raw → `TrialCollection` | model and tensor logic |
| `ColumnMapping` | raw name → canonical path | model requirements |
| `AdaptationResult` | collection and report | transformation logic |
| `Split` | coordinate-based participant split | slot content, model knowledge |
| `SplitResult` | train and eval `TrialCollection` | any transformation logic |
| Model contract | canonical paths required by one model | raw column mapping |
| `ModelAdapter` | dispatch and slots → tensors | model formula |
| `Trainer` | training loop, optimizer | data, preprocessing |
| `Model` | parameters, `compute_logits()` | everything else |

**ModelAdapter vs Model boundary:**
- ModelAdapter exposes one public interface and dispatches internally to a
  model-specific implementation
- ModelAdapter owns: class encoding (GCM), fill values, tensor reshaping
- Model owns: learnable parameters and the mathematical formula only
- Compute logic (e.g. class vocabulary fitting in GCM) belongs in
  `ModelAdapter.fit()`, not in the model and not in tensorize utilities

**Split vs ModelAdapter boundary:**
- Split operates on coordinates only — `participant_id`, `session_id`,
  `block_index`, `trial_index`, `task_name`, `condition`
- Split never unpacks slot content
- ModelAdapter never performs splitting

---

## Layer Designs

### Model Contract (model side)

Each model declares which canonical `slot.key` paths it needs. This contract
belongs entirely to the model side. DataAdapter and ColumnMapping never read it.
The declaration and registration mechanism is a deferred model-side decision.

Conceptually, a model contract exposes:

```python
contract.required  # ("response.choice", "outcome.reward")
contract.optional  # ("response.rt", "outcome.feedback")
```

**Responsibility:** declare which canonical paths are required and optional for
one model. No raw dataset names, tensor shapes, or representation decisions.

---

### ColumnMapping (user side)

The user declares how raw column names map to canonical field paths. Mapping is
defined for a dataset and does not depend on which model will consume it.
Coordinates use their canonical names; content fields use full `slot.key`
paths so slot identity is always explicit.

Column resolution has two paths:

1. If the user supplies a mapping, the named raw key must exist. It is renamed
   to the canonical key before validation and all later processing. If the raw
   key is absent, the error reports that supplied key.
2. If the user supplies no mapping for a registered field, the adapter looks
   for the canonical key directly in the raw data. For example,
   `outcome.reward` looks for raw key `reward`. A missing required key is an
   error; a missing optional key remains `None`.

A mapping cannot introduce a new canonical field. Its target must already be a
coordinate or a `slot.key` in the canonical field registry. New targets are
registered first. Coordinates with documented defaults may be created
when no raw key or explicit mapping is present.

```python
mapping = ColumnMapping(
    mappings={
        "response.choice": "resp",
        "outcome.reward": "points",
    }
)
mapping.resolve()
# {"response.choice": "resp", "outcome.reward": "points", ...}
```

Coordinate columns can also be mapped explicitly. `condition` has a special
rule: if the raw dataset has no condition column and no explicit mapping is
declared, every trial receives `condition = 1` automatically — no
error, no sentinel.

```python
mapping = ColumnMapping(
    mappings={
        "participant_id": "subject",
        "task_name":      "task",
        "condition":      "block_type",   # optional — defaults to 1 if absent
        "response.choice": "resp",
    }
)
```

For repeated raw fields, the user declares a pattern targeting a canonical
field path:

```python
mapping = ColumnMapping(
    mappings={
        "response.choice": "response",
        "stimulus.ground_truth": "label",
    },
    patterns={"stimulus.features": r"^stimulus_\d+$"}
)
```

Pattern resolution follows these rules:

- Match column names with regex `fullmatch`.
- If the pattern has a named numeric `index` capture, sort by that value.
- Otherwise preserve raw DataFrame column order.
- Stack matched values along the last axis into a numpy array; never sum or
  average them.
- An explicitly supplied pattern that matches no columns is an error.

Reusing one raw column for multiple canonical paths is allowed with a warning.
Mapping multiple raw columns to one scalar canonical path is ambiguous and is
an error: "Multiple columns target one field; use a pattern for multi-column
mapping." Multiple columns may target any field marked as a pattern target in
the canonical field registry, but only through a pattern.

**Responsibility:** translate raw column names to canonical content paths and
coordinate names. No model knowledge.

---

### DataAdapter (independent component)

Reusable one-shot facade over a composable pipeline. The facade runs the fixed
stage order internally; each low-level stage remains an independently callable
and inspectable pure function.

```python
adapter = DataAdapter(mapping)
result = adapter.adapt(
    "path/to/data.csv",
    filters={"participant_id": ["p01", "p02"]},
)
```

Mapping, defaults, missing-scalar normalization, filtering, validation, and
assembly are separate pipeline steps. They are implemented as independently
testable pure functions used by the DataAdapter facade, not as one large
transform. The internal assembly stage groups canonical columns by slot and,
in the first implementation, converts each row to one logical trial in a
`TrialCollection`.

Pure stage functions raise strict local errors. The `DataAdapter` facade
does not catch or convert them; the first error stops the pipeline and no
`AdaptationResult` is constructed. The facade has no mutable per-stage methods
or out-of-order states. Advanced callers may compose the low-level functions
directly and then own their explicit ordering.
Detailed orchestration, failure, filtering, validation, and report semantics
are specified in `docs/design_docs/AdapterDesign.md`.


Split is a separate explicit step, not part of DataAdapter. Splitting
happens after `AdaptationResult`, before `ModelAdapter`.

**Responsibility:** raw data → `TrialCollection`. Nothing else.

---

### AdaptationResult

Returned by `.adapt()` only after successful assembly. It owns a
`TrialCollection` and inspectable success metadata. Failures propagate from the
stage that detects them and do not produce a result object.

```python
result.collection        # TrialCollection
result.report()

# Example report:
# Coordinates:
#   participant_id  ← subject_id   ✓
#   session_id      ← session_id   ✓  (default)
#   block_index     ← block        ✓
#   trial_index     ← trial_no     ✓
#
# Slot: stimulus
#   ground_truth    ← label        ✓
#
# Slot: response
#   choice          ← resp         ✓
#   rt              ← rt           ✓  (optional)
#
# Slot: outcome
#   reward          ← points       ✓
#
# Missing required coordinates or mappings:
#   (none)
#
# Unmapped raw columns (ignored):
#   condition, block_type, accuracy
```

---

### ModelAdapter (public facade, model-specific internally)

Fits on training `TrialCollection` only. Transforms slot content →
tensors directly. No DataFrame intermediate.

Users interact with one `ModelAdapter` interface. It dispatches internally by
model type to a model-specific adapter implementation; those implementations
are not separate public APIs.

```python
model_adapter = ModelAdapter(model)
model_adapter.fit(split_result.train)
train_tensors = model_adapter.transform(split_result.train)
eval_tensors  = model_adapter.transform(split_result.eval)
```

`ModelAdapter.transform()` unpacks content slots into tensors according to the
model-side contract and its model-specific implementation. All logic that
requires fitting belongs in `ModelAdapter.fit()`.

**TODO — `mode` parameter:** `transform()` will accept a `mode` parameter
to support ablation experiments comparing slot-aware vs. slot-agnostic
representations. Not implemented now. Interface reserved:
`transform(tc, mode="structured")`. Design when foundation model
training pipeline is built.

---

## File Structure

```
src/mt/data/
  _field_registry.py    Canonical coordinates, slots, keys, requirements,
                        defaults, and path validation
                        Detail: docs/design_docs/FieldRegistryDesign.md

  _loading.py           Load any format → raw pd.DataFrame (internal only)
                        Supported: CSV, parquet, HuggingFace Dataset,
                        DataFrame
                        Column labels: strings, missing → "None", unique
                        Entry: load(source) -> pd.DataFrame
                        Detail: docs/design_docs/LoadingDesign.md

  _mapping.py           ColumnMapping — raw names → canonical field paths
                        Identity lookup, explicit mappings, regex patterns,
                        ordering, stacking, and collision checks
                        Entry: ColumnMapping(mappings, patterns)
                               mapping.apply(df) -> pd.DataFrame copy
                        Detail: docs/design_docs/MappingDesign.md

  _collection.py        TrialCollection — validated trial data
                        Coordinates — assert if absent:
                          participant_id, trial_index
                        Coordinates — default 1 if absent:
                          session_id, block_index, task_name, condition
                        Slots: task (key: "instructions", default None),
                               context, stimulus, response, outcome
                        Entry: TrialCollection(...)
                               .copy() -> TrialCollection
                               .select(...) -> TrialCollection
                               .to_dataframe() -> pd.DataFrame  (debug only)
                        Detail: docs/design_docs/CollectionDesign.md

  _adapter.py           DataAdapter facade and AdaptationResult
                        Pure helpers apply defaults, normalize scalar missing
                        values, filter, validate, and assemble one-row trials
                        Entry: DataAdapter(mapping)
                               .adapt(source, filters=None)
                                   -> AdaptationResult
                        Detail: docs/design_docs/AdapterDesign.md

Later data modules, designed and implemented only when reached:
  _split.py             Split TrialCollection into train/eval by participant
                        SplitResult defined here — not in _collection.py
                        Entry: split(tc, by="participant_id", test_ratio=0.2,
                                     strategy="single") -> SplitResult
                        SplitResult fields: train: TrialCollection,
                                            eval: TrialCollection
                        Note: strategy param reserved for future CV support.
                              Do not implement CV now. Add as new branch later
                              without changing existing interface.

  _assembly.py          Added in the second implementation stage
                        OneRowTrialAssembler and GroupedTrialAssembler

src/mt/models/common/
  _model_contract.py    Model-side required and optional canonical paths
                        Declaration mechanism deferred — see below

  _model_adapter.py     Public ModelAdapter facade — dispatch by model type,
                        fit encodings, slots → tensors
                        Entry: ModelAdapter(model)
                               .fit(train_tc: TrialCollection) -> self
                               .transform(tc: TrialCollection)
                                   -> dict[str, Tensor]
                        TODO: mode param reserved — do not implement now
                              transform(tc, mode="structured") planned for
                              foundation model ablation study
                        Replaces: model.preprocess_data(train_df, eval_df)
```

---

## What to Remove or Move

| Current file | Decision |
|---|---|
| `_preparation.py` | Removed after `_adapter.py` replacement |
| `_prepared.py` | Removed after collection/result replacement |
| `_requests.py` | Removed; one-shot DataAdapter owns the supported request |
| `_checking.py` | Removed; replace with adapter validation |
| `_reports.py` | Removed; adaptation reporting lives on AdaptationResult |
| `view/` | Removed spec-bound entrypoints; independent helpers remain legacy |
| `_contracts.py` | Removed; replaced by canonical field registry |
| `models/common/_preprocessing.py` | Migrate logic to `_model_adapter.py` |
| `BaseCognitiveModel.preprocess_data()` | Replace with ModelAdapter |

---

## Scripts Required

```
scripts/
  load_dataset.py      Load a dataset and print AdaptationResult report
                       Usage: uv run scripts/load_dataset.py \
                                --source data.csv \
                                --mapping '{"outcome.reward": "points"}'

  validate_dataset.py  Validate canonical structure and mapping
                       Print full report without selecting a model
```

---

## Tests Required

```
tests/data/
  test_loading.py        Every supported format loads correctly
  test_field_registry.py Canonical slots and fixed field paths are correct
  test_mapping.py        Explicit mappings rename raw keys before validation
                         Unmapped fixed fields use their canonical key
                         Missing supplied raw key produces a clear error
                         Missing required identity key produces a clear error
                         Coordinate mappings work
                         Patterns use fullmatch
                         Numeric index captures use numeric order
                         Patterns without index capture keep source order
                         Matched columns stack along the last axis
                         Empty pattern match raises a clear error
                         Reused raw column produces a warning
                         Multiple raw columns to one scalar field fail
                         Mapping is independent of models
  test_collection.py     TrialCollection builds correctly from valid data
                         Coordinate arrays have correct shape — all six
                         Coordinate and slot lengths stay aligned
                         Registered slot keys are exact and complete
                         All five content slots are present after build
                         Content is stored under the correct slot and key
                         copy/select do not mutate the source collection
                         Numpy-array slot values are copied
                         to_dataframe() is available but not tested
                         as a pipeline step
  test_adapter.py        Each pipeline step works independently
                         One adapt() call runs the complete ordered pipeline
                         Mapping runs after loading and before filtering
                         Missing optional fields receive registry defaults
                         Scalar missing sentinels normalize to None
                         One raw row produces one logical trial
                         Duplicate trial identity reports unsupported
                         multi-row trials
                         Successful result contains TrialCollection
                         Stage errors propagate and construct no result
  test_split.py          Split produces two non-overlapping TrialCollections
                         Split is by participant_id, not by row index
                         No participant appears in both train and eval
                         Coordinates and slot lists stay aligned after split
                         Split always precedes ModelAdapter.fit()

tests/models/
  test_model_contract.py Model declares required canonical field paths
  test_model_adapter.py  fit() on train TrialCollection only
                         Public facade dispatches to model-specific adapter
                         transform() on both train and eval
                         Encodings fitted on train applied correctly to eval
                         transform() produces model-ready tensor payloads
                         No DataFrame appears anywhere in the pipeline
```

---

## What Is Out of Scope

- Streaming or lazy loading of large datasets
- Heuristic column name inference — identity lookup uses exact canonical keys
- Multi-dataset joins or merges
- Cross-validation splits — modularized for future addition via
  strategy parameter, not implemented now

---

## Deferred Decisions

**Fitted missing-value treatment**
DataAdapter only canonicalizes scalar missing sentinels to `None`. It does not
delete trials, fill missing content, or alter missing values inside NumPy
arrays. Statistical strategies such as mean imputation must be designed after
split so they fit on training data only and cannot leak evaluation statistics.
Deletion must name explicit canonical paths and define its effect on sequence
history. The later design must decide whether this fitted boundary is a
reusable preprocessing transform or part of a model-specific ModelAdapter.

**Model contract declaration**
How models declare and register their required canonical paths (class
attribute, classmethod, separate map file, or decorator) is a model-side design
decision not yet settled. It does not affect DataAdapter or ColumnMapping, but
must be resolved before implementing ModelAdapter integration.

**ModelAdapter dispatch**
The public interface and model-specific internal behavior are decided. The
registry, class hierarchy, and unsupported-model failure behavior are deferred
to the detailed model-side design.

**`ModelAdapter.transform()` mode parameter**
A `mode` parameter is reserved for foundation model ablation experiments
comparing slot-aware vs. slot-agnostic representations. Not designed or
implemented yet. Tackle when foundation model training pipeline is built.

**HuggingFace Datasets export**
`TrialCollection` should support export to HF Datasets format for
foundation model training at scale. API not designed yet. Add as a
separate export method on `TrialCollection` when foundation model
training pipeline is built. Must not change the internal structure of
`TrialCollection`.
