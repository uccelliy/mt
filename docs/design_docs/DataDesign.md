# Design Doc: Canonical Data and Adapter System

**Status:** Proposed
**Author:** [your name]
**Date:** 2026-06-28

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

1. The **canonical vocabulary** defines shared slot and field names
2. The **user declares** how their raw dataset maps to canonical field paths
   without selecting a model
3. The **DataAdapter** produces a `TrialCollection` — fails early with a
   clear report if mapping is incomplete
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

## Canonical Vocabulary

| Slot | Key | Description |
|---|---|---|
| `task` | `instructions` | Task instructions; `None` if absent |
| `context` | *(user-defined)* | Trial framing; `None` if absent |
| `stimulus` | `ground_truth` | Correct label; `None` if absent |
| `stimulus` | *(user-defined)* | Presented object; `None` if absent |
| `response` | `choice` | Participant choice; `None` if absent |
| `response` | `rt` | Response time in milliseconds; `None` if absent |
| `outcome` | `reward` | Objective consequence; `None` if absent |
| `outcome` | `feedback` | Presented feedback; `None` if absent |

The five content slots are `task`, `context`, `stimulus`, `response`, and
`outcome`. A canonical content field is identified by its full `slot.key`
path, for example `response.choice` or `stimulus.ground_truth`. Key names do
not need to be globally unique across slots; both `context.color` and
`stimulus.color` may exist without conflict.

The canonical vocabulary is data-side and model-independent. There is no
data-side contract derived from a model. Each model has a separate model-side
contract that refers to canonical field paths.

---

## TrialCollection — Principle

The DataAdapter does not produce tensors or a flat DataFrame. It produces
a `TrialCollection` — a typed class that holds validated trial data in a
form that any model, including deep neural networks, can consume without
the canonical structure having made representational decisions on its behalf.

A flat DataFrame is a representational commitment: it says stimulus is a
fixed set of named scalars. `TrialCollection` defers that decision to each
model's adapter. The canonical structure is an interface, not a
representation.

**Why not HuggingFace Datasets:** HF Datasets solves the storage problem
(mixed scalars and nested features, Arrow-backed, efficient batching) and
is the right export target for foundation model training at scale.
It is the wrong internal representation for this pipeline because it has
no concept of the canonical vocabulary, participant-aware splitting, or
`ModelAdapter.fit()`. HF Datasets export is a future addition, not a
replacement for `TrialCollection`.

### Structure

`TrialCollection` has two fundamentally different kinds of fields:

**Coordinates** — scalar numpy arrays, shape `(n_trials,)`. Used for
splitting and filtering before any model is involved. These are addresses,
not content. Making them scalar is a database design decision, not a
modeling assumption.

**Content slots** — lists of dicts, length `n_trials`. Each dict holds
canonical or user-defined keys within one of the five canonical slots. The
canonical structure makes no claim about how values should be represented for
a model — that decision belongs to its model-specific adapter.

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
canonical `slot.key` paths before slot construction. `ColumnMapping` does not
read a model contract. Model contracts later refer to the same canonical paths.

```
Raw Dataset                  TrialCollection
  "resp"       ──→           response[i]["choice"]
  "points"     ──→           outcome[i]["reward"]
  "label"      ──→           stimulus[i]["ground_truth"]
  "cue"        ──→           context[i]["cue"]
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

A raw row is not assumed to equal one canonical logical trial. DataAdapter
owns the structural adaptation required when one trial spans multiple rows.
The concrete grouping and assembly design is deferred.

**Inspectability:** `TrialCollection.to_dataframe()` is available as a
developer tool for exploration and debugging. It is not a pipeline step
and must never appear in the critical path.

---

## Full Pipeline

```
Raw Dataset  (CSV, parquet, HuggingFace Dataset, DataFrame)
     ↓
DataAdapter              raw → TrialCollection
  .load()                any format → pd.DataFrame (internal staging only)
  .validate()            required coordinates and mappings exist
  .filter()              optional row filtering on coordinates
  .transform()           translate raw names/layout → canonical data
  .adapt()               build and return AdaptationResult
     ↓
AdaptationResult         TrialCollection, complete or partial
  result.complete        True / False
  result.collection      TrialCollection or None
  result.report()        what mapped, what is missing, what was ignored
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
paths so user-defined keys may repeat across different slots.

```python
mapping = ColumnMapping(
    overrides={
        "response.choice": "resp",
        "outcome.reward": "points",
        "context.cue": "cue",
    }
)
mapping.resolve()
# {"response.choice": "resp", "outcome.reward": "points",
#  "context.cue": "cue", ...}
```

Coordinate columns can also be overridden. `condition` has a special
rule: if the raw dataset has no condition column and no override is
declared, every trial receives `condition = 1` automatically — no
error, no sentinel.

```python
mapping = ColumnMapping(
    overrides={
        "participant_id": "subject",
        "task_name":      "task",
        "condition":      "block_type",   # optional — defaults to 1 if absent
        "response.choice": "resp",
    }
)
```

For repeated raw fields, the user may declare a pattern targeting a canonical
field path. Exact pattern and aggregation behavior remains a detailed design
decision:

```python
mapping = ColumnMapping(
    overrides={
        "response.choice": "response",
        "stimulus.ground_truth": "label",
    },
    patterns={"stimulus.features": r"^stimulus_\d+$"}
)
```

**Responsibility:** translate raw column names to canonical content paths and
coordinate names. No model knowledge.

---

### DataAdapter (independent component)

Composable pipeline. Each step is independently callable and inspectable.
Internally stages data as a raw DataFrame — this is never exposed outside
the adapter.

```python
adapter = DataAdapter(
    source="path/to/data.csv",
    mapping=mapping,
)

result = (
    adapter
    .load()
    .validate()
    .filter(participant_id=["p01", "p02"])
    .transform()
    .adapt()
)
```

Split is a separate explicit step, not part of DataAdapter. Splitting
happens after `AdaptationResult`, before `ModelAdapter`.

**Responsibility:** raw data → `TrialCollection`. Nothing else.

---

### AdaptationResult

Always returned, never raised silently.

```python
result.complete          # True / False
result.collection        # TrialCollection or None
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
# Slot: context
#   cue             ← cue          ✓
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
  _loading.py           Load any format → raw pd.DataFrame (internal only)
                        Supported: CSV, parquet, HuggingFace Dataset,
                        DataFrame
                        Entry: load(source) -> pd.DataFrame

  _mapping.py           ColumnMapping — raw names → canonical field paths
                        Supports fixed names and regex patterns
                        Also maps coordinate columns
                        Entry: ColumnMapping(overrides, patterns)
                               mapping.resolve() -> dict[str, str]

  _collection.py        TrialCollection — validated trial data
                        Coordinates — assert if absent:
                          participant_id, trial_index
                        Coordinates — default 1 if absent:
                          session_id, block_index, task_name, condition
                        Slots: task (key: "instructions", default None),
                               context, stimulus, response, outcome
                        Entry: TrialCollection(...)
                               .to_dataframe() -> pd.DataFrame  (debug only)

  _adapter.py           DataAdapter — raw → TrialCollection pipeline
                        Entry: DataAdapter(source, mapping)
                               .load() -> self
                               .validate() -> self
                               .filter(**kwargs) -> self
                               .transform() -> self
                               .adapt() -> AdaptationResult

  _result.py            AdaptationResult — TrialCollection and mapping report
                        Fields: complete, collection
                        Entry: result.report() -> str

  _split.py             Split TrialCollection into train/eval by participant
                        SplitResult defined here — not in _collection.py
                        Entry: split(tc, by="participant_id", test_ratio=0.2,
                                     strategy="single") -> SplitResult
                        SplitResult fields: train: TrialCollection,
                                            eval: TrialCollection
                        Note: strategy param reserved for future CV support.
                              Do not implement CV now. Add as new branch later
                              without changing existing interface.

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
| `_preparation.py` | Replace with `_adapter.py` — composable steps |
| `_prepared.py` | Replace with `_result.py` — cleaner contract |
| `_requests.py` | Fold into `_adapter.py` — DataRequest → DataAdapter |
| `_checking.py` | Fold into `_adapter.py` `.validate()` step |
| `_reports.py` | Fold into `_result.py` — part of result.report() |
| `view/` | Fold filtering and transforms into DataAdapter pipeline |
| `_contracts.py` | Replace data contract with canonical vocabulary |
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
  test_vocabulary.py     Canonical slots and fixed field paths are correct
  test_mapping.py        Qualified overrides resolve correctly
                         Coordinate overrides work
                         Pattern matching works for variable-key models
                         Mapping is independent of models
  test_collection.py     TrialCollection builds correctly from valid data
                         Coordinate arrays have correct shape — all six
                         participant_id absent → ValueError raised
                         trial_index absent → ValueError raised
                         session_id absent → defaults to array of 1s
                         block_index absent → defaults to array of 1s
                         task_name absent → defaults to array of 1s
                         condition absent → defaults to array of 1s
                         task["instructions"] present when column exists
                         task["instructions"] is None when column absent
                         All five content slots are present after build
                         Content is stored under the correct slot and key
                         to_dataframe() is available but not tested
                         as a pipeline step
  test_adapter.py        Each pipeline step works independently
                         Complete result contains TrialCollection
                         Partial result reports what is missing
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
- Automatic column name inference — mapping is always explicit
- Multi-dataset joins or merges
- Cross-validation splits — modularized for future addition via
  strategy parameter, not implemented now

---

## Deferred Decisions

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
