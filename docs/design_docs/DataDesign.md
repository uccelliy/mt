# Design Doc: Data Contract and Adapter System

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

1. The **model declares** exactly what slot keys it needs (contract)
2. The **user declares** how their raw dataset maps to canonical slot keys
   (mapping)
3. The **DataAdapter** produces a `TrialCollection` — fails early with a
   clear report if mapping is incomplete
4. The **ModelAdapter** fits on training data, transforms `TrialCollection`
   → tensors for both splits
5. The **Model** only sees tensors and computes logits

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
no concept of `DataContract`, participant-aware splitting, or
`ModelAdapter.fit()`. HF Datasets export is a future addition, not a
replacement for `TrialCollection`.

### Structure

`TrialCollection` has two fundamentally different kinds of fields:

**Coordinates** — scalar numpy arrays, shape `(n_trials,)`. Used for
splitting and filtering before any model is involved. These are addresses,
not content. Making them scalar is a database design decision, not a
modeling assumption.

**Content slots** — lists of dicts, length `n_trials`. Each dict holds
paradigm-specific keys validated against the `DataContract`. The canonical
structure makes no claim about what the keys mean or how they should be
represented — that decision belongs to the `ModelAdapter` for each model.

```python
@dataclass
class TrialCollection:
    # Coordinates — scalar, universal, used by split() and filter()
    participant_id: np.ndarray    # shape (n_trials,)
    session_id: np.ndarray
    block_index: np.ndarray
    trial_index: np.ndarray
    task_name: np.ndarray         # routing key — which task/contract applies
    condition: np.ndarray         # trial category; default 1 when absent

    # Content slots — list of dicts, keys validated against contract
    task: list[dict]        # instructions text only — no routing fields
    stimulus: list[dict]    # keys declared by model contract
    response: list[dict]    # keys declared by model contract
    outcome: list[dict]     # keys declared by model contract

    # Contract used to build and validate this collection
    contract: DataContract
```

**Why `task_name` and `condition` are coordinates, not slot content:**
the pipeline needs both before any model is involved — `task_name` for
contract routing, `condition` for filtering practice or forced-choice
trials. Infrastructure must never unpack slot content to do routing.
Coordinates are addresses; slot content is model input.

**`condition` default:** when the raw dataset has no condition column,
every trial receives `condition = 1`. This means "one condition,
undivided" — not an error, not a sentinel, just the degenerate case of
the general structure. No special-casing required downstream.

**`task` slot — instructions only:** the task slot holds the natural
language instructions shown to participants. Classical cognitive models
declare `task` as not required in their contract — the slot costs them
nothing. Foundation and language models consume it as a free-text input
with no taxonomy imposed.

**Slot key convention:** keys inside content slots always match model
contract key names. Raw names are translated by `ColumnMapping` before
slot construction. After translation, slot keys are fixed and validated
against the contract by `DataAdapter.validate()`.

```
Raw Dataset                  TrialCollection
  "resp"       ──→           response[i]["choice"]
  "points"     ──→           outcome[i]["reward"]
  "cue"        ──→           stimulus[i]["context"]
  "subject_id" ──→           participant_id[i]
  "cond"       ──→           condition[i]            (coordinate)
  "task"       ──→           task_name[i]            (coordinate)
  (absent)     ──→           condition[i] = 1        (default)
```

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
  .validate()            required slot keys exist, fail early with report
  .filter()              optional row filtering on coordinates
  .transform()           translate raw names → contract slot keys
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
ModelAdapter             fits on train only, applies to both
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
| `DataAdapter` | raw → `TrialCollection`, validation, filtering | model knowledge, tensor logic |
| `ColumnMapping` | raw name → contract slot key translation | any data transformation |
| `AdaptationResult` | `TrialCollection`, mapping report | any transformation logic |
| `Split` | coordinate-based participant split | slot content, model knowledge |
| `SplitResult` | train and eval `TrialCollection` | any transformation logic |
| `ModelAdapter` | fit encodings on train, slots → tensors | compute logic, formula |
| `Trainer` | training loop, optimizer | data, preprocessing |
| `Model` | parameters, `compute_logits()` | everything else |

**ModelAdapter vs Model boundary:**
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

### DataContract (model side)

The model declares what slot keys it needs within each content slot.

```python
contract = RescorlaWagnerModel.data_contract()
contract.stimulus.required   # ("context",)
contract.stimulus.optional   # ()
contract.response.required   # ("choice",)
contract.response.optional   # ("rt",)
contract.outcome.required    # ("reward",)
contract.outcome.optional    # ("feedback",)
```

**Responsibility:** declare which slot keys are required and optional per
content slot. No dataset knowledge. No tensor shapes. No representation
decisions.

---

### ColumnMapping (user side)

The user declares how raw column names map to contract slot keys. The
contract already knows which slot each key belongs to — the user overrides
only the raw name, not the slot assignment.

The model ships a default mapping where raw name equals contract key name.
The user overrides only what differs.

```python
mapping = ColumnMapping(
    model=RescorlaWagnerModel,
    overrides={
        "choice":   "resp",         # raw "resp" → response["choice"]
        "reward":   "points",       # raw "points" → outcome["reward"]
        "context":  "cue",          # raw "cue" → stimulus["context"]
    }
)
mapping.resolve()
# {"choice": "resp", "reward": "points", "context": "cue",
#  "rt": "rt", "feedback": "feedback", ...}
```

Coordinate columns can also be overridden. `condition` has a special
rule: if the raw dataset has no condition column and no override is
declared, every trial receives `condition = 1` automatically — no
error, no sentinel.

```python
mapping = ColumnMapping(
    model=RescorlaWagnerModel,
    overrides={
        "participant_id": "subject",
        "task_name":      "task",
        "condition":      "block_type",   # optional — defaults to 1 if absent
        "choice":         "resp",
    }
)
```

For variable-key models (e.g. GeneralizedContextModel), the user declares
a pattern. The contract declares the slot (`stimulus`) and the key prefix
(`features`). The user declares the raw pattern:

```python
mapping = ColumnMapping(
    model=GeneralizedContextModel,
    overrides={"choice": "response", "ground_truth": "label"},
    patterns={"features": r"^stimulus_\d+$"}
)
```

**Responsibility:** translate raw column names to contract slot keys and
coordinate names. Nothing else.

---

### DataAdapter (independent component)

Composable pipeline. Each step is independently callable and inspectable.
Internally stages data as a raw DataFrame — this is never exposed outside
the adapter.

```python
adapter = DataAdapter(
    source="path/to/data.csv",
    contract=contract,
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
#   context         ← cue          ✓
#
# Slot: response
#   choice          ← resp         ✓
#   rt              ← rt           ✓  (optional)
#
# Slot: outcome
#   reward          ← points       ✓
#
# Missing required keys:
#   (none)
#
# Unmapped raw columns (ignored):
#   condition, block_type, accuracy
```

---

### ModelAdapter (model side, independent component)

Fits on training `TrialCollection` only. Transforms slot content →
tensors directly. No DataFrame intermediate.

```python
model_adapter = ModelAdapter(model)
model_adapter.fit(split_result.train)
train_tensors = model_adapter.transform(split_result.train)
eval_tensors  = model_adapter.transform(split_result.eval)
```

`ModelAdapter.transform()` unpacks content slots into tensors according
to what the model contract declared. It does only mechanical conversion —
no compute logic, no classification, no statistics. All logic that
requires fitting belongs in `ModelAdapter.fit()`.

**Responsibility:** fit encodings on train `TrialCollection`. Convert
slot content → tensors. Nothing else.

---

## File Structure

```
src/mt/data/
  _loading.py           Load any format → raw pd.DataFrame (internal only)
                        Supported: CSV, parquet, HuggingFace Dataset,
                        DataFrame
                        Entry: load(source) -> pd.DataFrame

  _contract.py          DataContract — slot keys the model needs
                        Entry: DataContract.from_model(model) -> DataContract
                        Fields per slot: required: tuple[str], optional: tuple[str]

  _mapping.py           ColumnMapping — raw names → contract slot keys
                        Supports fixed names and regex patterns
                        Also maps coordinate columns
                        Entry: ColumnMapping(model, overrides, patterns)
                               mapping.resolve() -> dict[str, str]

  _collection.py        TrialCollection — validated trial data
                        Coordinates: participant_id, session_id,
                                     block_index, trial_index,
                                     task_name, condition (default 1)
                        Slots: task (instructions only), stimulus,
                               response, outcome
                        Entry: TrialCollection(...)
                               .to_dataframe() -> pd.DataFrame  (debug only)

  _adapter.py           DataAdapter — composable pipeline, raw → TrialCollection
                        Entry: DataAdapter(source, contract, mapping)
                               .load() -> self
                               .validate() -> self
                               .filter(**kwargs) -> self
                               .transform() -> self
                               .adapt() -> AdaptationResult

  _result.py            AdaptationResult — TrialCollection and mapping report
                        Fields: complete, collection
                        Entry: result.report() -> str

  _split.py             Split TrialCollection into train/eval by participant
                        Entry: split(tc, by="participant_id", test_ratio=0.2,
                                     strategy="single") -> SplitResult
                        SplitResult fields: train: TrialCollection,
                                            eval: TrialCollection
                        Note: strategy param reserved for future CV support.
                              Do not implement CV now. Add as new branch later
                              without changing existing interface.

src/mt/models/common/
  _model_adapter.py     ModelAdapter — fit encodings, slots → tensors
                        Entry: ModelAdapter(model)
                               .fit(train_tc: TrialCollection) -> self
                               .transform(tc: TrialCollection)
                                   -> dict[str, Tensor]
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
| `models/common/_preprocessing.py` | Migrate logic to `_model_adapter.py` |
| `BaseCognitiveModel.preprocess_data()` | Replace with ModelAdapter |

---

## Scripts Required

```
scripts/
  load_dataset.py      Load a dataset and print AdaptationResult report
                       Usage: uv run scripts/load_dataset.py \
                                --source data.csv \
                                --model RescorlaWagnerModel \
                                --mapping '{"reward": "outcome"}'

  validate_dataset.py  Validate dataset against model contract
                       Print full report without running the model
```

---

## Tests Required

```
tests/data/
  test_loading.py        Every supported format loads correctly
  test_contract.py       Contract reflects model declared slot keys
                         Required and optional keys per slot are correct
  test_mapping.py        Overrides resolve against model defaults
                         Coordinate overrides work
                         Pattern matching works for variable-key models
                         Missing required keys produce clear errors
  test_collection.py     TrialCollection builds correctly from valid data
                         Coordinate arrays have correct shape — all six
                         task_name coordinate is populated from raw data
                         condition defaults to 1 when absent from raw data
                         condition is overridden correctly when present
                         Slot keys match contract after build
                         task slot holds instructions text only —
                         no task_name or condition in slot dicts
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
  test_model_adapter.py  fit() on train TrialCollection only
                         transform() on both train and eval
                         Encodings fitted on train applied correctly to eval
                         transform() produces tensors only, no logic
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

**HuggingFace Datasets export**
`TrialCollection` should support export to HF Datasets format for
foundation model training at scale. API not designed yet. Add as a
separate export method on `TrialCollection` when foundation model
training pipeline is built. Must not change the internal structure of
`TrialCollection`.
