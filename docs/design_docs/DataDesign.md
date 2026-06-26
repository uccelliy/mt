# Design Doc: Data Contract and Adapter System

**Status:** Proposed
**Author:** Ruochen YIN
**Date:** 2026-06

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

A three-layer system where:

1. The **model declares** exactly what it needs (contract)
2. The **user declares** how their dataset maps to those needs (mapping)
3. The **adapter** does the translation and fails early with a clear report

No model should ever touch raw data. No dataset should ever need to know
what model will consume it.

---

## Inspiration

**HuggingFace:** tokenizer and model are separate objects. The tokenizer
fails early if the input is wrong before the model ever sees the data.

**Tidymodels:** recipe and model are separate objects bundled into a
workflow. The recipe is composable — built from explicit, ordered steps.
The model is never aware of preprocessing.

The key shared insight: **the transformation boundary must be explicit,
independently testable, and fail early.**

---

## Canonical DataFrame — The Central Idea

The adapter does not produce tensors. It produces a **Canonical DataFrame**
— a well-named, well-typed DataFrame whose column names always match the
model's tensor key names. This is the shared intermediate format that makes
the whole system composable.

The canonical DataFrame is not a new class or component. It is a naming
convention enforced by the adapter. After adaptation, every column name
speaks the model's language regardless of what the raw dataset called it.

```
Raw Dataset                   Canonical DataFrame
  "response"      ──→           "choice"
  "outcome"       ──→           "reward"
  "forced_trial"  ──→           "forced"
  "subject_id"    ──→           "participant"  (standard index)
  "block"         ──→           "task"         (standard index)
  "trial_number"  ──→           "trial"        (standard index)
```

The model's `tensorize()` always receives a canonical DataFrame and never
needs to know the original column names. The adapter never needs to know
what the model will do with the tensors.

### Canonical Column Conventions

Every canonical DataFrame must have these index columns:

| Column | Type | Meaning |
|---|---|---|
| `participant` | str | Unique participant identifier |
| `task` | str | Task or block identifier |
| `trial` | int | Trial number within task |

All other columns are model-specific and declared by the model's contract.

---

## Design

### Layer 1 — DataContract (model side)

The model declares what canonical columns it needs. This already exists in
`mt.models.common._contracts` as `ModelDataSpec`. It should be formalized
as the public-facing `DataContract`.

```python
# Model declares its contract — model side, not data side
contract = RescorlaWagnerModel.data_contract()

contract.required_columns   # ("choice", "reward")
contract.optional_columns   # ("forced",)
contract.derived_columns    # ("choice_for_updating",)
```

**Responsibility:** declare what canonical column names are needed.
No knowledge of any raw dataset. No knowledge of tensor shapes.

---

### Layer 2 — ColumnMapping (user side)

The user declares how their raw dataset column names map to canonical names.
The model ships default mappings for the common case. The user overrides
only what differs.

```python
# Model default mapping — assumes raw names match canonical names
# {"choice": "choice", "reward": "reward", "forced": "forced",
#  "participant": "participant", "task": "task", "trial": "trial"}

# User override — only what differs in their dataset
mapping = ColumnMapping(
    model=RescorlaWagnerModel,
    overrides={
        "choice": "response",
        "reward": "outcome",
        "participant": "subject_id",
        "trial": "trial_number",
    }
)

mapping.resolve()
# {"choice": "response", "reward": "outcome", "forced": "forced",
#  "participant": "subject_id", "task": "task", "trial": "trial_number"}
```

**Responsibility:** translate raw column names to canonical column names.
Nothing else.

---

### Layer 3 — DataAdapter (independent component)

The adapter takes a source, a contract, and a mapping. It runs a composable
pipeline of explicit steps and produces a canonical DataFrame.

```python
adapter = DataAdapter(
    source="path/to/data.csv",   # or DataFrame or HF dataset
    contract=contract,
    mapping=mapping,
)

result = (
    adapter
    .load()                                  # raw df
    .validate()                              # check required columns exist
    .filter(participant=["p01", "p02"])      # optional row filtering
    .transform()                             # apply column mapping
    .split(by="participant", test_ratio=0.2) # train/eval split
    .adapt()                                 # produce AdaptationResult
)
```

`.adapt()` produces a canonical DataFrame in `result`, not tensors.
The adapter never calls `tensorize()` — that is the model's job.

Each step is independently callable and inspectable.

**Responsibility:** raw data → canonical DataFrame. Nothing else.

---

### Layer 4 — Model.tensorize() (model side)

The model converts a canonical DataFrame to its own tensor format.
This is where model-specific logic lives: derived keys, fill values,
encoding, reshaping.

```python
# Adapter produces canonical df
result = adapter.adapt()

# Model converts canonical df → tensors
train_tensors = model.tensorize(result.train_df)
eval_tensors  = model.tensorize(result.eval_df)

trainer.fit(train_tensors)
```

`tensorize()` replaces `preprocess_data()` in the current codebase.
The difference: `tensorize()` always receives a canonical DataFrame
with column names it already knows. It never handles raw column names.

**Responsibility:** canonical DataFrame → model-specific tensors. Nothing else.

---

### Layer 5 — AdaptationResult

The result is always returned, never raised silently. Complete or partial,
the user can always inspect it.

```python
result.complete       # True / False
result.train_df       # canonical DataFrame or None
result.eval_df        # canonical DataFrame or None
result.report()       # human-readable summary

# Example report output:
# Mapped columns:
#   choice       ← response      ✓
#   reward       ← outcome       ✓
#   forced       ← forced        ✓  (optional)
#   participant  ← subject_id    ✓
#   task         ← task          ✓
#   trial        ← trial_number  ✓
#
# Missing required columns:
#   (none)
#
# Unmapped raw columns (ignored):
#   rt, condition, block_type
```

If `result.complete` is False, `train_df` and `eval_df` are None.
The report shows what mapped, what is missing, and what was ignored.

---

## File Structure

```
src/mt/data/
  _loading.py       Load any format → raw DataFrame
                    Supported: CSV, parquet, HuggingFace Dataset, DataFrame
                    Entry: load(source) -> pd.DataFrame

  _contract.py      DataContract — canonical columns the model needs
                    Entry: DataContract.from_model(model) -> DataContract

  _mapping.py       ColumnMapping — raw names → canonical names
                    Entry: ColumnMapping(model, overrides) -> ColumnMapping
                           mapping.resolve() -> dict[str, str]

  _adapter.py       DataAdapter — composable pipeline, raw → canonical df
                    Entry: DataAdapter(source, contract, mapping)
                           .load() -> self
                           .validate() -> self
                           .filter(**kwargs) -> self
                           .transform() -> self
                           .split(by, test_ratio) -> self
                           .adapt() -> AdaptationResult

  _result.py        AdaptationResult — canonical df, inspectable report
                    Fields: complete, train_df, eval_df
                    Entry: result.report() -> str

src/mt/models/common/
  _base.py          BaseCognitiveModel — add tensorize() here
                    Entry: model.tensorize(canonical_df) -> dict[str, Tensor]
                    Replaces: model.preprocess_data(train_df, eval_df)
```

---

## What to Remove or Move

| Current file | Decision |
|---|---|
| `_preparation.py` | Replace with `_adapter.py` — composable steps |
| `_prepared.py` | Replace with `_result.py` — cleaner name and contract |
| `_requests.py` | Fold into `_adapter.py` — DataRequest → DataAdapter |
| `_checking.py` | Fold into `_adapter.py` `.validate()` step |
| `_reports.py` | Fold into `_result.py` — part of AdaptationResult.report() |
| `view/` | Keep in `mt.data` — fold filtering, splitting, and transforms |
|           | into DataAdapter pipeline steps |

---

## Scripts Required

```
scripts/
  load_dataset.py       Load a dataset and print a summary report
                        Usage: uv run scripts/load_dataset.py \
                                 --source data.csv \
                                 --model RescorlaWagnerModel \
                                 --mapping '{"reward": "outcome"}'

  validate_dataset.py   Validate a dataset against a model contract
                        and print a full AdaptationResult report
                        without running the model
```

---

## Tests Required

```
tests/data/
  test_loading.py       Every supported format loads to a DataFrame
  test_contract.py      Contract correctly reflects model's declared needs
  test_mapping.py       Overrides resolve correctly against model defaults
                        Missing required keys produce clear errors
  test_adapter.py       Each pipeline step works independently
                        Complete result runs through to tensors
                        Partial result reports what is missing
  test_result.py        report() output is human-readable and complete
```

---

## What Is Out of Scope

- Streaming or lazy loading of large datasets
- Automatic column name inference (mapping is always explicit)
- Multi-dataset joins or merges

---

## Open Questions — Canonical DataFrame Design

These need to be answered before implementation starts.

**1. Index columns — are participant, task, trial always required?**
Some datasets may not have a task column (single-task experiments).
Should index columns be part of the contract or always enforced?

**2. Variable-column models (GeneralizedContextModel)**
Features arrive as `x1, x2, ... xN` — the number of columns varies by
dataset. How does ColumnMapping handle a pattern rather than a fixed name?
Currently handled via `MODEL_COLUMN_PATTERNS` — needs to be preserved.

**3. tensorize() signature**
Current `preprocess_data(train_df, eval_df)` takes both splits together
because some models encode across train+eval jointly (e.g. class encoding
in GeneralizedContextModel). Should `tensorize()` take both splits or one
at a time? Taking both is less clean but may be necessary.

**4. Split strategy**
Should `DataAdapter.split()` support cross-validation or only single
train/eval split for now?