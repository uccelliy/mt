# Design Doc: Data Contract and Adapter System

**Status:** Proposed
**Author:** [your name]
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

A system where:

1. The **model declares** exactly what canonical columns it needs (contract)
2. The **user declares** how their raw dataset maps to canonical names (mapping)
3. The **DataAdapter** produces a canonical DataFrame — fails early with a
   clear report if mapping is incomplete
4. The **ModelAdapter** fits on training data, transforms canonical df →
   tensors for both splits
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

## Canonical DataFrame — Principle

The DataAdapter does not produce tensors. It produces a **Canonical
DataFrame** — a DataFrame whose column names always match the model's
declared contract keys, regardless of what the raw dataset called them.

The canonical DataFrame is not a new class. It is a naming convention
enforced by the adapter.

```
Raw Dataset                Canonical DataFrame
  "response"    ──→          "choice"
  "outcome"     ──→          "reward"
  "subject_id"  ──→          "participant"
  "trial_no"    ──→          "trial"
```

The ModelAdapter always receives a canonical DataFrame and never needs to
know the original column names. The DataAdapter never needs to know what
the model will do with the data.

**Note:** The full canonical schema — required index columns, context
columns, standard types — requires further research into cognitive science
data standards and existing codebases. Schema definition is deferred.
Principle: canonical column names always match model contract key names.

---

## Full Pipeline

```
Raw Dataset  (CSV, parquet, HuggingFace Dataset, DataFrame)
     ↓
DataAdapter              raw → canonical DataFrame
  .load()                any format → pd.DataFrame
  .validate()            required columns exist, fail early with report
  .filter()              optional row filtering
  .transform()           rename columns → canonical names
     ↓
AdaptationResult         canonical DataFrame, complete or partial
  result.complete        True / False
  result.canonical_df    canonical DataFrame or None
  result.report()        what mapped, what is missing, what was ignored
     ↓
Split                    train_df / eval_df separated here
                         splitting happens BEFORE ModelAdapter fits
                         to prevent eval statistics leaking into training
     ↓              ↓
ModelAdapter             fits on train_df only, applies to both
  .fit(train_df)         learn encodings, class vocabularies, statistics
  .transform(df)         canonical df → tensors
                         mechanical conversion only — no compute logic
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

**Critical boundary:** ModelAdapter.fit() is called on train_df only.
ModelAdapter.transform() is then called separately on train_df and eval_df.
This mirrors the tidymodels recipe pattern exactly.

---

## Component Responsibilities

| Component | Owns | Does not own |
|---|---|---|
| `DataAdapter` | raw → canonical df, validation, filtering | model knowledge, tensor logic |
| `ColumnMapping` | raw name → canonical name translation | any data transformation |
| `AdaptationResult` | canonical df, mapping report | any transformation logic |
| `ModelAdapter` | fit encodings on train, df → tensors | compute logic, formula |
| `Trainer` | training loop, optimizer | data, preprocessing |
| `Model` | parameters, `compute_logits()` | everything else |

**ModelAdapter vs Model boundary:**
- ModelAdapter owns: class encoding (GCM), fill values, tensor reshaping
- Model owns: learnable parameters and the mathematical formula only
- Compute logic (e.g. class vocabulary fitting in GCM) belongs in
  ModelAdapter.fit(), not in the model and not in tensorize utilities

---

## Layer Designs

### DataContract (model side)

The model declares what canonical column names it needs.

```python
contract = RescorlaWagnerModel.data_contract()
contract.required_columns   # ("choice", "reward")
contract.optional_columns   # ("forced",)
```

**Responsibility:** declare canonical column names needed. No dataset
knowledge. No tensor shapes.

---

### ColumnMapping (user side)

The user declares how raw column names map to canonical names. The model
ships a default mapping. The user overrides only what differs.

```python
mapping = ColumnMapping(
    model=RescorlaWagnerModel,
    overrides={
        "choice": "response",
        "reward": "outcome",
        "participant": "subject_id",
    }
)
mapping.resolve()
# {"choice": "response", "reward": "outcome",
#  "forced": "forced", "participant": "subject_id", ...}
```

For variable-column models (e.g. GeneralizedContextModel), the user
declares a pattern rather than fixed names. Pattern matching moves out
of MODEL_COLUMN_PATTERNS in _contracts.py and into ColumnMapping:

```python
mapping = ColumnMapping(
    model=GeneralizedContextModel,
    overrides={"choice": "response", "ground_truth": "label"},
    patterns={"features": r"^stimulus_\d+$"}  # user declares pattern
)
```

**Responsibility:** translate raw column names to canonical names. Nothing
else.

---

### DataAdapter (independent component)

Composable pipeline. Each step is independently callable and inspectable.

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
    .filter(participant=["p01", "p02"])
    .transform()
    .adapt()
)
```

Split is a separate explicit step, not part of DataAdapter. See pipeline
above — splitting happens after AdaptationResult, before ModelAdapter.

**Responsibility:** raw data → canonical DataFrame. Nothing else.

---

### AdaptationResult

Always returned, never raised silently.

```python
result.complete          # True / False
result.canonical_df      # canonical DataFrame or None
result.report()

# Example report:
# Mapped:
#   choice       ← response     ✓
#   reward       ← outcome      ✓
#   forced       ← forced       ✓  (optional)
#   participant  ← subject_id   ✓
#
# Missing required columns:
#   (none)
#
# Unmapped raw columns (ignored):
#   rt, condition, block_type
```

---

### ModelAdapter (model side, independent component)

Fits on training data only. Transforms canonical df → tensors mechanically.

```python
model_adapter = ModelAdapter(model)
model_adapter.fit(train_df)           # fit on train only
train_tensors = model_adapter.transform(train_df)
eval_tensors  = model_adapter.transform(eval_df)
```

ModelAdapter.transform() does only mechanical conversion — no compute
logic, no classification, no statistics. All logic that requires fitting
belongs in ModelAdapter.fit().

**Responsibility:** fit encodings on train df. Convert canonical df →
tensors. Nothing else.

---

## File Structure

```
src/mt/data/
  _loading.py     Load any format → raw DataFrame
                  Supported: CSV, parquet, HuggingFace Dataset, DataFrame
                  Entry: load(source) -> pd.DataFrame

  _contract.py    DataContract — canonical columns the model needs
                  Entry: DataContract.from_model(model) -> DataContract

  _mapping.py     ColumnMapping — raw names → canonical names
                  Supports fixed names and regex patterns
                  Entry: ColumnMapping(model, overrides, patterns)
                         mapping.resolve() -> dict[str, str]

  _adapter.py     DataAdapter — composable pipeline, raw → canonical df
                  Entry: DataAdapter(source, contract, mapping)
                         .load() -> self
                         .validate() -> self
                         .filter(**kwargs) -> self
                         .transform() -> self
                         .adapt() -> AdaptationResult

  _result.py      AdaptationResult — canonical df and mapping report
                  Fields: complete, canonical_df
                  Entry: result.report() -> str

  _split.py       Split canonical df into train/eval
                  Entry: split(df, by, test_ratio,
                               strategy="single") -> SplitResult
                  Note: strategy param reserved for future CV support.
                        Do not implement CV now. Add as new branch later
                        without changing existing interface.

src/mt/models/common/
  _model_adapter.py   ModelAdapter — fit encodings, df → tensors
                      Entry: ModelAdapter(model)
                             .fit(train_df) -> self
                             .transform(df) -> dict[str, Tensor]
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
  test_loading.py      Every supported format loads correctly
  test_contract.py     Contract reflects model declared needs
  test_mapping.py      Overrides resolve against model defaults
                       Pattern matching works for variable columns
                       Missing required columns produce clear errors
  test_adapter.py      Each pipeline step works independently
                       Complete result contains canonical df
                       Partial result reports what is missing
  test_split.py        Train/eval split produces non-overlapping sets
                       Split always precedes ModelAdapter.fit()

tests/models/
  test_model_adapter.py  fit() on train only, transform() on both
                         Encodings fitted on train applied correctly to eval
                         transform() produces tensors only, no logic
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

**Canonical DataFrame schema**
Full definition of required index columns, context columns, and standard
types requires research into cognitive science data standards and existing
community codebases. Deferred. Current principle only: canonical column
names match model contract key names.

**Agent design rules for modularization**
Add to agent-rules.md later: prefer adding new functions over modifying
existing ones. Every component must be independently testable.