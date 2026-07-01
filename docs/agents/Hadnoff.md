# Handoff Summary

Date: 2026-07-01

## Current State

The replacement data layer is being designed, implemented, and tested one
module at a time. `_field_registry.py` and the replacement API in `_loading.py`
are implemented and verified against their detailed designs. `_mapping.py` is
also implemented and verified against `docs/design_docs/MappingDesign.md`.
Review mapping, then discuss `_collection.py`; do not implement collection
before its detailed design is approved. Do not design model or split work while
the current data module is active.

`docs/design_docs/DataDesign.md` is the working design reference and now
contains the decisions below.

---

## Confirmed High-Level Decisions

### Canonical Data

- DataAdapter produces a `TrialCollection`, not a canonical DataFrame and not
  tensors.
- The project needs a shared canonical field registry.
- There is no data-side contract tied to a model.
- Canonical names are defined before datasets and models use them.
- Fixed fields use exact canonical-key lookup when no mapping is supplied.
- Explicit `ColumnMapping` entries rename raw names to canonical paths.
- `ColumnMapping` is independent of every model and model contract.
- Heuristic column-name inference is out of scope.

### TrialCollection

- Infrastructure coordinates and content slots are separate.
- Current coordinates are `participant_id`, `session_id`, `block_index`,
  `trial_index`, `task_name`, and `condition`.
- `participant_id` and `trial_index` are required.
- `session_id`, `block_index`, `task_name`, and `condition` default to `1`.
- Content slots are `task`, `context`, `stimulus`, `response`, and `outcome`.
- The `task` slot uses canonical key `instructions`, defaulting to `None`.
- `response.choice` is the only required content key.
- `stimulus.ground_truth` defaults to `None`; models that need it declare it in
  their model-side contract.
- `stimulus` and `context` may both be empty for reinforcement-learning and
  decision datasets without an explicit stimulus field.
- Slot values may be strings, integers, floats, booleans, `None`, or numpy
  arrays.
- DataFrames, coordinate arrays, slot dicts, and numpy-array values are copied
  when returning new pipeline values; caller-owned inputs are not mutated.
- Stage one supports exactly one raw row per logical trial and rejects duplicate
  trial identities.
- Stage two adds grouped multi-row trials behind an assembly strategy.

### Column Patterns and Collisions

- Regex uses fullmatch.
- A named numeric `index` capture sorts matches numerically.
- Patterns without that capture preserve raw DataFrame column order.
- Matched values stack along the last axis; they are not summed or averaged.
- An explicit pattern with no matches is an error.
- Pattern targets are `task.instructions`, `stimulus.ground_truth`,
  `stimulus.features`, `response.choice`, and `outcome.feedback`.
- Reusing one raw column for multiple canonical paths warns.
- Multiple raw columns cannot target one scalar canonical path; the error tells
  the user to declare a pattern for multi-column mapping.

### Model Side

- Each model will declare a model-side contract containing the canonical fields
  it requires.
- The model contract does not define the canonical field registry and does not
  participate in raw-data mapping.
- Existing model contracts and `preprocess_data()` implementations are legacy
  and will be redesigned.
- Models own learnable parameters and mathematical formulas.

### ModelAdapter

- Users see one public `ModelAdapter` interface.
- Internally, ModelAdapter dispatches to a model-specific implementation.
- Model-specific adapters own fitted encodings, fill behavior, reshaping,
  derived tensor fields, and tensor conversion.
- ModelAdapter fits on training data only and applies that state to both train
  and evaluation data.
- The internal registry, class hierarchy, and failure behavior are deferred to
  detailed design.

### Pipeline Boundaries

The agreed overall flow is:

```
Raw Dataset
  → load
  → map
  → defaults
  → filter
  → validate
  → assemble one-row trials
  → DataAdapter result
  → Canonical TrialCollection
  → data views and Split
  → public ModelAdapter
  → model-specific adapter
  → tensors
  → Trainer
  → Model.compute_logits()
```

Split always happens before ModelAdapter fits. Evaluation data must not affect
fitted preprocessing state.

DataAdapter is a facade over independently testable pure functions. Mapping is
not part of one large transform.

---

## Current Working Details from DataDesign

- First-stage files are `_field_registry.py`, `_loading.py`, `_mapping.py`,
  `_collection.py`, and `_adapter.py`.
- `_adapter.py` contains the facade, result/report, and small pure helpers for
  defaults, filtering, validation, and one-row assembly.
- `_assembly.py` is created only in stage two for grouped multi-row trials.
- Supported source targets are currently CSV, parquet, HuggingFace Dataset, and
  pandas DataFrame.
- Loading normalizes every raw column label to a string, maps missing labels to
  `"None"`, and rejects duplicates after normalization. Mapping verifies
  uniqueness but does not repeat label conversion.
- `AdaptationResult` is intended to carry the collection, completion status,
  and an inspectable report.
- `TrialCollection.to_dataframe()` is a debugging tool, not a pipeline step.
- The initial split is a single train/evaluation split; cross-validation is
  deferred.
- HuggingFace Dataset export is deferred until the foundation-model pipeline.
- A future ModelAdapter representation `mode` is reserved for ablations, but
  its API is not yet designed.

These are working details. A module may be implemented only after that module's
API, invariants, errors, and tests are approved.

---

## Superseded Assumptions

Do not carry these older assumptions forward:

- DataAdapter produces a canonical DataFrame.
- Canonical naming is derived from a selected model.
- `ColumnMapping` accepts or inspects a model.
- DataAdapter obtains a data-side contract through
  `DataContract.from_model(model)`.
- One generic implementation performs every model's adaptation logic.
- The final architecture must always treat one raw row as one logical trial.
- Current `MODEL_TENSOR_COLUMNS` and `_preprocessing.py` define the future
  architecture.

---

## Module-by-Module Order

For each module: discuss its API and invariants, update the design, implement
it, run focused tests and lint, then proceed.

1. `_field_registry.py`
2. `_loading.py`
3. `_mapping.py`
4. `_collection.py`
5. `_adapter.py`
6. `_assembly.py` for multi-row trials

Split and model-side design remain deferred until these data modules are done.

---

## Existing Code Status

- Cognitive formula implementations remain usable.
- The old data-side `DataContract`, its checker, and contract-specific loading
  helpers have been removed.
- `src/mt/data/_field_registry.py` and the replacement `load()` API in
  `_loading.py` are implemented. The old `load_dataframe()` and
  `load_hf_dataset()` APIs and exports have been removed; current CSV and
  parquet callers use `load()`.
- `_preparation.py` now calls `load()` and checks its own `required_columns`,
  but remains a temporary legacy module. When `_adapter.py` is implemented,
  migrate or remove preparation behavior and tests rather than treating this
  bridge as part of the replacement architecture.
- LLM supervision owns its local JSON and JSONL reading; these formats are not
  supported by the shared replacement loader.
- `src/mt/data/_mapping.py` now owns immutable `ColumnMapping` configuration,
  inspectable `MappingResolution`, identity/fixed/pattern resolution, pattern
  stacking, ignored-column reporting, and reused-source warnings.
- `src/mt/models/common/_contracts.py` is a legacy dataframe/tensor registry.
- `src/mt/models/common/_preprocessing.py` is legacy preprocessing.
- Model `preprocess_data()` methods and
  `Trainer.preprocess_dataframes()` are legacy boundaries.
- Evaluation, training, CLI, and utilities remain pending refactor.

Do not edit legacy code merely to make it resemble the unfinished design.

---

## Documentation Still to Synchronize

- Update `agent-rules.md` data rules after the design becomes authoritative.
- Remove legacy `preprocess_data()` instructions from agent documentation during
  the approved model-adapter migration.
- Fill the off-limits-files section in `agent-rules.md` when scope
  boundaries are finalized.

---

## Files to Read at the Start of the Next Design Session

1. `docs/agents/PROJECT.md`
2. `docs/agents/ARCHITECTURE.md`
3. `docs/agents/CONVENTIONS.md`
4. `docs/agents/Hadnoff.md`
5. `docs/design_docs/DataDesign.md`
6. `docs/design_docs/FieldRegistryDesign.md`
7. `docs/design_docs/LoadingDesign.md`
8. `docs/design_docs/MappingDesign.md`
9. `docs/agents/agent-rules.md`
