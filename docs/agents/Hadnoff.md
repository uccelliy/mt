# Handoff Summary

Date: 2026-06-29

## Current State

The replacement data architecture is still being designed. It is not ready for
implementation. The immediate priority is the overall architecture, especially
the canonical vocabulary and canonical container. File names, class layouts,
method signatures, pipeline method names, and implementation logic come later.

`docs/design_docs/DataDesign.md` is the working design document. It has been
updated with coordinate presence rules and task-instruction semantics, but some
of its model-derived `DataContract` and `ColumnMapping(model=...)` wording is
superseded by the latest decisions below. Reconcile that document in a future
design pass; do not implement the superseded sections.

---

## Confirmed High-Level Decisions

### Canonical Data

- DataAdapter produces a `TrialCollection`, not a canonical DataFrame and not
  tensors.
- The project needs a shared canonical vocabulary.
- There is no data-side contract tied to a model.
- Canonical names are defined before datasets and models use them.
- `ColumnMapping` explicitly maps raw names to canonical names.
- `ColumnMapping` is independent of every model and model contract.
- Automatic column-name inference is out of scope.

### TrialCollection

- Infrastructure coordinates and content slots are separate.
- Current coordinates are `participant_id`, `session_id`, `block_index`,
  `trial_index`, `task_name`, and `condition`.
- `participant_id` and `trial_index` are required.
- `session_id`, `block_index`, `task_name`, and `condition` default to `1`.
- Content slots are `task`, `stimulus`, `response`, and `outcome`.
- The `task` slot uses canonical key `instructions`, defaulting to `None`.
- Raw rows do not necessarily correspond one-to-one with logical trials.
- DataAdapter owns raw-layout to canonical-structure adaptation.
- The concrete multi-row trial grouping design is deferred.

### Model Side

- Each model will declare a model-side contract containing the canonical fields
  it requires.
- The model contract does not define canonical vocabulary and does not
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
  → DataAdapter
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

---

## Current Working Details from DataDesign

- Supported source targets are currently CSV, parquet, HuggingFace Dataset, and
  pandas DataFrame.
- `AdaptationResult` is intended to carry the collection, completion status,
  and an inspectable report.
- `TrialCollection.to_dataframe()` is a debugging tool, not a pipeline step.
- The initial split is a single train/evaluation split; cross-validation is
  deferred.
- HuggingFace Dataset export is deferred until the foundation-model pipeline.
- A future ModelAdapter representation `mode` is reserved for ablations, but
  its API is not yet designed.

These are working details, not permission to begin implementation while the
overall architecture remains open.

---

## Superseded Assumptions

Do not carry these older assumptions forward:

- DataAdapter produces a canonical DataFrame.
- Canonical naming is derived from a selected model.
- `ColumnMapping` accepts or inspects a model.
- DataAdapter obtains a data-side contract through
  `DataContract.from_model(model)`.
- One generic implementation performs every model's adaptation logic.
- A raw dataframe row always represents one logical trial.
- Current `MODEL_TENSOR_COLUMNS` and `_preprocessing.py` define the future
  architecture.

---

## Unresolved Overall Design

Resolve these before detailed file and API design:

1. Define the canonical vocabulary for decision-making and basic cognition.
2. Define extension rules for task-specific canonical fields.
3. Finalize the semantic boundary of a canonical logical unit without assuming
   raw row cardinality.
4. Decide how heterogeneous tasks coexist in one TrialCollection.
5. Refine the high-level DataAdapter, data-view, Split, and ModelAdapter flow.

After the overall design is approved, specify:

- Required files and module responsibilities
- Classes, fields, methods, and public exports
- State transitions and invariants
- Error versus result-report semantics
- Model-adapter dispatch and fitted-state serialization
- Migration from legacy data and model preprocessing APIs
- Required tests and acceptance criteria

---

## Existing Code Status

- Cognitive formula implementations remain usable.
- `src/mt/data/` is the legacy implementation to be replaced after design.
- `src/mt/models/common/_contracts.py` is a legacy dataframe/tensor registry.
- `src/mt/models/common/_preprocessing.py` is legacy preprocessing.
- Model `preprocess_data()` methods and
  `Trainer.preprocess_dataframes()` are legacy boundaries.
- Evaluation, training, CLI, and utilities remain pending refactor.

Do not edit legacy code merely to make it resemble the unfinished design.

---

## Documentation Still to Synchronize

- Revise `DataDesign.md` after the canonical vocabulary and overall pipeline are
  settled.
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
6. `docs/agents/agent-rules.md`
