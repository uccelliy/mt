# ARCHITECTURE.md

## Project Overview

`mt` is a Python research package for cognitive science modeling. It has three
main concerns:

- Reproducible cognitive model implementations
- A canonical data representation for cognitive datasets
- Multi-task foundation-model training and evaluation

Package name: `mt`
Source root: `src/mt`
Python: 3.10
Dependency manager: `uv`

---

## Current Design Status

The data architecture is under active design and is being implemented one
approved module at a time. `docs/design_docs/DataDesign.md` is the working
design reference.

The previous data-side dataframe contract has been removed. The repository
still contains the legacy model preprocessing and model data-spec registry;
they are not the source of truth for the replacement architecture.

| Area | Status |
|---|---|
| Cognitive model formulas | Stable |
| Model data contracts and preprocessing | Legacy — redesign pending |
| `src/mt/data/` | First-stage registry through adapter implemented |
| `src/mt/evaluation/` | Unstable — pending refactor |
| `src/mt/training/` | Unstable — pending refactor |
| `src/mt/cli/` | Unstable — pending refactor |
| `src/mt/utils/` | Unstable — pending refactor |
| `scripts/` | Run scripts only, not part of the package |
| `experiments/` | AI-generated, under review — treat as read-only |
| `tests/` | Mirrors the source package |

Do not extend a legacy data-facing interface unless the task explicitly asks
for migration work. Design, implement, and test one replacement data module at
a time; do not begin a module until that module's API is approved.

---

## Repository Structure

```
src/mt/
  models/
    common/         Base classes and current legacy data interfaces
    cognitive/      Cognitive model formulas and modules
    baselines/      Community baseline implementations
    llm/            LLM backends
  data/             Canonical field registry plus legacy modules being replaced
  evaluation/       Metrics and evaluation; pending refactor
  training/         Trainer; pending refactor
  cli/              Entry points; pending refactor
  utils/            Utilities; pending refactor

scripts/            Run and test scripts, not part of the package
experiments/        Under review; do not modify without explicit instruction
tests/              Pytest suite mirroring src/mt
```

---

## Target Data and Model Architecture

The agreed high-level flow is:

```
Raw Dataset
  → load
  → map raw columns to canonical paths
  → apply defaults
  → normalize scalar missing values to None
  → filter
  → validate
  → assemble logical trials
  → DataAdapter result
  → Canonical TrialCollection
  → data views and Split
  → public ModelAdapter facade
  → model-specific adapter implementation
  → tensors
  → Trainer
  → Model.compute_logits()
```

Mapping is a separate stage immediately after loading and before filtering.
The DataAdapter is a reusable one-shot facade over independently testable pure
functions rather than one large transform. `adapt(source, filters=...)` owns
the fixed stage order; advanced callers may compose low-level stages directly.
Stage errors propagate immediately; no failure result is constructed.

### Canonical Data

Canonical data is governed by a shared canonical field registry. There is no
data-side contract tied to a model. Raw dataset fields are mapped to canonical
names before a model is selected.

The canonical container is `TrialCollection`, not a flat canonical DataFrame.
Its working structure has:

- Scalar coordinates used by infrastructure for ordering, filtering, routing,
  and splitting
- Content slots named `task`, `context`, `stimulus`, `response`, and `outcome`
- Canonical keys inside those slots

Current working coordinate rules are:

| Coordinate | Presence rule |
|---|---|
| `participant_id` | Required; no default |
| `trial_index` | Required; no default |
| `session_id` | Defaults to `1` |
| `block_index` | Defaults to `1` |
| `task_name` | Defaults to `1` |
| `condition` | Defaults to `1` |

The `task` slot contains the canonical key `instructions`. Its value defaults
to `None` when instructions are unavailable.

Current content keys are `task.instructions`, `stimulus.ground_truth`,
`stimulus.features`, `response.choice`, `response.rt`, `outcome.reward`, and
`outcome.feedback`; `context` has no key yet. Only `response.choice` is a
required content key. `stimulus.ground_truth` defaults to `None`; a model that
needs it declares that requirement in its model-side contract. A mapping target
must already exist in this registry. New generic keys are added only when an
actual model needs them.

The first implementation supports one raw row per logical trial and rejects a
duplicate trial identity with a clear unsupported-multi-row error. The second
stage adds `OneRowTrialAssembler` and `GroupedTrialAssembler` behind an
assembly strategy without changing mapping, filtering, validation, or the
TrialCollection interface.

### ColumnMapping

`ColumnMapping` is dataset-facing and model-independent. It translates raw
names registered in the canonical field registry. Unmapped fixed fields use
exact canonical-key lookup. Explicit mappings rename before later stages.

Regex patterns use full matches. A named numeric `index` capture determines
numeric ordering; otherwise source column order is retained. Matches stack
along the last axis. Reusing one raw column warns; multiple raw columns cannot
target one scalar field and must use a pattern for multi-column mapping.
Pattern targets are `task.instructions`, `stimulus.ground_truth`,
`stimulus.features`, `response.choice`, and `outcome.feedback`.

### Model Contract

Each model will have a model-side contract declaring which canonical fields it
requires. The contract does not define canonical names and does not participate
in raw-data mapping. Its declaration and registration mechanism has not yet
been designed.

The current `MODEL_TENSOR_COLUMNS` and `ModelDataSpec` registry is a legacy
implementation detail and will be redesigned.

### ModelAdapter

Users see one public `ModelAdapter` interface. Internally it dispatches by model
type to a model-specific adapter implementation.

Model-specific adapters own model-specific encodings, fitted preprocessing
state, fill behavior, reshaping, derived tensor fields, and tensor conversion.
They fit on training data only and apply the fitted state to both training and
evaluation data. The dispatch and registration mechanism is a later detailed
design decision.

### Model

Models own learnable parameters and mathematical formulas. They receive
already-prepared tensors and implement `compute_logits()`. Dataset loading,
canonical mapping, splitting, fitted encodings, and tensor construction remain
outside model classes.

### Split Boundary

Splitting happens after canonical adaptation and before a ModelAdapter is
fitted. This is a fixed boundary: evaluation data must never contribute to a
fitted encoding or preprocessing statistic.

Split operates on canonical coordinates and does not inspect model-specific
tensor representations. The initial implementation is expected to support one
train/evaluation split; cross-validation remains out of scope for now.

---

## Component Boundaries

| Component | Owns | Does not own |
|---|---|---|
| Canonical field registry | Shared names and semantics | Raw names, model tensors |
| `ColumnMapping` | Raw name to canonical name | Model requirements |
| `DataAdapter` | Raw layout to canonical data | Model-specific processing |
| `TrialCollection` | Canonical coordinates and slots | Tensor representation |
| Split/data views | Canonical selection and partitioning | Model adaptation |
| Model contract | Required canonical fields for one model | Raw mapping |
| Public `ModelAdapter` | Stable model-adaptation interface | Model formula |
| Specific adapter | Fitted encoding and tensor construction | Training loop |
| `Trainer` | Optimization and evaluation loops | Raw/canonical adaptation |
| Model | Parameters and formula | Data preparation |

---

## First-Stage Data Modules

The first implementation uses five files. Separation comes from pure function
boundaries, not one file per small operation.

```
src/mt/data/
  _field_registry.py Canonical names, requirements, defaults, path checks
  _loading.py      Supported sources → DataFrame
  _mapping.py      Identity, explicit, and regex column mapping
  _collection.py   TrialCollection, copy, selection, inspection
  _adapter.py      Pipeline facade, pure stage helpers, result/report
```

Pipeline functions return copies and never mutate caller-owned DataFrames or
TrialCollections. Coordinate arrays, slot dictionaries, and numpy-array slot
values are copied; immutable scalar values may be shared safely.

`_assembly.py` is added only in the second stage, when multi-row trials are
implemented. Split and model-side modules are designed later, one module at a
time.

The collection and adapter are implemented against
`docs/design_docs/CollectionDesign.md` and
`docs/design_docs/AdapterDesign.md`. Grouped multi-row assembly is a separate
second-stage design and implementation task.

---

## Current Model Implementation

The current model modules remain useful formula implementations, but their
data-facing methods are scheduled for replacement:

- `mt.models.common._contracts` contains the legacy dataframe/tensor registry.
- `mt.models.common._preprocessing` contains legacy dataframe preprocessing.
- Several model classes still implement `preprocess_data()`.
- `Trainer.preprocess_dataframes()` still calls the legacy model method.

Do not use these interfaces to infer the replacement design. When migration is
approved, preprocessing will move behind the public ModelAdapter boundary and
model classes will retain only configuration, parameters, and formulas.

Legacy saved-model module paths remain protected by `LEGACY_MODEL_MODULES` in
`mt.models.common._base`. Never remove an existing compatibility entry without
an explicit migration plan.

---

## Before Starting Data or Model-Adapter Work

- Read `docs/design_docs/DataDesign.md` and the latest handoff.
- Distinguish confirmed high-level decisions from deferred detailed design.
- Treat canonical field-registry design as the current priority.
- Treat one raw row as one logical trial in the first implementation only.
- Do not make `ColumnMapping` depend on a model.
- Do not expose model-specific adapters as separate user-facing APIs.
- Finish one module's file, class, method, invariant, and test design before
  implementing and testing that module; then continue to the next module.
