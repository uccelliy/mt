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

The data architecture is under active design and is not ready for
implementation. `docs/design_docs/DataDesign.md` is the working design
document, not a finished implementation specification.

The current repository still contains the previous dataframe contract and
model preprocessing system. That code describes the present implementation,
but it is not the source of truth for the replacement architecture.

| Area | Status |
|---|---|
| Cognitive model formulas | Stable |
| Model data contracts and preprocessing | Legacy — redesign pending |
| `src/mt/data/` | Legacy — replacement architecture in design |
| `src/mt/evaluation/` | Unstable — pending refactor |
| `src/mt/training/` | Unstable — pending refactor |
| `src/mt/cli/` | Unstable — pending refactor |
| `src/mt/utils/` | Unstable — pending refactor |
| `scripts/` | Run scripts only, not part of the package |
| `experiments/` | AI-generated, under review — treat as read-only |
| `tests/` | Mirrors the source package |

Do not extend a legacy data-facing interface unless the task explicitly asks
for migration work. Do not begin implementing the replacement data system
until the overall design and its detailed API specification are approved.

---

## Repository Structure

```
src/mt/
  models/
    common/         Base classes and current legacy data interfaces
    cognitive/      Cognitive model formulas and modules
    baselines/      Community baseline implementations
    llm/            LLM backends
  data/             Current data implementation; redesign in progress
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
  → DataAdapter
  → Canonical TrialCollection
  → data views and Split
  → public ModelAdapter facade
  → model-specific adapter implementation
  → tensors
  → Trainer
  → Model.compute_logits()
```

The exact DataAdapter stage order and method names are not settled. They are
part of the later detailed design pass.

### Canonical Data

Canonical data is governed by a shared canonical vocabulary. There is no
data-side contract tied to a model. Raw dataset fields are mapped to canonical
names before a model is selected.

The canonical container is `TrialCollection`, not a flat canonical DataFrame.
Its working structure has:

- Scalar coordinates used by infrastructure for ordering, filtering, routing,
  and splitting
- Content slots named `task`, `stimulus`, `response`, and `outcome`
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

The canonical vocabulary, its extension rules, and the complete semantics of
each slot are still unresolved. They must be designed before implementation.

Raw rows are not assumed to correspond one-to-one with logical trials. Some
datasets store one trial in multiple rows. The DataAdapter owns structural
adaptation from raw layout to canonical logical units, but the grouping and
assembly design is deferred to the detailed design stage.

### ColumnMapping

`ColumnMapping` is dataset-facing and model-independent. It translates raw
names to names from the canonical vocabulary. It never reads a model contract
and never performs model preprocessing.

Automatic column-name inference is out of scope. Mapping is explicit. Pattern
support may be used for repeated raw fields, but its exact behavior is still a
detailed design decision.

### Model Contract

Each model will have a model-side contract declaring which canonical fields it
requires. The contract does not define canonical names and does not participate
in raw-data mapping. Its declaration and registration mechanism has not yet
been designed.

The current `MODEL_TENSOR_COLUMNS`, `ModelDataSpec`, and dataframe-derived
contract code are legacy implementation details and will be redesigned.

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
| Canonical vocabulary | Shared names and semantics | Raw names, model tensors |
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
- Treat canonical vocabulary design as the current priority.
- Do not assume a raw row equals one logical trial.
- Do not make `ColumnMapping` depend on a model.
- Do not expose model-specific adapters as separate user-facing APIs.
- Do not implement replacement files until the design reaches file, class,
  method, invariant, and failure-semantics level.
