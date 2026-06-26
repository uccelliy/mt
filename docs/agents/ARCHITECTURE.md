# ARCHITECTURE.md

## Project Overview

`mt` is a Python research package for cognitive science modeling. It has three
main concerns:

- Cognitive model implementations (classical formulas as PyTorch modules)
- A data contract standard for cognitive datasets
- LLM fine-tuning pipeline for multi-task cognitive experiments (planned)

Package name: `mt`
Source root: `src/mt`
Python: 3.10
Dependency manager: `uv`

---

## Stability Status

| Area | Status |
|---|---|
| `src/mt/models/` | Stable — safe to work in |
| `src/mt/data/` | In progress — contracts being redesigned |
| `src/mt/evaluation/` | Unstable — pending refactor, do not extend |
| `src/mt/training/` | Unstable — pending refactor, do not extend |
| `src/mt/cli/` | Unstable — pending refactor, do not extend |
| `src/mt/utils/` | Unstable — pending refactor, do not extend |
| `scripts/` | Run scripts only, not part of the package |
| `experiments/` | AI-generated, under review — treat as read-only |
| `tests/` | Mirrors src/mt structure |

**Agent rule: only work in `src/mt/models/` unless explicitly told otherwise.**

---

## Folder Structure

```
src/mt/
  models/           All model implementations  ← STABLE
    common/         Base classes and shared contracts
    cognitive/      Classical cognitive model implementations
    baselines/      Community baseline implementations
    llm/            LLM backends (planned)
  data/             Data loading and contracts  ← IN PROGRESS
  evaluation/       Metrics and evaluation      ← PENDING REFACTOR
  training/         Trainer                     ← PENDING REFACTOR
  cli/              Entry points                ← PENDING REFACTOR
  utils/            Utilities                   ← PENDING REFACTOR

scripts/            Run and test scripts (not part of the package)
experiments/        Under review — do not modify
tests/              Pytest suite
```

---

## Models — Detailed (Stable)

### `mt.models.common`
The foundation all models build on.

- `_base.py` — `BaseCognitiveModel` and `FormulaOnlyCognitiveModel`.
  All cognitive models and baselines must subclass one of these.
  Also contains `LEGACY_MODEL_MODULES` for backward-compatible loading.
- `_contracts.py` — `MODEL_TENSOR_COLUMNS` registry and `ModelDataSpec`.
  Single source of truth for which dataframe columns each model needs.
- `_preprocessing.py` — Shared preprocessing utilities (`df_to_tensors`,
  column matching, encoding helpers). All data conversion lives here.

### `mt.models.cognitive`
Classical cognitive model implementations. Each file is one model:

```
_dual_systems.py
_generalized_context.py
_gp_ucb.py
_hyperbolic_discounting.py
_linear_regression.py
_multitask_reinforcement_learning.py
_odd_one_out.py
_prospect_theory.py
_reference_point.py
_rescorla_wagner.py
_weighted_additive.py
```

Each model: subclasses `BaseCognitiveModel`, 
implements `compute_logits()`, and delegates `preprocess_data()` to a
function in `mt.models.common._preprocessing`.

### `mt.models.baselines`
Community baseline implementations. Same structure as cognitive models.
These are reference implementations — do not optimize them.

```
_lookup_table.py
_lookup_table_dunning.py
_noise_ceiling.py
_rational.py
```

---

## Key Design Decisions (Models)

**Models own only parameters and formula.**
Training, evaluation, preprocessing, and metrics are all external to the
model class. A model file contains exactly:  `__init__`,
and `compute_logits`. Nothing else.

**Data contracts are the single source of truth for column mapping.**
`MODEL_TENSOR_COLUMNS` in `mt.models.common._contracts` is the registry
every model registers against. Adding a new model means adding an entry
here before writing any preprocessing code.

**Preprocessing lives in `mt.models.common._preprocessing`, not in models.**
Each model's `preprocess_data()` calls a named function from `_preprocessing`
rather than implementing logic inline. This keeps model files formula-only.

**Legacy module paths are maintained via `LEGACY_MODEL_MODULES`.**
When a module is moved, its old path is added to `LEGACY_MODEL_MODULES` in
`_base.py` so saved model payloads continue to load correctly. Never delete
old entries from this mapping.

---

## Adding a New Cognitive Model

Follow these steps in order — do not skip or reorder:

1. Add tensor column mapping to `MODEL_TENSOR_COLUMNS` in
   `mt.models.common._contracts`
2. Add preprocessing function to `mt.models.common._preprocessing`
3. Create model file `mt.models.cognitive._<model_name>.py`
   — subclass `BaseCognitiveModel`
   — implement `compute_logits()`
   — implement `preprocess_data()` by calling your preprocessing function
4. Add tests in `tests/models/`

Do not start step 3 before step 1. The contract must exist before the model.

---

## Before You Start Any Task

- Does a preprocessing function already exist for the data shape I need?
  Check `mt.models.common._preprocessing` first.
- Is the model already registered or partially implemented somewhere?
  Check `mt.models._registry` and `LEGACY_MODEL_MODULES` in `_base.py`.
- Does a result type already exist for what I am returning?
  Check `mt.evaluation.results` — but do not extend it until the
  evaluation module is refactored.