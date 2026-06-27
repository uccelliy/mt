# Handoff Summary
date: 18:29 Sat Jun 27 2026

## What Was Done

All core agent working documents have been written:

| File | Status |
|---|---|
| `agent-rules.md` | Done — sections 8 (data contract) and 9 (off-limits files) still need content |
| `CONVENTIONS.md` | Done |
| `ARCHITECTURE.md` | Done — unstable sections flagged |
| `PROJECT.md` | Done |
| `design-data-contract.md` | Done — ready for implementation |

---

## Key Decisions Already Made

### Project
- Package name: `mt`, source root: `src/mt`, Python 3.10, uv for deps
- Immediate focus: data contract system — nothing else extended until done
- Stable area: `src/mt/models/` only

### Data Contract Design
- DataAdapter produces a Canonical DataFrame, not tensors
- Canonical DataFrame = naming convention only, not a new class
- Column names in canonical df always match model contract key names
- Full canonical schema deferred — needs research into community standards

### Pipeline (in order, must not change)
```
Raw Dataset
  → DataAdapter (load, validate, filter, transform) → AdaptationResult
  → Split (train_df / eval_df) ← split BEFORE ModelAdapter fits
  → ModelAdapter.fit(train_df) then .transform(train_df / eval_df)
  → Trainer.fit(train_tensors) / .evaluate(eval_tensors)
  → Model.compute_logits()
```

### Component Boundaries
- DataAdapter: raw → canonical df. No model knowledge
- ColumnMapping: raw name → canonical name. User-declared, explicit only
- ModelAdapter: fits encodings on train df only, transforms df → tensors
- Model: parameters + compute_logits() only. Nothing else
- Compute logic (e.g. GCM class encoding): ModelAdapter.fit(), not model
- tensorize() = mechanical conversion only, no logic

### Files to Remove / Replace
- `_preparation.py` → `_adapter.py`
- `_prepared.py` → `_result.py`
- `_requests.py` → fold into `_adapter.py`
- `_checking.py` → fold into `.validate()` step
- `_reports.py` → fold into `result.report()`
- `view/` → fold into DataAdapter pipeline
- `_preprocessing.py` → migrate to `_model_adapter.py`
- `BaseCognitiveModel.preprocess_data()` → replaced by ModelAdapter

### ColumnMapping
- User declares explicitly — no automatic inference ever
- Model ships default mapping (raw name = canonical name)
- User overrides only what differs
- Pattern support for variable-column models (e.g. GeneralizedContextModel)
  via `patterns={"features": r"^stimulus_\d+$"}` — logic moves out of
  MODEL_COLUMN_PATTERNS into ColumnMapping

### Split
- Single train/eval split only for now
- strategy="single" parameter reserved — CV added later as new branch
- Split always happens after DataAdapter, before ModelAdapter.fit()

### Conventions
- Line length: 80 characters
- String quotes: double for strings, single for keys
- Comments: above for blocks, inline for quick notes
- Blank lines: one everywhere
- Docstrings: one-liner only if name is not self-explanatory
- Type hints: only where non-obvious
- File order: imports → constants → classes → functions → utils
- Private methods: _single_underscore
- Private module functions: no prefix

### Agent Rules
- Default mode: DISCUSS, never execute unless told explicitly
- One step at a time, stop and report after each step
- Never guess — stop and ask when ambiguous
- Only work in `src/mt/models/` unless explicitly told otherwise
- Never import libraries not in pyproject.toml
- Always use uv add package==version for new deps

---

## What to Do Next

1. Resolve deferred canonical DataFrame schema — research cognitive science
   data standards before implementing DataAdapter
2. Implement data contract system in this order:
   - `_contract.py`
   - `_mapping.py`
   - `_loading.py`
   - `_adapter.py`
   - `_result.py`
   - `_split.py`
   - `_model_adapter.py` in `src/mt/models/common/`
3. Add agent-rules.md sections 8 and 9 (data contract rules, off-limits files)
4. Add modularization rule to agent-rules.md

---

## Files to Load at Start of New Chat

Paste these in order:
1. This handoff summary
2. `design-data-contract.md` (the implementation reference)
3. `agent-rules.md` (behavioral contract)
4. `ARCHITECTURE.md` (current codebase map)