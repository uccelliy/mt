# ARCHITECTURE.md

## Project Overview

`mt` is a Python research package for cognitive science modeling. It has two
main concerns:

- Reproducible cognitive model implementations
- Scoring and evaluating language models on behavioral trial transcripts,
  head-to-head against Centaur

Package name: `mt`
Source root: `src/mt`
Python: 3.10
Dependency manager: `uv`

---

## Current Design Status

The current mainline is the Centaur evaluation workstream: LLM-side NLL scoring
on Psych-101 transcripts, plus context-window and finetuning factorization.
`docs/centaur-eval-design.md` is the scientific design and
`docs/centaur-eval-handoff.md` is the running status document.

There is no data-contract layer. The `src/mt/data/` package — canonical field
registry, loading, mapping, `TrialCollection`, `DataAdapter`, and data views —
was removed once the mainline moved to LLM evaluation; nothing outside its own
tests ever consumed it. Datasets are handled ad hoc at the point of use.
Do not reintroduce a generic data-contract abstraction without an explicit task.

| Area | Status |
|---|---|
| Cognitive model formulas | Stable |
| `src/mt/models/llm/` | Active — supervision, backends, LoRA finetuning |
| `src/mt/evaluation/` | Active — transcript scoring, context windows, sequence baselines |
| Model data contracts and preprocessing | Legacy — used only by `Trainer` |
| `src/mt/training/` | Legacy — dataframe-based cognitive-model trainer |
| `src/mt/utils/` | Utilities |
| `scripts/` | Run scripts only, not part of the package |
| `outputs/` | Scoring results and analysis CSVs; gitignored artifacts |
| `tests/` | Mirrors the source package |

---

## Repository Structure

```
src/mt/
  models/
    common/         Base classes and legacy dataframe/tensor interfaces
    cognitive/      Cognitive model formulas and modules
    baselines/      Community baseline implementations
    llm/            LLM backends, marked-text supervision, LoRA finetuning
  evaluation/       Metrics, transcript scoring, context windows, sequence baselines
  training/         Trainer for cognitive models; legacy dataframe interface
  utils/            Token counting, hardware monitoring

scripts/
  experiments/      Scoring runners, preflight checks, figure builders
  *.slurm           HPC job scripts (Minitaur-8B, 4xV100)
  *.ps1             Local CUDA launch scripts (Llama base E0/E3)
outputs/            Scoring CSVs, analysis CSVs, figures
tests/              Pytest suite mirroring src/mt
```

---

## Centaur Evaluation Pipeline

The scoring path is text-first: Psych-101-style transcripts carry `<<...>>`
markers around the tokens a model is scored on. Everything downstream keys off
those spans.

```
Psych-101 transcript text
  → find_target_spans()            locate <<...>> choice positions
  → tokenize + label masking       supervise only marked tokens
  → per-choice NLL                 scoring runners in scripts/experiments/
  → outputs/scoring/*.csv          raw per-choice records
  → analysis CSVs + figures        scripts/experiments/build_*.py
```

`mt.models.llm.supervision` is the single point that owns the marker
convention. It provides:

- `LEFT_TARGET_MARKER` / `RIGHT_TARGET_MARKER` and `find_target_spans()` — the
  span locator shared by every scorer
- `preprocess_marked_text()` — tokenization with label masking outside spans
- `load_marked_text_supervision_dataset()` and `make_lm_collate_fn()` — the
  finetuning data path used by `mt.models.llm.finetuning`
- `format_record_as_marked_text()` — serialize a tabular row into the same
  marked-text convention

Consumers of `find_target_spans()` are `mt.evaluation.context_windows`,
`mt.evaluation.sequence_baselines`, `mt.evaluation.transcript_scoring`, and the
`scripts/experiments/` runners. When the marker convention changes, it changes
in `supervision.py` only.

### Evaluation modules

| Module | Owns |
|---|---|
| `evaluation.transcript_scoring` | Per-choice NLL over a full transcript |
| `evaluation.context_windows` | Truncating transcript history to `w` segments |
| `evaluation.sequence_baselines` | Count-based sequence baselines (E2) |
| `evaluation.metrics` | `choice_nll` and core metrics |
| `evaluation.results` | Structured result containers |

---

## Component Boundaries

| Component | Owns | Does not own |
|---|---|---|
| `models.llm.supervision` | Marker convention, span location, LM tokenization | Scoring policy, model loading |
| `models.llm.backends` | Model/tokenizer loading and quantization | Metrics |
| `evaluation.*` | Metrics and scoring over transcripts | Model weights, plotting |
| `scripts/experiments/` | Run configuration, output CSVs, figures | Reusable library logic |
| `Trainer` | Optimization and evaluation loops for cognitive models | Data preparation |
| Model | Parameters and formula | Data preparation |

Library logic belongs in `src/mt/`; `scripts/` holds run configuration and
output wiring only. If a scoring helper is needed by two scripts, move it into
`src/mt/evaluation/` rather than importing across scripts.

---

## Current Model Implementation

The cognitive model modules are stable formula implementations. Their
data-facing methods remain on the legacy dataframe interface:

- `mt.models.common._contracts` contains the `MODEL_TENSOR_COLUMNS` /
  `ModelDataSpec` dataframe-to-tensor registry.
- `mt.models.common._preprocessing` contains dataframe preprocessing.
- Several model classes implement `preprocess_data()`.
- `Trainer.preprocess_dataframes()` calls the model method.

This interface is not on the current mainline. Leave it as is unless a task
explicitly asks for cognitive-model work.

Legacy saved-model module paths remain protected by `LEGACY_MODEL_MODULES` in
`mt.models.common._base`. Never remove an existing compatibility entry without
an explicit migration plan.

---

## Before Starting Evaluation Work

- Read `docs/centaur-eval-design.md` (scientific design, experiment breakdown
  in §12) and `docs/centaur-eval-handoff.md` (current status, code map in §2).
- Check which experiments are already complete before running anything —
  full-grid scoring runs are expensive.
- Never mix 4-bit runtime results with published BF16 Centaur numbers; keep
  quantized results in their own column.
- New scoring behavior gets a test in `tests/evaluation/`.
