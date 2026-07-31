# ARCHITECTURE.md

## Project Overview

`mt` is a Python research package with one concern: scoring language models on
behavioral trial transcripts under a controlled protocol, head-to-head against
Centaur and against count-based baselines.

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

Two subsystems were removed deliberately; do not reintroduce either without an
explicit task:

- **The data-contract layer** (`src/mt/data/` — canonical field registry,
  loading, mapping, `TrialCollection`, `DataAdapter`, data views). Nothing
  outside its own tests ever consumed it. Datasets are handled ad hoc at the
  point of use.
- **The classical cognitive models** (`src/mt/models/{cognitive,common}/`,
  `src/mt/training/`, and the dataframe model contracts). Never wired into the
  evaluation path. The count-based baselines fill the "non-neural floor" role.

Every module now in the tree is reachable from an entry point.

| Area | Status |
|---|---|
| `src/mt/models/llm/` | Active — marked-text supervision, LoRA finetuning |
| `src/mt/evaluation/` | Active — transcript scoring, context windows |
| `src/mt/models/baselines/` | Active — count-based sequence and population baselines |
| `src/mt/utils/` | Utilities — hardware monitoring |
| `scripts/` | Run scripts only, not part of the package |
| `outputs/` | Scoring results and analysis CSVs; gitignored artifacts |
| `tests/` | Mirrors the source package |

---

## Repository Structure

```
src/mt/
  models/
    baselines/      Count-based baselines: sequence.py, population.py
    llm/            Marked-text supervision, LoRA finetuning
  evaluation/       Transcript scoring, context windows
  utils/            Hardware monitoring

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
`mt.models.baselines.sequence`, `mt.evaluation.transcript_scoring`, and the
`scripts/experiments/` runners. When the marker convention changes, it changes
in `supervision.py` only.

### Evaluation modules

| Module | Owns |
|---|---|
| `evaluation.transcript_scoring` | Per-choice NLL over a full transcript |
| `evaluation.context_windows` | Truncating transcript history to `w` segments |
| `models.baselines.sequence` | Within-session count baselines, transcript label space (E2) |
| `models.baselines.population` | Pooled count baselines, canonical choice space (E2-pop) |

The two baseline modules exist separately because Psych-101 randomizes response
keys per participant: counts pooled across participants are meaningless in raw
transcript label space, and only become meaningful in the canonical space of
the original HF tables.

---

## Component Boundaries

| Component | Owns | Does not own |
|---|---|---|
| `models.llm.supervision` | Marker convention, span location, LM tokenization | Scoring policy, model loading |
| `models.llm.finetuning` | LoRA finetuning entry point | Scoring |
| `models.baselines.*` | Count fitting and per-choice baseline NLL | File I/O, participant alignment |
| `evaluation.*` | Scoring LM transcripts | Model loading, plotting |
| `scripts/experiments/` | Run configuration, model loading, output CSVs, figures | Reusable scientific logic |

Library logic belongs in `src/mt/`; `scripts/` holds run configuration and
output wiring only. If a scientific helper is needed by two scripts, move it
into `src/mt/` rather than importing across scripts. Model loading is the one
deliberate exception: `scripts/experiments/_common.load_model` owns it, because
quantization and adapter placement are run configuration.

---

## Before Starting Evaluation Work

- Read `docs/centaur-eval-design.md` (scientific design, experiment breakdown
  in §12) and `docs/centaur-eval-handoff.md` (current status, code map in §2).
- Check which experiments are already complete before running anything —
  full-grid scoring runs are expensive.
- Never mix 4-bit runtime results with published BF16 Centaur numbers; keep
  quantized results in their own column.
- New scoring behavior gets a test mirroring its package path under `tests/`.
