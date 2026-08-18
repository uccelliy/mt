# AGENTS.md

A trial-native behavioral cognitive model project built on the project's own P500
data. The current program has three papers: language serialization versus typed
trial state; shared choice, response-time, and nonresponse dynamics; and the
unity/diversity of executive control across a population of language models. The
completed Psych-101/Centaur audit is historical motivation, not the active
experiment program.

## Setup commands

- Install deps: `uv pip install -e ".[dev]"`
- Test:        `pytest`                          (pytest configured with pythonpath=["src"])
- Lint:        `ruff check`                      (line-length 100, target-version py310)
- Format:      `ruff format`

## Project layout

- `src/mt/models/llm/` — LLM backends, marked-text supervision (`supervision.py`), LoRA finetuning
- `src/mt/models/baselines/` — count-based baselines: `sequence.py` (transcript label space, E2), `population.py` (canonical choice space, E2-pop)
- `src/mt/evaluation/` — transcript scoring, context windows
- `src/mt/utils/` — shared utilities (hardware monitoring)
- `scripts/experiments/` — scoring runners, preflight checks, figure builders, `compare_scoring.py` (validates a new environment against a known-good score CSV)
- `scripts/*.slurm` — HPC job scripts (formal Centaur-adapter roster plus legacy validation, 4×V100), all sourcing `scripts/hpc_env.sh`, the single place site-specific settings (modules, `HF_HOME`) live; `scripts/*.ps1` — local CUDA launch scripts
- HPC bring-up runbook: `docs/Server test design.md` — read it before touching `scripts/*.slurm`
- `outputs/` — scoring CSVs, analysis CSVs, figures (gitignored artifacts)
- `docs/` — research notes (Obsidian vault); `docs/agents/` is the agent-facing set

## Code style

- Python 3.10 or 3.11 (`requires-python = ">=3.10,<3.12"`) — 3.11 is what the ULHPC cluster's 2023b toolchain provides; local envs stay on 3.10
- Lint: `ruff check` (line-length 100)
- Type hints preferred on public APIs
- `mt.models.llm.supervision` is the single owner of the `<<...>>` marked-text convention — every scorer locates choice positions through `find_target_spans()`, never by re-implementing the marker scan
- Baselines keep fitting and scoring separate: fit returns state, score is a pure function of that state plus one sequence; held-out participants never reach a fit
- Scientific logic lives in `src/mt/`; `scripts/` owns run configuration, file I/O, and output shaping only (model loading is the deliberate exception, in `_common.load_model`)
- There is no generic data-contract layer and no classical cognitive-model suite; both were removed deliberately — see `docs/agents/ARCHITECTURE.md`

## Testing instructions

- Unit tests: `pytest` (configured with pythonpath=["src"])
- Add a test for every new behavior — match the package layout (tests for `src/mt/evaluation/` go in `tests/evaluation/`)
- All tests and `ruff check` must pass before committing

## Research direction (current focus)

**Now — a three-paper P500 program.** Paper A asks whether compressing native
trial state into language changes inductive bias, data efficiency, and task
transfer. Paper B asks whether choice, RT, and nonresponse contain shared response
dynamics across participants and tasks. Paper C tests whether performance across
many model checkpoints follows the human-inspired unity/diversity structure of
executive control. Reduced history is an optional Paper A diagnostic, not an
independent research line.

- Design of record: `docs/benchmark design.md`; project scope and conclusion
  boundaries: `docs/agents/PROJECT.md`.
- Paper-specific designs: `docs/p500-structured-input-paper.md`,
  `docs/p500-choice-rt-paper.md`, and
  `docs/p500-llm-executive-function-paper.md`.
- Paper A builds text and structured views from the same canonical P500 trial
  state and uses the same candidate-action head, choice targets, splits, and
  matched budgets. Structured input must not mean JSON serialized into tokens.
- Paper A is choice-only: two frozen text renderers versus typed state, with
  data-efficiency, held-out-instrument/family, and targeted architecture
  replication. It owns Trial-State v1 and freezes the structured interface.
- Paper B reuses that frozen interface and compares choice-only, RT/event-only,
  joint choice–RT/event, and shuffled-RT controls. Its claim requires calibrated
  RT/event distributions and shared-dynamics transfer, not merely a choice-NLL
  gain from an auxiliary head.
- Paper C builds a separate no-leak task-execution battery. It treats checkpoint
  as the subject and base lineage as the dependency/generalization cluster;
  prompts, seeds, quantizations, and related fine-tunes are repeated or nested
  measurements, not independent subjects.
- Paper C starts with a construct/interface audit and 6–10-model smoke, then a
  30–50-checkpoint pilot. Do not launch the confirmatory roster until each EF
  construct has at least three independent paradigms and the reliability,
  ceiling/floor, prompt-variance, and Monte Carlo power gates pass.
- Paper C's primary roster requires immutable revisions and a common lossless
  symbolic interface. True-visual VLM tasks are a separate multi-group
  sensitivity; mutable APIs are supplementary only.
- Updating tasks use an auditable stateful streaming protocol: present each
  stimulus once, preserve native model state, and do not replay the transcript
  or provide an external scratchpad/retrieval channel.
- Paper C uses confirmatory MIRT/GLLVM as its main model. A new nonlinear/deep
  generative measurement method is a distant follow-up, not current scope.
- A Paper A short-history condition may remove prior cross-trial human behavior,
  but preserves all task-sufficient state. Any representation×history claim
  crosses the same short/full budgets with both text and structured inputs.
- Papers A/B split by participant, never random rows. Choice NLL and RT log
  predictive density stay separate and aggregate through participant then
  task/family; Paper C validates by lineage.
- P500 RT is incomplete and family-dependent. Paper B freezes per-family timer,
  deadline, no-response, timeout/censoring, zero/null, and sequence-total-RT
  semantics before training; it must not rely on one global missing-value rule.
- Dataset-specific sufficient-state extraction stays explicit by family. Share
  the P500 model-facing trial-state boundary; do not build a generic data layer.
- Never reuse the human-response text renderer as Paper C's test environment:
  some task views contain derived answer information or omit actual visual state.
  LLM wall-clock/API latency is not human response time.

**Psych-101/Centaur is closed.** Its preliminary NF4 result shows that behavioral
finetuning largely fixes cold-start answer-interface behavior and leaves a
smaller residual, but does not identify a cognitive mechanism or training
intervention. Do not run new Psych-101 analyses or resume the roster, E6, Track
S, 70B, or A8–A19 unless the direction is explicitly reopened. Keep only the
frozen design, result snapshot, and matched 8B raw anchor until final archival
checks. Never mix local 4-bit results with published BF16 numbers.

## PR & commit conventions

- Branch from the default branch; never push to it directly
- Commit message style: see existing log — `feat:` / `fix:` / `refactor:` / `[Dev]` tags observed
- Keep `AGENTS.md` and `docs/agents/` in sync as the project grows
- Do not commit large checkpoints, training data, scoring outputs, or `.env`-style files

## Security

- Never commit secrets — `.env` and similar belong in `.gitignore`
- Training data and large checkpoints live in gitignored paths
