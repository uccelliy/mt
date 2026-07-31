# AGENTS.md

A foundation behavioral cognitive model project built around a controlled head-to-head evaluation against Centaur (Binz & Schulz 2025), which encodes cognitive trials as natural language. The current mainline scores LLMs on Psych-101 transcripts to separate what comes from behavioral finetuning from what comes from general pretraining plus in-context learning.

## Setup commands

- Install deps: `uv pip install -e ".[dev]"`
- Test:        `pytest`                          (pytest configured with pythonpath=["src"])
- Lint:        `ruff check`                      (line-length 100, target-version py310)
- Format:      `ruff format`

## Project layout

- `src/mt/models/llm/` — LLM backends, marked-text supervision (`supervision.py`), LoRA finetuning
- `src/mt/models/cognitive/` — 13 cognitive model baselines (formula-first) from the Centaur supplement
- `src/mt/models/baselines/` — classical ML baselines
- `src/mt/evaluation/` — transcript scoring, context windows, sequence baselines, metrics, result containers
- `src/mt/training/` — training loops for cognitive models (legacy dataframe interface, off the current path)
- `src/mt/utils/` — shared utilities
- `scripts/experiments/` — scoring runners, preflight checks, figure builders
- `scripts/*.slurm` — HPC job scripts (Minitaur-8B, 4×V100); `scripts/*.ps1` — local CUDA launch scripts
- `outputs/` — scoring CSVs, analysis CSVs, figures (gitignored artifacts)
- `docs/` — research notes (Obsidian vault); `docs/agents/` is the agent-facing set

## Code style

- Python 3.10 only (`requires-python = ">=3.10,<3.11"`)
- Lint: `ruff check` (line-length 100)
- Type hints preferred on public APIs
- Cognitive models are formula-first: pure equation code in `mt.models.cognitive.formulas`, model modules are thin wrappers that define learnable parameters and call the formula layer
- Cognitive models inherit from `BaseCognitiveModel`; `forward(data) -> compute_logits` is the convention
- `mt.models.llm.supervision` is the single owner of the `<<...>>` marked-text convention — every scorer locates choice positions through `find_target_spans()`, never by re-implementing the marker scan
- There is no generic data-contract layer; handle each dataset at the point of use

## Testing instructions

- Unit tests: `pytest` (configured with pythonpath=["src"])
- Add a test for every new behavior — match the package layout (tests for `src/mt/evaluation/` go in `tests/evaluation/`)
- All tests and `ruff check` must pass before committing

## Research direction (current focus)

**Now — Centaur evaluation, phase one (LLM inference-side scoring only).**
Score Centaur/Minitaur and a matched Llama-3.1-8B **base** control on Psych-101
transcripts, and factor the difference into context length vs. behavioral
finetuning. Headline result so far: at 8B/NF4 the context-gain curves nearly
coincide for `w ≥ 1`; the finetuning advantage is concentrated at trial 0.

- Experiment definitions and status: `docs/centaur-eval-design.md` (§12) and
  `docs/centaur-eval-handoff.md` (§1 status, §2 code map) — **read the handoff
  before starting a run**; the full grid is expensive and mostly cached
- Never mix 4-bit runtime numbers with published BF16 Centaur results; keep
  quantized results in their own column

**Longer term.** Train sequence models at the trial level across **two
complementary cognitive domains** — decision-making (Centaur's Psych-101
territory: bandit, prospect theory, two-step, RL) and basic cognitive abilities
(the user's own data: perception, attention, memory, cognitive control).
Centaur covers the decision layer only; together the two target a complete
"foundation cognitive model" that Centaur does not. Cross-domain transfer
(decision → basic cognition and back, joint training) is the unique
contribution space. Fitting Response Time alongside choice is a further
differentiator — Centaur does not model RT, and RT is a golden signal for
basic-cognition tasks.

## PR & commit conventions

- Branch from the default branch; never push to it directly
- Commit message style: see existing log — `feat:` / `fix:` / `refactor:` / `[Dev]` tags observed
- Keep `AGENTS.md` and `docs/agents/` in sync as the project grows
- Do not commit large checkpoints, training data, scoring outputs, or `.env`-style files

## Security

- Never commit secrets — `.env` and similar belong in `.gitignore`
- Training data and large checkpoints live in gitignored paths
