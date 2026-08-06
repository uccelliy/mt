# AGENTS.md

A foundation behavioral cognitive model project built around a controlled head-to-head evaluation against Centaur (Binz & Schulz 2025), which encodes cognitive trials as natural language. The current mainline scores LLMs on Psych-101 transcripts to separate what comes from behavioral finetuning from what comes from general pretraining plus in-context learning.

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

**Now — Centaur evaluation, phase one (LLM inference-side scoring only).**
Score the official Centaur-8B adapter and a matched Llama-3.1-8B **base** control on Psych-101
transcripts, and factor the difference into context length vs. behavioral
finetuning. The merged Minitaur checkpoint is excluded from the formal roster;
its old runs remain only as deployment-damage evidence. Headline result so far:
at 8B/NF4 the historical context-gain curves nearly
coincide for `w ≥ 1`; the finetuning advantage is concentrated at the first
marked-choice segment/anchor.

- Archived experiment definitions and status: `docs/archive/centaur-eval-design.md` (§12) and
  `docs/archive/centaur-eval-handoff.md` (§1 status, §2 code map) — **read the handoff
  before starting a run**; the full grid is expensive and mostly cached
- Never mix 4-bit runtime numbers with published BF16 Centaur results; keep
  quantized results in their own column
- The formal E3 runner subsumes the ordinary full Track P pass: it writes all
  full-context choices once plus the six finite-window anchor conditions into
  the same three raw tables. Submit it through `scripts/submit_roster.sh e3`.
  Its finite-window unit is a syntactic marked-choice text segment, not a
  recovered cognitive trial; Psych-101-test does not provide trial offsets.

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
