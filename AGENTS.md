# AGENTS.md

A foundation behavioral cognitive model that trains sequence models on structured trial data across multiple cognitive tasks. The main comparison target is Centaur (Binz & Schulz 2025), which encodes trials as natural language; this project maps structured/tabular data through a shared canonical field registry.

## Setup commands

- Install deps: `uv pip install -e ".[dev]"`
- Test:        `pytest`                          (pytest configured with pythonpath=["src"])
- Lint:        `ruff check`                      (line-length 100, target-version py310)
- Format:      `ruff format`

## Project layout

- `src/mt/data/` — canonical field registry, loading, transforms, splitting, view construction, tensors
- `src/mt/models/cognitive/` — 13 cognitive model baselines (formula-first) from the Centaur supplement
- `src/mt/models/llm/` — LLM adapters for trial data
- `src/mt/models/baselines/` — classical ML baselines
- `src/mt/evaluation/` — core metrics and structured result containers
- `src/mt/training/` — training loops and schedulers
- `src/mt/utils/` — shared utilities
- `experiments/llm/centaur/` — unsupported Centaur reference experiments for future comparison work
- `scripts/` — entry points for ad-hoc runs
- `docs/` — research notes (Obsidian vault)
- `jobs/` — HPC / training job scripts

## Code style

- Python 3.10 only (`requires-python = ">=3.10,<3.11"`)
- Lint: `ruff check` (line-length 100)
- Type hints preferred on public APIs
- Data layer is formula-first: pure equation code in `mt.models.cognitive.formulas`, model modules are thin wrappers that define learnable parameters and call the formula layer
- Cognitive models inherit from `BaseCognitiveModel`; `forward(data) -> compute_logits` is the convention
- The canonical field registry is the source of truth for shared field names
- Structured/tabular data is the canonical input format for both classical baselines and LLM training; no silent free-text fallbacks

## Testing instructions

- Unit tests: `pytest` (configured with pythonpath=["src"])
- Add a test for every new behavior — match the package layout (tests for `src/mt/data/` go in `tests/data/`)
- All tests and `ruff check` must pass before committing

## Research direction (current focus)

- Train sequence models on structured trial data at the trial level, across **two complementary cognitive domains**:
  - **Decision-making** (Centaur's Psych-101 territory: bandit, prospect theory, two-step, RL)
  - **Basic cognitive abilities** (the user's own data: perception, attention, memory, cognitive control, etc.)
- These two domains are **complementary** — centaur covers the decision layer; the user's data covers the basic-cognition layer. Together they target a complete "foundation cognitive model" that centaur does not.
- Head-to-head comparison with Centaur (NL-encoded) on the decision domain; cross-domain transfer (decision → basic cognition, basic cognition → decision, joint training) is the unique contribution space
- Fit Response Time in addition to choice (Centaur does not model RT; this is a TODO in `README.md` — and RT is a golden signal for basic-cognition tasks)
- The canonical field registry in `src/mt/data/` must accommodate both decision-trial schema (choice, reward, RT) and basic-cognition schema (stimulus, response, accuracy, RT, sometimes signal/noise parameters).
- See `docs/` and `experiments/llm/` for working notes and the Centaur baseline

## PR & commit conventions

- Branch from the default branch; never push to it directly
- Commit message style: see existing log — `feat:` / `fix:` / `refactor:` / `[Dev]` tags observed
- Keep `AGENTS.md` and `docs/agents/` in sync as the project grows
- Do not commit large checkpoints, training data, or `.env`-style files

## Security

- Never commit secrets — `.env` and similar belong in `.gitignore`
- Training data and large checkpoints live in gitignored paths
