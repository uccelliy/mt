# mt

`mt` is a trial-native model of human behavior across P500 cognitive tasks. The
current research program separates representation learning from shared
choice–time dynamics and cross-model executive-function measurement into three
papers.

## Current program

- **Paper A — Trial-native state vs. language serialization:** does compressing
  structured trials into text change choice learning, data efficiency, and task
  transfer?
- **Paper B — Shared response dynamics:** can one model jointly explain choice,
  response time, and nonresponse, and transfer timing structure across tasks?
- **Paper C — Unity and diversity of executive control:** do performance
  differences across checkpoints, while accounting for base lineage, favor a
  unitary, original three-component, or revised Common-EF structure, and do the
  resulting factors predict external interactive abilities?

The current design is [docs/benchmark design.md](docs/benchmark%20design.md).
Paper-level designs are
[structured input](docs/p500-structured-input-paper.md),
[choice–RT dynamics](docs/p500-choice-rt-paper.md), and
[LLM executive control](docs/p500-llm-executive-function-paper.md).
The earlier Psych-101/Centaur evaluation is frozen under `docs/archive/`; no new
Psych-101 analyses are part of the mainline.

## Data and model boundary

P500 currently covers 100 participants, 21 task families, 62 instruments, and
about 777k trials. Before formal training, the project is building a versioned
trial-state dataset with participant-safe splits, typed stimuli and actions,
task-sufficient state, stable trial IDs, and explicit RT observation masks.

Paper A uses a shared candidate-action head for text and structured views and is
choice-only. Paper B freezes the resulting structured interface, then models
choice, RT, and response events jointly. Current-trial RT is a target, never an
input; choice and RT are evaluated separately.

Paper C uses a separate no-leak, agent-facing P500 battery. Model checkpoints are
the measured units and base lineages are dependency clusters; prompt or sampling
repeats estimate reliability rather than increase the model sample size.

## Installation

```bash
uv pip install -e ".[dev]"
```

## Checks

```bash
pytest
ruff check
```

## Project layout

```text
docs/                  current design, project notes, and frozen research
src/mt/models/         model, supervision, and baseline code
src/mt/evaluation/     evaluation and aggregation logic
scripts/               run configuration and artifact builders
outputs/               local runs and derived results
tests/                 unit tests
```

Raw private data, large derived tables, checkpoints, and run artifacts remain
gitignored. Converter code, schemas, QA tests, and compact summaries should be
tracked.
