# a model for multiple cognitive tasks

## Features
- **Controlled LLM evaluation on behavioral transcripts** — per-choice NLL scoring
  on Psych-101, with context-window truncation and a matched base-model control
- **Count-based baselines** — the non-neural floor, in both transcript label
  space (within-session) and canonical choice space (pooled across participants)

## PLAN
- **Factor LLM behavioral fit** into context length vs. behavioral finetuning
- **Using tabular data to fine-tune LLM**
- **Fit Response Time**

## Installation
```bash
uv pip install -e ".[dev]"
```

## Project Structure
```text
mt/
|-- docs/
|   |-- agents/                  agent-facing project docs
|   |-- centaur-eval-design.md   scientific design
|   `-- centaur-eval-handoff.md  running status and code map
|-- scripts/
|   |-- experiments/             scoring runners, preflight, figure builders
|   |-- *.slurm                  HPC job scripts
|   `-- *.ps1                    local CUDA launch scripts
|-- outputs/                     scoring CSVs, analysis CSVs, figures
|-- src/
|   `-- mt/
|       |-- evaluation/
|       |   |-- context_windows.py     transcript history truncation
|       |   `-- transcript_scoring.py  per-choice NLL
|       |-- models/
|       |   |-- baselines/
|       |   |   |-- sequence.py        within-session counts (E2)
|       |   |   `-- population.py      pooled canonical counts (E2-pop)
|       |   `-- llm/
|       |       |-- supervision.py     `<<...>>` marked-text convention
|       |       |-- finetuning.py      LoRA finetuning entry point
|       |       `-- finetune_artifacts.py
|       `-- utils/
|-- tests/
|-- README.md
`-- pyproject.toml
```

## Quick Start

```bash
pytest                    # test suite
ruff check                # lint
mt-finetune-llm --help    # LoRA finetuning entry point
```

Scoring runs are driven from `scripts/experiments/`. Check
`docs/centaur-eval-handoff.md` for which experiments are already complete
before launching a run — the full grid is expensive.

## Acknowledgments
