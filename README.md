# a model for multiple cognitive tasks

## Features
- **Controlled LLM evaluation on behavioral transcripts** — per-choice NLL scoring
  on Psych-101, with context-window truncation and a matched base-model control
- **Classical cognitive model baselines** — 13 formula-first implementations
- **Count-based sequence baselines** — a non-neural floor for the same tasks

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
|       |   |-- transcript_scoring.py  per-choice NLL
|       |   |-- sequence_baselines.py  count-based baselines
|       |   |-- metrics.py
|       |   `-- results.py
|       |-- models/
|       |   |-- baselines/
|       |   |-- cognitive/
|       |   |-- common/
|       |   `-- llm/
|       |       |-- backends.py        model loading and quantization
|       |       |-- supervision.py     `<<...>>` marked-text convention
|       |       `-- finetuning.py      LoRA finetuning entry point
|       |-- training/
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
