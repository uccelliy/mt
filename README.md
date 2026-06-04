# a model for multiple cognitive tasks

## Features

## Installation
```bash
conda env create -f environment.yml
conda activate mt_env
```
## Project Structure
```text
mt/
├── analysis/
│   ├── OOD_generalization/
│   ├── baseline_model_compare/
│   ├── neural_alignment/
│   ├── picture/
├── docs/
├── jobs/
├── scripts/
├── src/
│   ├── mt/
│   │   ├── data/
│   │   ├── evaluation/
│   │   ├── models/
│   │   │   ├── centaur/
│   │   │   ├── cog_model/
│   │   │   │   ├── cog_params.py
│   │   │   │   ├── dual_system_model.py
│   │   │   │   ├── dunning_kruger_model.py
│   │   │   │   ├── preprocessing.py
│   │   │   │   ├── rational_model.py
│   │   │   │   └── rescola_wagner_model.py
│   │   │   ├── finetune/
│   │   │   ├── __init__.py
│   │   │   ├── llm_backend.py
│   │   │   ├── noiseceiling.py
│   │   │   └── trainers.py
│   │   ├── utils/
│   │   └── __init__.py
│   ├── mt.egg-info/
├── README.md
├── environment.yml
└── pyproject.toml

```
## Quick Start

## Acknowledgments
