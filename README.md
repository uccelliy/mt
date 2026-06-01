# a model for multiple cognitive tasks

## Features

## Installation
```bash
conda env create -f environment.yml
conda activate mt_env
```
## Project Structure
mt/
├── analysis/
├── jobs/
│   └── hcp_train_test.sh
├── scripts/
│   └── eval_finetune.py
├── src/
│   ├── mt/
│       ├── data/
│       │   ├── data_provider.py
│       │   └── finetune_dataset.py
│       ├── models/
│       │   ├── finetune/
│       │   │   └── finetune.py
│       │   ├── cog_model/
│       │       ├── base.py
│       │       ├── cog_params.py
│       │       ├── dual_system_model.py
│       │       ├── dunning_kruger_model.py
│       │       ├── rational_model.py
│       │       ├── rational_model_nontorch.py
│       │       └── rescola_wagner_model.py
│       ├── utils/
│       │   └── debug_tool.py
│       └── __init__.py
├── README.md
├── environment.yml
└── pyproject.toml

## Quick Start

## Acknowledgments
