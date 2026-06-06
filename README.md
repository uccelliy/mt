# a model for multiple cognitive tasks

## Features
- **A shared data process protocol for cognitive model**
- **Classical cognitive model baselines**
- **TODO**
## PLAN
- **A shared tabular data process protocol and a tabular data contract**
- **Using tabular data to fine-tune LLM**
- **Fit Response Time**

## Installation
```bash
pip install -e ".[dev]"
```
## Project Structure
```text
mt/
|-- docs/
|-- experiments/
|   `-- llm/
|-- scripts/
|-- src/
|   `-- mt/
|       |-- data/
|       |   |-- contracts.py
|       |   |-- filtering.py
|       |   |-- loading.py
|       |   |-- preparation.py
|       |   |-- requests.py
|       |   |-- splitting.py
|       |   |-- transforms.py
|       |   `-- views.py
|       |-- evaluation/
|       |   |-- baseline_comparison/
|       |   |-- generalization/
|       |   |-- metrics.py
|       |   |-- neural_alignment/
|       |   |-- results.py
|       |   |-- runners.py
|       |   |-- specs.py
|       |   `-- visualization/
|       |-- models/
|       |   |-- baselines/
|       |   |-- cognitive/
|       |   `-- llm/
|       |-- training/
|       `-- utils/
|-- README.md
`-- pyproject.toml

```
## Quick Start

## Acknowledgments
