# Refactoring Progress Report

## Project Goal

This project started from the Centaur paper codebase, but the current goal is not to fully reproduce every analysis in the original paper. The goal is to build a modular evaluation framework for cognitive and behavioral sequence models.

The intended framework should allow different model families to be evaluated under a shared data pipeline, participant split, metric definition, and result logging protocol:

- Centaur and other LLM-based models
- Classical cognitive models
- Noise ceiling / baseline models
- Future custom behavioral sequence models

The immediate milestone is to make classical cognitive models runnable from raw data loading to k-fold evaluation, while keeping the code structure simple enough for research iteration.

## Design Principle

The current refactor follows one main dependency rule:

```text
analysis scripts
    import and call
src/mt library code
```

The reusable code under `src/mt` should not depend on the repo-level `analysis` scripts. Analysis scripts define specific experimental protocols, while `src/mt` provides reusable loading, splitting, model, preprocessing, and evaluation utilities.

## Current Module Structure

```text
src/mt/
  data/
    loading.py
    splitting.py
    finetune_dataset.py

  models/
    cog_model/
      preprocessing.py
      rational_model.py
      rescola_wagner_model.py
      dual_system_model.py
      dunning_kruger_model.py
      cog_params.py
    centaur/
    finetune/
    llm_backend.py
    trainers.py
    noiseceiling.py

  evaluation/
    metrics.py
    runners.py
    results.py

analysis/
  baseline_model_compare/
    rational_model_compare.py
    rescola_wagner_model_compare.py
    ceiling.py
```

## Data Layer

The data layer has been separated into loading and splitting.

### `mt.data.loading`

`loading.py` currently provides:

- `load_dataframe(path, columns=None)`
- `load_hf_dataset(source, split, columns, **kwargs)`

For cognitive models, the pipeline currently uses pandas DataFrames. `load_dataframe` supports local tabular files and Hugging Face parquet paths, and automatically keeps default metadata columns:

```python
DEFAULT_COLUMNS = ["participant", "task", "trial"]
```

This allows each cognitive model to declare only the task/model-specific columns it needs, for example:

```python
RationalModel.required_columns = ["choice", "ground_truth"]
RescorlaWagnerModel.required_columns = ["reward", "choice"]
```

Then analysis code can read only the needed columns:

```python
df = load_dataframe(path, RationalModel.required_columns)
```

### `mt.data.splitting`

`splitting.py` currently provides participant-level k-fold splitting:

```python
for train_df, eval_df in split_data_kfold(df, num_splits=10):
    ...
```

The split is based on held-out participants, matching the structure used in the original Centaur evaluation and the earlier reproduction scripts.

## Cognitive Model Preprocessing

Model-specific dataframe-to-tensor conversion has been moved out of the generic data layer and into:

```text
src/mt/models/cog_model/preprocessing.py
```

This file contains:

- `pd_to_pth`
- `preprocess_rational_data`
- `preprocess_rescorla_wagner_data`
- `preprocess_dual_system_data`
- `preprocess_dunning_kruger_data`

The reason for this move is that these functions are not generic data loading utilities. They encode model-specific assumptions, such as:

- Rescorla-Wagner `choice_for_updating`
- Optional `forced` trial masking
- Dual-system step-1 / step-2 reshaping
- Dunning-Kruger trial-specific normalization
- Rational model `ground_truth` to choice-distribution lookup

The cognitive model files now import preprocessing functions from `mt.models.cog_model.preprocessing`, not from `mt.data`.

## Model Layer

Each cognitive model now exposes a minimal common pattern:

```python
class SomeModel(nn.Module):
    required_columns = [...]

    def preprocess_data(self, train_df, eval_df):
        ...

    def forward(self, data):
        ...
```

This is not yet a final unified model interface. It is a practical intermediate interface that supports the current goal: running classical cognitive models through the same loading and splitting pipeline.

Current cognitive models include:

- `RationalModel`
- `RescorlaWagnerModel`
- `DualSystemsModel`
- `DunningKruger`

## Baseline Compare Scripts

Two baseline comparison scripts have been added or updated under:

```text
analysis/baseline_model_compare/
```

### Rational model

```text
analysis/baseline_model_compare/rational_model_compare.py
```

This script loads the digit span dataset:

```text
hf://datasets/marcelbinz/enkavi2019digitspan/exp1/train-00000-of-00001.parquet
```

It then runs:

```text
load_dataframe
-> split_data_kfold
-> Trainer.fit_and_evaluate
-> fold-level NLL
```

### Rescorla-Wagner model

```text
analysis/baseline_model_compare/rescola_wagner_model_compare.py
```

This script loads the Feng et al. dynamics dataset:

```text
hf://datasets/marcelbinz/feng2021dynamics/exp1/train-00000-of-00001.parquet
```

It then evaluates `RescorlaWagnerModel` across participant folds.

These scripts show that the new data/model path can already support real cognitive baseline evaluation.

## Smoke Test

A lightweight synthetic smoke test was added:

```text
scripts/smoke_cog_pipeline.py
```

It tests the core cognitive-model pipeline without needing network access:

```text
synthetic dataframe
-> load_dataframe
-> split_data_kfold
-> model.preprocess_data
-> model.forward
-> finite cross-entropy check
```

It currently tests `RescorlaWagnerModel` in two settings:

- Standard `choice` / `reward` bandit data
- Bandit data with a `forced` column, verifying that forced trials are masked with `ignore_index`

The smoke test passed in the project environment:

```bash
conda run -n mt_env python scripts/smoke_cog_pipeline.py
```

Output:

```text
cognitive pipeline smoke test passed
```

## LLM / Centaur Backend Cleanup

The original Centaur scripts used direct `unsloth` imports, which caused problems on Mac or CPU-only environments.

A new helper was added:

```text
src/mt/models/llm_backend.py
```

It provides:

- `load_causal_lm`
- `get_is_quantized`
- `set_is_quantized`

The loader chooses Unsloth only when CUDA and the optional dependency are available. Otherwise it falls back to Hugging Face Transformers. This removes repeated platform checks from Centaur-related scripts and keeps the code importable on Mac.

## Current Validation

The following checks have passed:

```bash
conda run -n mt_env python scripts/smoke_cog_pipeline.py
conda run -n mt_env python -m compileall -q src/mt scripts
conda run -n mt_env python -m compileall -q analysis
```

The analysis compile step produced only non-blocking string escape warnings in a plotting script.

`pytest` was not run because `pytest` is not currently installed in `mt_env`.

## Known Issues

### Trainer is still temporary

`src/mt/models/trainers.py` still contains a simple `Trainer` class that performs both fitting and evaluation:

```text
preprocess_data
-> optimize cross entropy on train data
-> compute cross entropy on eval data
```

This is sufficient for the current baseline scripts but should eventually move out of `mt.models` into an evaluation or runner layer.

### Dunning-Kruger preprocessing needs cleanup

The Dunning-Kruger preprocessing currently filters out `trial == 24`, but the normalizer still has length 28. This can create a tensor length mismatch on synthetic checks. The trial filtering and normalizer definition need to be aligned.

### Result logging is not yet standardized

Current comparison scripts print fold losses and aggregate manually. A structured result object or logger is still needed.

### Model registry is not yet implemented

The code does not yet support:

```bash
--model rational
--model rescorla_wagner
```

This should be added after the runner/evaluation boundary is clearer.

## Next Steps

The next engineering steps should be incremental:

1. Finish stabilizing all cognitive model preprocess/forward paths with small synthetic smoke tests.
2. Move `Trainer` into a more accurate evaluation runner, for example `FitEvaluateRunner`.
3. Make the runner return structured metrics instead of only a loss tensor.
4. Add a small model registry so analysis scripts can select models by name.
5. Standardize result logging for fold-level and task-level metrics.
6. Later, define an adapter interface for LLM/Centaur and future behavioral sequence models.

The short-term target is:

```text
raw dataset path
-> load_dataframe
-> participant k-fold split
-> model-specific preprocessing
-> fit/evaluate
-> fold-level NLL
```

Once this is stable for a few classical cognitive models, the same structure can be extended to Centaur, ordinary LLMs, and the future custom behavioral sequence model.
