from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_generalized_context.py"
SPEC = importlib.util.spec_from_file_location("run_generalized_context", SCRIPT_PATH)
assert SPEC is not None
assert SPEC.loader is not None
run_generalized_context = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_generalized_context)

build_dataset_metadata = run_generalized_context.build_dataset_metadata
build_fit_metadata = run_generalized_context.build_fit_metadata
build_fold_metadata = run_generalized_context.build_fold_metadata
build_run_metadata = run_generalized_context.build_run_metadata


def test_build_dataset_metadata_records_source_context() -> None:
    metadata = build_dataset_metadata(
        dataset="demo",
        path=Path("/tmp/demo.csv"),
        rows_before_sequence_filter=12,
        label_column="correct_choice",
    )

    assert metadata == {
        "dataset": "demo",
        "path": "/tmp/demo.csv",
        "rows_before_sequence_filter": 12,
        "label_column": "correct_choice",
    }


def test_build_fold_metadata_keeps_only_fold_context() -> None:
    metadata = build_fold_metadata(
        fold=1,
        train_rows=9,
        eval_rows=3,
    )

    assert metadata == {
        "fold": 1,
        "train_rows": 9,
        "eval_rows": 3,
    }


def test_build_run_metadata_records_training_context_once() -> None:
    args = argparse.Namespace(
        data_dir=Path("/tmp/data"),
        datasets=("demo",),
        num_splits=10,
        max_folds=2,
        num_iter=100,
        lr=0.01,
        drop_incomplete_sequences=True,
    )

    assert build_run_metadata(args) == {
        "model": "GeneralizedContextModel",
        "data_dir": "/tmp/data",
        "datasets": ["demo"],
        "num_splits": 10,
        "max_folds": 2,
        "num_iter": 100,
        "lr": 0.01,
        "drop_incomplete_sequences": True,
    }


def test_build_fit_metadata_combines_context_without_changing_report_shape() -> None:
    run_metadata = {"model": "GeneralizedContextModel", "num_iter": 100}
    dataset_metadata = {"dataset": "demo", "path": "/tmp/demo.csv"}
    fold_metadata = {"fold": 1, "train_rows": 9, "eval_rows": 3}

    assert build_fit_metadata(run_metadata, dataset_metadata, fold_metadata) == {
        "model": "GeneralizedContextModel",
        "num_iter": 100,
        "dataset": "demo",
        "path": "/tmp/demo.csv",
        "fold": 1,
        "train_rows": 9,
        "eval_rows": 3,
    }
