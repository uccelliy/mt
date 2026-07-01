"""Run the generalized context model on selected tabular datasets.

Run from the repo root:

    uv run python scripts/run_generalized_context.py
    
todo: refactor the logger including metadata and save process
    
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mt.data import load, save_json_report, split_data_kfold  # noqa: E402
from mt.models import GeneralizedContextModel  # noqa: E402
from mt.training.trainer import Trainer  # noqa: E402


DEFAULT_DATA_DIR = ROOT.parent / "mt_data" / "data"
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "generalized_context"
DATASET_LABEL_COLUMNS = {
    "marcelbinz__badham2017deficits": "correct_choice",
    "marcelbinz__speekenbrink2008learning": "TrueStateOfWeather",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing dataset CSV files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=tuple(DATASET_LABEL_COLUMNS),
        help="Dataset stems to run. Each stem is resolved as <data-dir>/<stem>.csv.",
    )
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--num-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--drop-incomplete-sequences",
        action="store_true",
        help="Drop participant-task sequences with missing trial rows.",
    )
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def prepare_dataset(path: Path, dataset: str, *, drop_incomplete: bool) -> tuple[Any, dict[str, Any]]:
    df = load(path)
    label_column = DATASET_LABEL_COLUMNS.get(dataset)
    if "ground_truth" not in df.columns:
        if label_column is None or label_column not in df.columns:
            raise KeyError(
                f"{dataset} needs a ground_truth column or a known label column; "
                f"got columns {list(df.columns)}"
            )
        df = df.rename(columns={label_column: "ground_truth"})

    metadata = build_dataset_metadata(
        dataset=dataset,
        path=path,
        rows_before_sequence_filter=len(df),
        label_column="ground_truth" if label_column is None else label_column,
    )
    if drop_incomplete:
        df, sequence_metadata = keep_complete_sequences(df)
        metadata.update(sequence_metadata)
    else:
        metadata["rows_after_sequence_filter"] = len(df)
        metadata["dropped_sequence_rows"] = 0

    return df, metadata


def build_dataset_metadata(
    *,
    dataset: str,
    path: Path,
    rows_before_sequence_filter: int,
    label_column: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "path": str(path),
        "rows_before_sequence_filter": rows_before_sequence_filter,
        "label_column": label_column,
    }


def build_fold_metadata(
    *,
    fold: int,
    train_rows: int,
    eval_rows: int,
) -> dict[str, Any]:
    return {
        "fold": fold,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
    }


def build_run_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model": "GeneralizedContextModel",
        "data_dir": str(args.data_dir),
        "datasets": list(args.datasets),
        "num_splits": args.num_splits,
        "max_folds": args.max_folds,
        "num_iter": args.num_iter,
        "lr": args.lr,
        "drop_incomplete_sequences": args.drop_incomplete_sequences,
    }


def build_fit_metadata(
    run_metadata: dict[str, Any],
    dataset_metadata: dict[str, Any],
    fold_metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        **run_metadata,
        **dataset_metadata,
        **fold_metadata,
    }


def keep_complete_sequences(df):
    group_columns = ["participant", "task"]
    trial_column = "trial"
    missing = [column for column in (*group_columns, trial_column) if column not in df.columns]
    if missing:
        raise KeyError(f"Missing sequence columns: {missing}")

    trial_counts = df.groupby(group_columns)[trial_column].nunique().rename("num_trials")
    expected_trials = df.groupby("task")[trial_column].nunique().rename("expected_trials")
    sequence_stats = trial_counts.reset_index().merge(
        expected_trials.reset_index(),
        on="task",
        how="left",
    )
    complete = sequence_stats[sequence_stats["num_trials"] == sequence_stats["expected_trials"]]
    complete_keys = complete[group_columns]
    filtered = df.merge(complete_keys, on=group_columns, how="inner")

    metadata = {
        "rows_after_sequence_filter": len(filtered),
        "dropped_sequence_rows": len(df) - len(filtered),
        "sequences_before_filter": len(sequence_stats),
        "sequences_after_filter": len(complete),
        "dropped_sequences": len(sequence_stats) - len(complete),
        "expected_trials_by_task": {
            str(task): int(count) for task, count in expected_trials.to_dict().items()
        },
    }
    return filtered, metadata


def run_dataset(
    dataset: str,
    args: argparse.Namespace,
    run_metadata: dict[str, Any],
) -> dict[str, Any]:
    path = args.data_dir / f"{dataset}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    df, dataset_metadata = prepare_dataset(
        path,
        dataset,
        drop_incomplete=args.drop_incomplete_sequences,
    )
    folds = []
    for fold, (train_df, eval_df) in enumerate(split_data_kfold(df, num_splits=args.num_splits)):
        if args.max_folds is not None and fold >= args.max_folds:
            break

        model = GeneralizedContextModel()
        trainer = Trainer(
            model,
            num_iter=args.num_iter,
            lr=args.lr,
            show_progress=args.show_progress,
        )
        fold_metadata = build_fold_metadata(
            fold=fold,
            train_rows=len(train_df),
            eval_rows=len(eval_df),
        )
        metadata = build_fit_metadata(run_metadata, dataset_metadata, fold_metadata)
        result = trainer.fit_and_evaluate_result(train_df, eval_df, metadata=metadata)
        row = {
            **fold_metadata,
            "final_train_loss": result.training.final_loss,
            "eval_choice_nll": result.evaluation.metrics["choice_nll"],
            "beta": float(model.beta.detach().cpu().item()),
            "finite": math.isfinite(result.training.final_loss)
            and math.isfinite(result.evaluation.loss_value),
        }
        folds.append(row)
        print(
            f"{dataset} fold={fold} "
            f"train_loss={row['final_train_loss']:.6f} "
            f"eval_choice_nll={row['eval_choice_nll']:.6f} "
            f"beta={row['beta']:.6f} "
            f"finite={row['finite']}"
        )

    return {
        "metadata": dataset_metadata,
        "folds": folds,
    }


def main() -> None:
    args = parse_args()
    run_metadata = build_run_metadata(args)
    dataset_results = []
    for dataset in args.datasets:
        dataset_results.append(run_dataset(dataset, args, run_metadata))

    report = {
        "run": run_metadata,
        "datasets": dataset_results,
    }
    output_path = save_json_report(report, args.output_dir, "generalized_context")
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
