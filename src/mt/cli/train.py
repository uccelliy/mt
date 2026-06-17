"""Train/evaluate a registered cognitive model from command-line arguments."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

from mt.data import load_dataframe, split_data_by_column
from mt.models import get_model_class
from mt.training.trainer import Trainer


DEFAULT_OUTPUT_DIR = Path("logs") / "train_cognitive_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Registered model class name.")
    parser.add_argument("--data", nargs="+", type=Path, required=True, help="CSV/parquet/json data files.")
    parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help="Rename a dataframe column before training. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model-kwargs",
        default="{}",
        help='JSON object passed to the model constructor, e.g. \'{"num_options": 3}\'.',
    )
    parser.add_argument("--split-column", default="participant")
    parser.add_argument("--num-splits", type=int, default=2)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--shuffle-splits", action="store_true")
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--num-iter", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--save-model-dir", type=Path, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--drop-incomplete-sequences", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def parse_renames(rename_args: list[str]) -> dict[str, str]:
    renames = {}
    for item in rename_args:
        if "=" not in item:
            raise ValueError(f"Invalid --rename value {item!r}; expected SOURCE=TARGET.")
        source, target = item.split("=", 1)
        if not source or not target:
            raise ValueError(f"Invalid --rename value {item!r}; expected SOURCE=TARGET.")
        renames[source] = target
    return renames


def load_model_kwargs(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("--model-kwargs must decode to a JSON object.")
    return value


def prepare_dataframe(path: Path, renames: dict[str, str], *, drop_incomplete: bool):
    df = load_dataframe(path)
    present_renames = {source: target for source, target in renames.items() if source in df.columns}
    if present_renames:
        df = df.rename(columns=present_renames)

    metadata: dict[str, Any] = {
        "data": str(path),
        "dataset": path.stem,
        "rows_before_sequence_filter": len(df),
        "renamed_columns": present_renames,
    }
    if drop_incomplete:
        df, sequence_metadata = keep_complete_sequences(df)
        metadata.update(sequence_metadata)
    else:
        metadata["rows_after_sequence_filter"] = len(df)
        metadata["dropped_sequence_rows"] = 0
    return df, metadata


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
    filtered = df.merge(complete[group_columns], on=group_columns, how="inner")
    metadata = {
        "rows_after_sequence_filter": len(filtered),
        "dropped_sequence_rows": len(df) - len(filtered),
        "sequences_before_filter": len(sequence_stats),
        "sequences_after_filter": len(complete),
        "dropped_sequences": len(sequence_stats) - len(complete),
    }
    return filtered, metadata


def train_one_dataset(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    renames = parse_renames(args.rename)
    model_kwargs = load_model_kwargs(args.model_kwargs)
    model_class = get_model_class(args.model)
    df, dataset_metadata = prepare_dataframe(
        path,
        renames,
        drop_incomplete=args.drop_incomplete_sequences,
    )

    results = []
    splits = split_data_by_column(
        df,
        args.split_column,
        num_splits=args.num_splits,
        shuffle=args.shuffle_splits,
        random_state=args.random_state,
    )
    for fold, (train_df, eval_df) in enumerate(splits):
        if args.max_folds is not None and fold >= args.max_folds:
            break

        model = model_class(**model_kwargs)
        trainer = Trainer(
            model,
            num_iter=args.num_iter,
            lr=args.lr,
            show_progress=args.show_progress,
        )
        metadata = {
            **dataset_metadata,
            "model": args.model,
            "model_kwargs": model_kwargs,
            "fold": fold,
            "split_column": args.split_column,
            "num_splits": args.num_splits,
            "num_iter": args.num_iter,
            "lr": args.lr,
            "train_rows": len(train_df),
            "eval_rows": len(eval_df),
        }
        save_path = None
        if args.save_model_dir is not None:
            args.save_model_dir.mkdir(parents=True, exist_ok=True)
            save_path = args.save_model_dir / f"{path.stem}_{args.model}_fold{fold}.pt"

        result = trainer.fit_and_evaluate_result(
            train_df,
            eval_df,
            save_path=save_path,
            metadata=metadata,
            split_metadata={"fold": fold, "split_column": args.split_column},
        )
        row = {
            **metadata,
            "final_train_loss": result.training.final_loss,
            "eval_choice_nll": result.evaluation.metrics["choice_nll"],
            "finite": math.isfinite(result.training.final_loss)
            and math.isfinite(result.evaluation.loss_value),
            "save_path": str(save_path) if save_path is not None else None,
        }
        results.append(row)
        print(
            f"{path.stem} model={args.model} fold={fold} "
            f"train_loss={row['final_train_loss']:.6f} "
            f"eval_choice_nll={row['eval_choice_nll']:.6f} "
            f"finite={row['finite']}"
        )

    return results


def save_results(results: list[dict[str, Any]], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"train_cognitive_model_{timestamp}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        import torch

        torch.manual_seed(args.seed)

    all_results = []
    for path in args.data:
        all_results.extend(train_one_dataset(path, args))

    output_path = save_results(all_results, args.output_dir)
    print(f"Wrote results to {output_path}")
