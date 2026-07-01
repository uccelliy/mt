"""Text supervision helpers for language-model training data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import glob
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from mt.data._loading import load


LEFT_TARGET_MARKER = "<<"
RIGHT_TARGET_MARKER = ">>"


def find_target_spans(
    text: str,
    *,
    left_marker: str = LEFT_TARGET_MARKER,
    right_marker: str = RIGHT_TARGET_MARKER,
) -> list[tuple[int, int]]:
    """Return character spans between supervision markers."""

    spans: list[tuple[int, int]] = []
    pos = 0
    while True:
        left = text.find(left_marker, pos)
        if left == -1:
            break
        start = left + len(left_marker)
        right = text.find(right_marker, start)
        if right == -1:
            break
        spans.append((start, right))
        pos = right + len(right_marker)
    return spans


def preprocess_marked_text(
    text: str,
    tokenizer: Any,
    max_length: int,
    *,
    truncation: str = "head",
    left_marker: str = LEFT_TARGET_MARKER,
    right_marker: str = RIGHT_TARGET_MARKER,
) -> dict[str, list[int]]:
    """Tokenize text and supervise only tokens inside target markers."""

    if getattr(tokenizer, "eos_token", None) is not None:
        text = text + tokenizer.eos_token

    target_spans = find_target_spans(
        text,
        left_marker=left_marker,
        right_marker=right_marker,
    )
    encoded = tokenizer(
        text,
        truncation=False,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    labels = [-100] * len(input_ids)

    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_start == token_end:
            continue
        if any(
            token_start < span_end and token_end > span_start
            for span_start, span_end in target_spans
        ):
            labels[token_index] = input_ids[token_index]

    input_ids, labels = truncate_tokenized_example(
        input_ids,
        labels,
        max_length=max_length,
        truncation=truncation,
    )
    return {"input_ids": input_ids, "labels": labels}


def truncate_tokenized_example(
    input_ids: list[int],
    labels: list[int],
    *,
    max_length: int,
    truncation: str = "head",
) -> tuple[list[int], list[int]]:
    """Truncate a tokenized example while preserving input/label alignment."""

    if len(input_ids) <= max_length:
        return input_ids, labels
    if truncation == "head":
        return input_ids[:max_length], labels[:max_length]
    if truncation == "tail":
        return input_ids[-max_length:], labels[-max_length:]
    raise ValueError(f"Unknown truncation strategy: {truncation!r}")


def preprocess_supervised_example(
    example: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
    *,
    text_column: str = "text",
    format_kind: str = "text",
    context_columns: Sequence[str] | None = None,
    target_columns: Sequence[str] | None = None,
    truncation: str = "head",
) -> dict[str, list[int]]:
    """Build supervised text from a row and tokenize it."""

    text = format_supervised_example(
        example,
        text_column=text_column,
        format_kind=format_kind,
        context_columns=context_columns,
        target_columns=target_columns,
    )
    return preprocess_marked_text(text, tokenizer, max_length=max_length, truncation=truncation)


def format_supervised_example(
    example: Mapping[str, Any],
    *,
    text_column: str = "text",
    format_kind: str = "text",
    context_columns: Sequence[str] | None = None,
    target_columns: Sequence[str] | None = None,
) -> str:
    """Return text with one or more target spans marked for supervised loss."""

    if format_kind == "text":
        if text_column not in example:
            raise KeyError(f"Text column {text_column!r} is missing from the example.")
        return str(example[text_column])
    if format_kind == "record":
        return format_record_as_marked_text(
            example,
            context_columns=context_columns,
            target_columns=target_columns,
        )
    raise ValueError(f"Unknown supervision format: {format_kind!r}")


def format_record_as_marked_text(
    example: Mapping[str, Any],
    *,
    context_columns: Sequence[str] | None = None,
    target_columns: Sequence[str] | None = None,
) -> str:
    """Serialize a tabular trial row as structured input plus marked target JSON.

    This is a data-preparation helper for converting structured records into the
    same marked-text convention used by Psych-101. Training code should consume
    the resulting text rather than know about the source record schema.
    """

    if not target_columns:
        raise ValueError("target_columns is required when formatting a record.")

    row = {str(key): _json_ready(value) for key, value in example.items()}
    target_names = [str(column) for column in target_columns]
    missing = [column for column in target_names if column not in row]
    if missing:
        raise KeyError(f"Missing target columns for record formatting: {missing}")

    if context_columns is None:
        target_set = set(target_names)
        context_names = [column for column in row if column not in target_set]
    else:
        context_names = [str(column) for column in context_columns]
        missing_context = [column for column in context_names if column not in row]
        if missing_context:
            raise KeyError(f"Missing context columns for record formatting: {missing_context}")

    context = {column: row[column] for column in context_names}
    target = {column: row[column] for column in target_names}
    context_json = json.dumps(context, ensure_ascii=True, sort_keys=False, separators=(",", ":"))
    target_json = json.dumps(target, ensure_ascii=True, sort_keys=False, separators=(",", ":"))
    return f"trial={context_json}\ntarget={LEFT_TARGET_MARKER}{target_json}{RIGHT_TARGET_MARKER}\n"


def load_marked_text_supervision_dataset(
    source: str | Path,
    tokenizer: Any,
    *,
    source_kind: str = "hf",
    dataset_config: str | None = None,
    split: str = "train",
    file_pattern: str = "*.csv",
    text_column: str = "text",
    max_samples: int | None = None,
    eval_fraction: float = 0.05,
    seed: int = 42,
    max_length: int = 512,
    truncation: str = "head",
    num_proc: int = 1,
) -> DatasetDict:
    """Load Psych-101-style marked text and tokenize it for causal-LM training."""

    return load_supervision_dataset(
        source,
        tokenizer,
        source_kind=source_kind,
        dataset_config=dataset_config,
        split=split,
        file_pattern=file_pattern,
        text_column=text_column,
        format_kind="text",
        max_samples=max_samples,
        eval_fraction=eval_fraction,
        seed=seed,
        max_length=max_length,
        truncation=truncation,
        num_proc=num_proc,
    )


def load_supervision_dataset(
    source: str | Path,
    tokenizer: Any,
    *,
    source_kind: str = "hf",
    dataset_config: str | None = None,
    split: str = "train",
    file_pattern: str = "*.csv",
    text_column: str = "text",
    format_kind: str = "text",
    context_columns: Sequence[str] | None = None,
    target_columns: Sequence[str] | None = None,
    max_samples: int | None = None,
    eval_fraction: float = 0.05,
    seed: int = 42,
    max_length: int = 512,
    truncation: str = "head",
    num_proc: int = 1,
) -> DatasetDict:
    """Load, split, tokenize, and filter a marked-text supervision dataset."""

    raw_dataset = load_raw_supervision_dataset(
        source,
        source_kind=source_kind,
        dataset_config=dataset_config,
        split=split,
        file_pattern=file_pattern,
    )
    if max_samples is not None and max_samples > 0:
        raw_dataset = raw_dataset.select(range(min(max_samples, len(raw_dataset))))

    split_dataset = split_train_eval(raw_dataset, eval_fraction=eval_fraction, seed=seed)
    remove_columns = split_dataset["train"].column_names
    tokenized_dataset = split_dataset.map(
        lambda example: preprocess_supervised_example(
            example,
            tokenizer,
            max_length,
            text_column=text_column,
            format_kind=format_kind,
            context_columns=context_columns,
            target_columns=target_columns,
            truncation=truncation,
        ),
        remove_columns=remove_columns,
        num_proc=num_proc,
    )
    return tokenized_dataset.filter(has_supervised_token)


def load_raw_supervision_dataset(
    source: str | Path,
    *,
    source_kind: str = "hf",
    dataset_config: str | None = None,
    split: str = "train",
    file_pattern: str = "*.csv",
) -> Dataset:
    """Load a Hugging Face or local file dataset for marked-text supervision."""

    if source_kind == "hf":
        if dataset_config:
            return load_dataset(str(source), dataset_config, split=split)
        return load_dataset(str(source), split=split)
    if source_kind == "file":
        return _load_local_dataset(source, pattern=file_pattern)
    raise ValueError(f"Unknown source kind: {source_kind!r}")


def split_train_eval(dataset: Dataset, *, eval_fraction: float, seed: int) -> DatasetDict:
    """Split a raw dataset, keeping tiny datasets usable for smoke tests."""

    if not 0 <= eval_fraction < 1:
        raise ValueError("eval_fraction must be in [0, 1).")
    if eval_fraction == 0 or len(dataset) < 2:
        return DatasetDict({"train": dataset, "test": dataset})
    return dataset.train_test_split(test_size=eval_fraction, seed=seed)


def has_supervised_token(example: Mapping[str, Any]) -> bool:
    return any(label != -100 for label in example["labels"])


def make_lm_collate_fn(tokenizer: Any):
    """Return a padding collator for causal-LM examples."""

    def collate_fn(batch: Sequence[Mapping[str, Sequence[int]]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        attention_mask = [torch.ones_like(item, dtype=torch.long) for item in input_ids]

        return {
            "input_ids": pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=tokenizer.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                labels,
                batch_first=True,
                padding_value=-100,
            ),
        }

    return collate_fn


def _load_local_dataset(source: str | Path, *, pattern: str) -> Dataset:
    path = Path(source)
    paths: list[Path]
    if path.is_dir():
        paths = sorted(path.glob(pattern))
    elif glob.has_magic(str(source)):
        paths = sorted(Path(match) for match in glob.glob(str(source)))
    else:
        paths = [path]

    if not paths:
        raise FileNotFoundError(f"No local dataset files found for {source!s}.")

    dataframes = [_load_local_dataframe(path) for path in paths]
    df = pd.concat(dataframes, ignore_index=True) if len(dataframes) > 1 else dataframes[0]
    return Dataset.from_pandas(df, preserve_index=False)


def _load_local_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    return load(path)


def _json_ready(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if not isinstance(value, (list, tuple, dict)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value
