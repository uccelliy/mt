"""Build a paper-protocol NLL result from a completed E0 score cache.

The official Centaur evaluator uses 36 task families, right-truncates each
transcript to 32,768 tokenizer tokens, and reports a session-mean loss per
task.  A completed full-context E0 cache can be reused exactly whenever no
marked response token straddles that cutoff.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from mt.data._llm_supervision import find_target_spans
from mt.evaluation.transcript_scoring import map_spans_to_token_indices


PAPER_TASKS = (
    "badham2017deficits", "bahrami2020four", "enkavi2019adaptivenback",
    "enkavi2019digitspan", "enkavi2019gonogo", "enkavi2019recentprobes",
    "feng2021dynamics", "flesch2018comparing", "frey2017cct",
    "frey2017risk", "gershman2018deconstructing", "gershman2020reward",
    "hebart2023things", "hilbig2014generalized", "kool2016when",
    "kool2017cost", "lefebvre2017behavioural", "levering2020revisiting",
    "ludwig2023human", "peterson2021using", "plonsky2018when",
    "ruggeri2022globalizability", "sadeghiyeh2020temporal",
    "schulz2020finding", "somerville2017charting", "speekenbrink2008learning",
    "steingroever2015data", "tomov2020discovery", "tomov2021multitask",
    "waltz2020differential", "wilson2014humans", "wu2023chunking",
    "wulff2018description", "wulff2018sampling", "xiong2023neural",
    "zorowitz2023data",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        help="Completed full-context E0 score CSV")
    parser.add_argument("--replacement", action="append", default=[],
                        help="UTF-8-corrected score CSV whose sessions "
                             "replace matching source sessions; repeatable")
    parser.add_argument("--data", required=True,
                        help="Psych-101-test prompts .jsonl")
    parser.add_argument("--model", required=True,
                        help="Tokenizer source used for E0")
    parser.add_argument("--max-tokens", type=int, default=32768,
                        help="Official max_seq_length, including BOS")
    parser.add_argument("--output", required=True,
                        help="Filtered per-choice paper-protocol CSV")
    parser.add_argument("--summary", required=True,
                        help="Per-task official-style summary CSV")
    args = parser.parse_args()

    source = read_scores(args.source)
    for replacement_path in args.replacement:
        source = replace_sessions(source, read_scores(replacement_path))

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              local_files_only=True)
    expected = expected_paper_choices(args.data, tokenizer, args.max_tokens)
    output = select_expected_scores(source, expected)
    summary = summarize_paper_scores(output)

    output_path = Path(args.output)
    summary_path = Path(args.summary)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    summary.to_csv(summary_path)

    print(f"wrote {len(output)} retained choice scores to {output_path}")
    print(f"wrote {len(summary)} task summaries to {summary_path}")
    print(f"sessions: {output[['experiment', 'participant']].drop_duplicates().shape[0]}")
    print("official-style task-macro eval loss: "
          f"{summary['official_eval_loss'].mean():.6f}")
    print("token-micro NLL (diagnostic): "
          f"{output['nll'].sum() / output['num_tokens'].sum():.6f}")


def paper_task(experiment):
    """Return the official task family for an exact experiment id."""

    return next((task for task in PAPER_TASKS
                 if str(experiment).startswith(task)), None)


def read_scores(path):
    """Read an E0 result CSV with stable session identifiers."""

    required = {"experiment", "participant", "choice_index", "nll",
                "num_tokens"}
    frame = pd.read_csv(path, dtype={"experiment": str, "participant": str})
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing score columns: {sorted(missing)}")
    frame["choice_index"] = frame["choice_index"].astype(int)
    return frame


def replace_sessions(source, replacement):
    """Replace every source session represented by the replacement CSV."""

    replacement_keys = set(zip(replacement["experiment"],
                               replacement["participant"]))
    source_keys = list(zip(source["experiment"], source["participant"]))
    keep = [key not in replacement_keys for key in source_keys]
    return pd.concat([source.loc[keep], replacement], ignore_index=True)


def expected_paper_choices(data_path, tokenizer, max_tokens):
    """Map each retained official response choice to its task and token count."""

    expected = []
    crossing = []
    with open(data_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task = paper_task(row["experiment"])
            if task is None:
                continue
            full, truncated = choice_token_counts(row["text"], tokenizer,
                                                   max_tokens)
            for choice_index, (all_tokens, kept_tokens) in enumerate(
                    zip(full, truncated)):
                if kept_tokens == 0:
                    continue
                if kept_tokens != all_tokens:
                    crossing.append((row["experiment"], row["participant"],
                                     choice_index, all_tokens, kept_tokens))
                    continue
                expected.append({
                    "paper_task": task,
                    "experiment": str(row["experiment"]),
                    "participant": str(row["participant"]),
                    "choice_index": choice_index,
                    "expected_num_tokens": all_tokens,
                })
    if crossing:
        preview = crossing[:5]
        raise ValueError(
            "A marked response crosses the token cutoff; its aggregate "
            f"choice NLL cannot be reused. Examples: {preview}"
        )
    return pd.DataFrame(expected)


def choice_token_counts(text, tokenizer, max_tokens):
    """Return full and head-truncated marked-token counts per choice."""

    full = tokenizer(text, add_special_tokens=True, truncation=False,
                     return_offsets_mapping=True)
    truncated = tokenizer(text, add_special_tokens=True, truncation=True,
                          max_length=max_tokens,
                          return_offsets_mapping=True)
    spans = find_target_spans(text)
    full_groups = map_spans_to_token_indices(list(full["offset_mapping"]),
                                             spans)
    truncated_groups = map_spans_to_token_indices(
        list(truncated["offset_mapping"]), spans)
    full_counts = [len([index for index in group if index > 0])
                   for group in full_groups]
    truncated_counts = [len([index for index in group if index > 0])
                        for group in truncated_groups]
    return full_counts, truncated_counts


def select_expected_scores(scores, expected):
    """Keep exactly the cached choice scores retained by the paper cutoff."""

    keys = ["experiment", "participant", "choice_index"]
    duplicate = scores.duplicated(keys, keep=False)
    if duplicate.any():
        raise ValueError("Score cache has duplicate session/choice rows.")
    result = expected.merge(scores, on=keys, how="left", validate="one_to_one")
    missing = result["nll"].isna()
    if missing.any():
        preview = result.loc[missing, keys].head().to_dict("records")
        raise ValueError(f"Score cache misses retained choices: {preview}")
    mismatch = result["num_tokens"].astype(int) != result["expected_num_tokens"]
    if mismatch.any():
        preview = result.loc[mismatch, keys + ["num_tokens",
                                                "expected_num_tokens"]]
        raise ValueError("Cached token counts disagree with tokenizer: "
                         f"{preview.head().to_dict('records')}")
    columns = ["paper_task", "experiment", "participant", "choice_index",
               "nll", "num_tokens"]
    return result[columns].sort_values(columns[:4]).reset_index(drop=True)


def summarize_paper_scores(frame):
    """Summarize with the official batch-size-one session-mean convention."""

    sessions = frame.groupby(["paper_task", "experiment", "participant"])
    sessions = sessions.agg(session_nll=("nll", "sum"),
                            session_tokens=("num_tokens", "sum"),
                            choices=("nll", "size"))
    sessions["session_token_nll"] = (
        sessions["session_nll"] / sessions["session_tokens"]
    )
    task = sessions.groupby("paper_task").agg(
        official_eval_loss=("session_token_nll", "mean"),
        n_sessions=("session_token_nll", "size"),
        choices=("choices", "sum"),
        target_tokens=("session_tokens", "sum"),
        nll_sum=("session_nll", "sum"),
    )
    task["token_micro_nll"] = task["nll_sum"] / task["target_tokens"]
    return task[["official_eval_loss", "token_micro_nll", "n_sessions",
                 "choices", "target_tokens"]]


if __name__ == "__main__":
    main()
