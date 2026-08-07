"""Run formal Track-P context-window scoring (benchmark design §7.1).

This runner subsumes the ordinary full-context Track-P run.  When ``full``
is requested, it scores the original transcript once and writes every choice
under ``condition=full``.  Each finite window is evaluated only at the target
grid and written under ``condition=e3:w=<window>``.  All conditions use the
same three raw tables as the full-context runner:

    predictions.csv   one row per condition/choice
    pred_topk.csv     one row per condition/choice/token-position/rank
    pred_options.csv  one row per condition/choice/legal option

The model's greedy readout is never fed back into the transcript.  The full
human transcript remains the teacher-forced source for every later prefix.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

import pandas as pd
from transformers import AutoTokenizer

from _common import (
    append_records,
    empty_device_cache,
    failure_log_for,
    fsync_directory,
    fsync_files,
    guard_output,
    is_session_failure,
    library_versions,
    load_model,
    load_sessions,
    log_session_failure,
    parse_shard,
    pick_device,
    resolve_dtype,
    session_key,
    skip_log_for,
)
from run_transcript_scoring import PREDICTION_COLUMNS as TRACK_P_PREDICTION_COLUMNS
from mt.evaluation.context_windows import (
    SEGMENTATION_PROTOCOL,
    WINDOW_UNIT,
    grid_targets,
    score_window_grid,
    segment_transcript,
)
from mt.evaluation.transcript_scoring import ContextLengthError, legal_mass, score_marked_text
from mt.models.llm.supervision import find_target_spans
from mt.utils.slurm_progress import ProgressNotifier, is_lead_worker


WINDOW_COLUMNS = [
    "window",
    "target_index",
    "effective_window",
    "position_grid",
    "n_segments",
    "target_fraction",
    "is_anchor",
]
PREDICTION_COLUMNS = [*TRACK_P_PREDICTION_COLUMNS, *WINDOW_COLUMNS]
SESSION_COMMIT_SCHEMA = 1


def parse_windows(value):
    """Parse and validate a unique, non-negative window list."""

    windows = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            raise argparse.ArgumentTypeError("window list contains an empty item")
        if raw == "full":
            window = "full"
        else:
            try:
                window = int(raw)
            except ValueError as error:
                raise argparse.ArgumentTypeError(
                    f"window must be a non-negative integer or 'full', got {raw!r}"
                ) from error
            if window < 0:
                raise argparse.ArgumentTypeError(f"window must be non-negative, got {window}")
        if window in windows:
            raise argparse.ArgumentTypeError(f"duplicate window {window!r}")
        windows.append(window)
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True, help="HF model name or path")
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional LoRA adapter applied on the (possibly quantized) base",
    )
    parser.add_argument("--data", required=True, help="Path to prompts .jsonl")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset label stored in every row (default: data parent directory)",
    )
    parser.add_argument(
        "--condition-prefix",
        default="e3",
        choices=["e3"],
        help="Finite-window condition prefix; full context is always condition=full",
    )
    parser.add_argument(
        "--choice-readout",
        required=True,
        choices=["greedy-unconstrained-1token"],
        help="Explicit full-vocabulary greedy one-token Track-P readout",
    )
    parser.add_argument("--experiment", default=None, help="Filter to one experiment id")
    parser.add_argument("--participant", default=None, help="Filter one participant id")
    parser.add_argument("--participants", type=int, default=None, help="Limit session count")
    parser.add_argument(
        "--max-participants",
        type=int,
        default=None,
        help="Sample at most this many sessions per experiment",
    )
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument(
        "--shard",
        default=None,
        help="'k/n': process every n-th session starting at k",
    )
    parser.add_argument(
        "--windows",
        type=parse_windows,
        default=parse_windows("0,1,2,5,10,20,full"),
        help="Unique comma-separated non-negative windows; 'full' scores Track P once",
    )
    parser.add_argument(
        "--num-positions",
        type=int,
        default=5,
        help="Target positions per session; e0-informed requires 5",
    )
    parser.add_argument(
        "--position-grid",
        default="e0-informed",
        choices=["even", "e0-informed"],
        help="Target-position strategy for finite windows",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only sessions with an atomically published session commit marker",
    )
    parser.add_argument(
        "--batch-tokens", type=int, default=16384, help="Max padded tokens per forward batch"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        choices=[20],
        default=20,
        help="Formal §2 diagnostic width (fixed at 20)",
    )
    parser.add_argument(
        "--max-options",
        type=int,
        choices=[256],
        default=256,
        help="Formal single-token option-support cap (fixed at 256)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="Skip sessions longer than this many characters",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
        help="Model weight dtype",
    )
    parser.add_argument(
        "--load",
        default="none",
        choices=["none", "8bit", "4bit"],
        help="bitsandbytes quantization (CUDA only)",
    )
    parser.add_argument("--notify-email", default=None, help="Progress notification email")
    parser.add_argument("--notify-label", default="mt-e3", help="Progress notification label")
    parser.add_argument(
        "--output-dir", required=True, help="Run directory receiving the three raw tables"
    )
    parser.add_argument("--device", default=None, help="cuda / mps / cpu (default: auto)")
    args = parser.parse_args()
    if args.position_grid == "e0-informed" and args.num_positions != 5:
        parser.error("--position-grid e0-informed requires --num-positions 5")

    shard = parse_shard(args.shard)
    tables = table_paths(args.output_dir, shard)
    commits = session_commit_dir(args.output_dir, shard)
    for path in tables.values():
        guard_output(path, args.resume)
    if args.resume:
        for path in tables.values():
            removed = repair_incomplete_csv_tail(path)
            if removed:
                print(f"repaired incomplete CSV tail: {path} ({removed} bytes removed)")
    guard_commits(commits, args.resume)
    failures = failure_log_for(tables["predictions"])
    rows = load_sessions(
        args.data,
        experiment=args.experiment,
        participant=args.participant,
        participants=args.participants,
        max_participants=args.max_participants,
        seed=args.seed,
        shard=shard,
        max_chars=args.max_chars,
        skip_log=skip_log_for(tables["predictions"]),
    )
    done = set()
    if args.resume:
        done = set(read_session_commits(commits))
    pending = [row for row in rows if session_key(row) not in done]

    device = args.device or pick_device()
    dtype = resolve_dtype(args.dtype, device)
    print(
        f"scoring {len(pending)} of {len(rows)} sessions x {args.windows} "
        f"at {args.num_positions} {args.position_grid} anchors on {device} "
        f"({dtype}) with {args.model} ({len(done)} already done)"
    )

    run = {
        "model": args.model + (f"+{args.adapter}" if args.adapter else ""),
        "dataset": args.dataset or Path(args.data).parent.name,
    }
    write_manifest(args.output_dir, args, run, shard)

    notifier = None
    if args.notify_email and is_lead_worker(args.shard) and pending:
        notifier = ProgressNotifier(len(pending), args.notify_label, args.notify_email)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, dtype, device, load=args.load, adapter=args.adapter)

    for count, row in enumerate(pending, start=1):
        observations = score_session_windows(
            model,
            tokenizer,
            row,
            args.windows,
            args.num_positions,
            device,
            args.batch_tokens,
            failures,
            args.position_grid,
            top_k=args.top_k,
            max_options=args.max_options,
            condition_prefix=args.condition_prefix,
        )
        write_session(tables, row, observations, run, commits=commits)
        empty_device_cache(device)
        if notifier:
            notifier.update(count)
        if count % 10 == 0 or count == len(pending):
            print(f"progress: {count}/{len(pending)} sessions", flush=True)

    report(tables, failures)


def table_paths(output_dir, shard):
    """Return the three formal table paths, with a suffix for each shard."""

    suffix = f"_shard{shard[0]}" if shard else ""
    directory = Path(output_dir)
    return {
        name: directory / f"{name}{suffix}.csv"
        for name in ("predictions", "pred_topk", "pred_options")
    }


def repair_incomplete_csv_tail(path, *, chunk_size=1024 * 1024):
    """Remove only a crash-truncated final CSV record, returning removed bytes.

    Every complete row written by pandas ends in a newline. The scanner tracks
    RFC-4180 double-quoted fields so embedded newlines are not mistaken for a
    record boundary. Complete prefix rows are retained for deterministic merge
    deduplication; only the unfinished suffix is discarded before replay.
    """

    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0

    outside, inside, after_quote = range(3)
    state = outside
    position = 0
    last_record_end = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            for byte in chunk:
                position += 1
                if state == outside:
                    if byte == ord('"'):
                        state = inside
                    elif byte == ord("\n"):
                        last_record_end = position
                elif state == inside:
                    if byte == ord('"'):
                        state = after_quote
                elif byte == ord('"'):
                    state = inside
                else:
                    state = outside
                    if byte == ord("\n"):
                        last_record_end = position

    removed = position - last_record_end
    if not removed:
        return 0
    if last_record_end == 0:
        path.unlink()
        return position
    with path.open("r+b") as handle:
        handle.truncate(last_record_end)
        handle.flush()
        os.fsync(handle.fileno())
    return removed


def session_commit_dir(output_dir, shard):
    """Return the per-shard directory of atomic session commit markers."""

    suffix = f"_shard{shard[0]}" if shard else ""
    return Path(output_dir) / f"session_commits{suffix}"


def _session_commit_name(experiment, participant):
    identity = json.dumps(
        [str(experiment), str(participant)], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".json"


def _validate_session_commit(path, payload):
    required = {"schema", "experiment", "participant", "status", "prediction_rows", "conditions"}
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(f"{path}: session commit is missing fields {missing}")
    if payload["schema"] != SESSION_COMMIT_SCHEMA:
        raise RuntimeError(f"{path}: unsupported session commit schema {payload['schema']!r}")
    if payload["status"] not in {"success", "failure"}:
        raise RuntimeError(f"{path}: invalid session commit status {payload['status']!r}")
    prediction_rows = payload["prediction_rows"]
    if (
        not isinstance(prediction_rows, int)
        or isinstance(prediction_rows, bool)
        or prediction_rows < 0
    ):
        raise RuntimeError(f"{path}: prediction_rows must be a non-negative integer")
    conditions = payload["conditions"]
    if not isinstance(conditions, dict) or any(
        not isinstance(condition, str)
        or not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        for condition, count in conditions.items()
    ):
        raise RuntimeError(f"{path}: conditions must map strings to non-negative integers")
    if sum(conditions.values()) != prediction_rows:
        raise RuntimeError(f"{path}: condition counts do not sum to prediction_rows")
    if payload["status"] == "success" and prediction_rows == 0:
        raise RuntimeError(f"{path}: successful session commit has no prediction rows")
    if payload["status"] == "failure" and prediction_rows != 0:
        raise RuntimeError(f"{path}: failed session commit contains prediction rows")
    expected_name = _session_commit_name(payload["experiment"], payload["participant"])
    if path.name != expected_name:
        raise RuntimeError(f"{path}: filename does not match its session identity")


def read_session_commits(directory):
    """Read validated commit markers keyed by ``(experiment, participant)``."""

    directory = Path(directory)
    if not directory.exists():
        return {}
    commits = {}
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"{path}: invalid session commit: {error}") from error
        _validate_session_commit(path, payload)
        key = (str(payload["experiment"]), str(payload["participant"]))
        if key in commits:
            raise RuntimeError(f"{directory}: duplicate session commit for {key}")
        commits[key] = payload
    return commits


def guard_commits(directory, resume):
    """Refuse a fresh run when atomic completion state already exists."""

    directory = Path(directory)
    if not resume and directory.exists() and any(directory.glob("*.json")):
        raise SystemExit(f"{directory} contains session commits; pass --resume to continue into it")


def publish_session_commit(directory, meta, predictions):
    """Atomically publish a receipt only after all three table appends return."""

    key = session_key(meta)
    condition_counts = Counter(str(row["condition"]) for row in predictions)
    payload = {
        "schema": SESSION_COMMIT_SCHEMA,
        "experiment": key[0],
        "participant": key[1],
        "status": "success" if predictions else "failure",
        "prediction_rows": len(predictions),
        "conditions": dict(sorted(condition_counts.items())),
    }
    directory = Path(directory)
    directory_existed = directory.exists()
    directory.mkdir(parents=True, exist_ok=True)
    if not directory_existed:
        fsync_directory(directory.parent)
    path = directory / _session_commit_name(*key)
    _validate_session_commit(path, payload)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise RuntimeError(f"{path}: existing session commit differs from deterministic replay")
        return path
    descriptor, temporary_name = tempfile.mkstemp(
        dir=directory, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def score_session_windows(
    model,
    tokenizer,
    row,
    windows,
    num_positions,
    device,
    batch_tokens,
    failures,
    position_grid="even",
    *,
    top_k=20,
    max_options=256,
    condition_prefix="e3",
):
    """Return ``(condition metadata, ChoiceScore)`` pairs for one session.

    ``full`` is deliberately not a window-grid cell: it is ordinary full Track P
    transcript scored once, with all choices retained.  Finite cells retain
    only target-segment choices and share the frozen full-session option
    support supplied by :func:`score_window_grid`.
    """

    segmented = segment_transcript(row["text"])
    n_segments = len(segmented.segments)
    targets = grid_targets(n_segments, num_positions, strategy=position_grid)
    finite_windows = [window for window in windows if window != "full"]
    requests = [
        (window, (target, min(target, window))) for window in finite_windows for target in targets
    ]
    cells = list(dict.fromkeys(cell for _, cell in requests))

    try:
        full_scores = (
            score_marked_text(
                model,
                tokenizer,
                row["text"],
                device=device,
                max_batch_tokens=batch_tokens,
                top_k=top_k,
                max_options=max_options,
            )
            if "full" in windows
            else []
        )
        grid = (
            score_window_grid(
                model,
                tokenizer,
                segmented,
                cells,
                device=device,
                max_batch_tokens=batch_tokens,
                top_k=top_k,
                max_options=max_options,
            )
            if cells
            else []
        )
    except (RuntimeError, ContextLengthError) as error:
        if not is_session_failure(error):
            raise
        log_session_failure(failures, row, error, device)
        return []

    denominator = max(n_segments - 1, 1)
    target_set = set(targets)
    observations = []
    if full_scores:
        choice_targets = [
            target
            for target, segment in enumerate(segmented.segments)
            for _ in find_target_spans(segment)
        ]
        if len(full_scores) != len(choice_targets):
            raise RuntimeError(
                "Full-context scorer returned a different number of choices "
                "than the lossless window segmentation."
            )
        for score in full_scores:
            target = choice_targets[score.choice_index]
            metadata = window_metadata(
                condition="full",
                window="full",
                target=target,
                effective_window=target,
                position_grid=position_grid,
                n_segments=n_segments,
                denominator=denominator,
                is_anchor=target in target_set,
            )
            observations.append((metadata, score))

    scores_by_cell = dict(zip(cells, grid))
    for window, cell in requests:
        target, effective_window = cell
        metadata = window_metadata(
            condition=f"{condition_prefix}:w={window}",
            window=window,
            target=target,
            effective_window=effective_window,
            position_grid=position_grid,
            n_segments=n_segments,
            denominator=denominator,
            is_anchor=True,
        )
        observations.extend((metadata, score) for score in scores_by_cell[cell])
    return observations


def window_metadata(
    *,
    condition,
    window,
    target,
    effective_window,
    position_grid,
    n_segments,
    denominator,
    is_anchor,
):
    """Build metadata copied verbatim to all three child ledgers."""

    return {
        "condition": condition,
        "window": window,
        "target_index": target,
        "effective_window": effective_window,
        "position_grid": position_grid,
        "n_segments": n_segments,
        "target_fraction": target / denominator,
        "is_anchor": is_anchor,
    }


def shape_session(meta, observations, run):
    """Shape one E3 session into the same three raw tables as full Track P."""

    predictions, topk, options = [], [], []
    for window_meta, score in observations:
        key = {
            **run,
            **window_meta,
            "experiment": meta["experiment"],
            "participant": meta["participant"],
            "choice_index": score.choice_index,
        }
        record = dict.fromkeys(PREDICTION_COLUMNS, "")
        record.update(
            key,
            pred_choice=score.pred_choice,
            pred_token_id=score.pred_token_id,
            human_choice=score.human_choice,
            k_options=score.k_options,
            nll=score.nll,
            num_tokens=score.num_tokens,
            top1_prob=score.top1_prob,
            legal_mass=legal_mass(score.options),
            pred_entropy=score.pred_entropy,
            options_status=score.options_status,
            raw_generation=score.raw_generation,
            format_ok=score.format_ok,
        )
        predictions.append(record)
        topk.extend(
            {
                **key,
                "token_index": candidate.token_index,
                "rank": candidate.rank,
                "token_id": candidate.token_id,
                "token": candidate.token,
                "logprob": candidate.logprob,
            }
            for candidate in score.topk
        )
        options.extend(
            {
                **key,
                "option": option.option,
                "logprob": option.logprob,
                "n_tokens": option.n_tokens,
                "is_human": option.is_human,
                "is_correct": "",
            }
            for option in score.options
        )
    return predictions, topk, options


def write_session(tables, meta, observations, run, *, commits=None):
    """Append all table rows, then atomically publish session completion."""

    predictions, topk, options = shape_session(meta, observations, run)
    append_records(tables["pred_topk"], topk)
    append_records(tables["pred_options"], options)
    append_records(tables["predictions"], predictions)
    if commits is not None:
        if predictions:
            fsync_files(tables.values())
        publish_session_commit(commits, meta, predictions)


def write_manifest(output_dir, args, run, shard):
    """Record every protocol value needed to reproduce the E3 run."""

    path = Path(output_dir) / (f"run_shard{shard[0]}.json" if shard else "run.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        **run,
        "adapter": args.adapter,
        "data": args.data,
        # Kept for compatibility with formal merge acceptance.  Row-level
        # conditions are full and e3:w=<n>; this is their experiment prefix.
        "condition": args.condition_prefix,
        "condition_prefix": args.condition_prefix,
        "full_condition": "full" if "full" in args.windows else None,
        "windows": args.windows,
        "num_positions": args.num_positions,
        "position_grid": args.position_grid,
        "segmentation_protocol": SEGMENTATION_PROTOCOL,
        "window_unit": WINDOW_UNIT,
        "load": args.load,
        "dtype": args.dtype,
        "batch_tokens": args.batch_tokens,
        "choice_readout": args.choice_readout,
        "top_k": args.top_k,
        "max_options": args.max_options,
        "shard": args.shard,
        "commit": git_commit(),
        "versions": library_versions(),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def report(tables, failures):
    """Print validation counts without replacing the raw-table analysis."""

    if not tables["predictions"].exists():
        raise SystemExit(f"No choice scores were written to {tables['predictions']}.")
    frame = pd.read_csv(tables["predictions"])
    totals = frame.groupby("condition", sort=False).agg(
        choices=("nll", "size"),
        nll_sum=("nll", "sum"),
        choice_tokens=("num_tokens", "sum"),
    )
    totals["token_nll"] = totals["nll_sum"] / totals["choice_tokens"]
    print(f"total condition-choice scores in {tables['predictions']}: {len(frame)}")
    print(totals[["choices", "choice_tokens", "token_nll"]].to_string())
    scored = (frame["options_status"] == "scored").mean()
    print(f"legal-option coverage: {100 * scored:.1f}% of condition-choice rows")
    if failures.exists():
        print(f"note: {len(pd.read_csv(failures))} sessions logged as failed in {failures}")


if __name__ == "__main__":
    main()
