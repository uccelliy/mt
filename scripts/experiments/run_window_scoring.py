"""Score choices under truncated context windows (E3 entry point)."""

from __future__ import annotations

import argparse

import pandas as pd
from transformers import AutoTokenizer

from _common import (
    append_records,
    completed_sessions,
    empty_device_cache,
    failure_log_for,
    guard_output,
    load_model,
    load_sessions,
    parse_shard,
    pick_device,
    resolve_dtype,
    session_key,
    skip_log_for,
)
from mt.evaluation.context_windows import (
    grid_targets,
    score_window_grid,
    segment_transcript,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="HF model name or path")
    parser.add_argument("--data", required=True,
                        help="Path to prompts .jsonl")
    parser.add_argument("--experiment", default=None,
                        help="Filter to one experiment id")
    parser.add_argument("--participants", type=int, default=None,
                        help="Limit total session count")
    parser.add_argument("--max-participants", type=int, default=None,
                        help="Sample at most this many sessions per "
                             "experiment (seeded, reproducible)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--shard", default=None,
                        help="'k/n': process every n-th session starting "
                             "at k (data parallelism across GPUs)")
    parser.add_argument("--windows", default="0,1,2,5,10,20,full",
                        help="Comma-separated window sizes, 'full' allowed")
    parser.add_argument("--num-positions", type=int, default=8,
                        help="Grid positions per session")
    parser.add_argument("--resume", action="store_true",
                        help="Skip sessions already in the output CSV")
    parser.add_argument("--batch-tokens", type=int, default=16384,
                        help="Max padded tokens per forward batch")
    parser.add_argument("--max-chars", type=int, default=None,
                        help="Skip sessions longer than this many chars "
                             "(memory guard; ~3.3 chars per token)")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "fp32", "fp16", "bf16"],
                        help="Model weight dtype (default: per-device auto)")
    parser.add_argument("--load", default="none",
                        choices=["none", "8bit", "4bit"],
                        help="bitsandbytes quantization (CUDA only; "
                             "approximate NLL, for preview not final numbers)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu (default: auto)")
    args = parser.parse_args()

    guard_output(args.output, args.resume)
    failures = failure_log_for(args.output)
    windows = [w if w == "full" else int(w)
               for w in args.windows.split(",")]
    rows = load_sessions(args.data, experiment=args.experiment,
                         participants=args.participants,
                         max_participants=args.max_participants,
                         seed=args.seed, shard=parse_shard(args.shard),
                         max_chars=args.max_chars,
                         skip_log=skip_log_for(args.output))
    done = set()
    if args.resume:
        done = completed_sessions(args.output) | completed_sessions(failures)
    pending = [r for r in rows if session_key(r) not in done]
    device = args.device or pick_device()
    dtype = resolve_dtype(args.dtype, device)
    print(f"scoring {len(pending)} of {len(rows)} sessions x "
          f"{len(windows)} windows x {args.num_positions} positions "
          f"on {device} ({dtype}) with {args.model} "
          f"({len(done)} already done)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, dtype, device, load=args.load)

    for count, row in enumerate(pending, start=1):
        records = score_session_windows(model, tokenizer, row, windows,
                                        args.num_positions, device,
                                        args.batch_tokens, failures)
        append_records(args.output, records)
        empty_device_cache(device)
        if count % 10 == 0 or count == len(pending):
            print(f"progress: {count}/{len(pending)} sessions", flush=True)

    frame = pd.read_csv(args.output)
    print(f"total scores in {args.output}: {len(frame)}")
    print(frame.groupby("window")['nll'].mean().to_string())
    if failures.exists():
        n_failed = len(pd.read_csv(failures))
        print(f"note: {n_failed} sessions logged as failed in {failures}")

def score_session_windows(model, tokenizer, row, windows, num_positions,
                          device, batch_tokens, failures):
    """Score one session's window grid, logging OOM instead of crashing."""

    segmented = segment_transcript(row['text'])
    targets = grid_targets(len(segmented.segments), num_positions)
    cells = []
    labels = []
    for window in windows:
        for target in targets:
            effective = target if window == "full" else window
            cells.append((target, effective))
            labels.append(window)
    try:
        grid = score_window_grid(model, tokenizer, segmented, cells,
                                 device=device, max_batch_tokens=batch_tokens)
    except RuntimeError as error:
        empty_device_cache(device)
        append_records(failures, [{'experiment': row['experiment'],
                                   'participant': row['participant'],
                                   'chars': len(row['text']),
                                   'error': str(error)[:200]}])
        print(f"  OOM on {row['experiment']} p{row['participant']} "
              f"({len(row['text'])} chars): logged and skipped", flush=True)
        return []
    records = []
    for label, cell_records in zip(labels, grid):
        for record in cell_records:
            record.update(experiment=row['experiment'],
                          participant=row['participant'], window=label)
            records.append(record)
    return records

if __name__ == "__main__":
    main()
