"""Score Psych-101-style transcripts with a causal LM (E0 entry point)."""

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
from mt.evaluation.transcript_scoring import score_session_rows

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
    parser.add_argument("--resume", action="store_true",
                        help="Skip sessions already in the output CSV")
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Sessions scored between output flushes")
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
    print(f"scoring {len(pending)} of {len(rows)} sessions on {device} "
          f"({dtype}) with {args.model} ({len(done)} already done)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, dtype, device, load=args.load)

    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start:start + args.chunk_size]
        records = score_chunk(model, tokenizer, chunk, device,
                              args.batch_tokens, failures)
        append_records(args.output, records)
        empty_device_cache(device)
        print(f"progress: {min(start + args.chunk_size, len(pending))}"
              f"/{len(pending)} sessions", flush=True)

    frame = pd.read_csv(args.output)
    per_participant = frame.groupby(['experiment', 'participant'])[
        'nll'].mean()
    per_experiment = per_participant.groupby('experiment').mean()
    print(f"total choice scores in {args.output}: {len(frame)}")
    print(f"macro NLL over {len(per_experiment)} experiments: "
          f"{per_experiment.mean():.4f}")
    if failures.exists():
        n_failed = len(pd.read_csv(failures))
        print(f"note: {n_failed} sessions logged as failed in {failures}")

def score_chunk(model, tokenizer, chunk, device, batch_tokens, failures):
    """Score a chunk, isolating and logging any per-session OOM failures."""

    try:
        return score_session_rows(model, tokenizer, chunk, device=device,
                                  max_batch_tokens=batch_tokens)
    except RuntimeError:
        empty_device_cache(device)
    records = []
    for row in chunk:
        try:
            records += score_session_rows(model, tokenizer, [row],
                                          device=device,
                                          max_batch_tokens=batch_tokens)
        except RuntimeError as error:
            empty_device_cache(device)
            append_records(failures, [{'experiment': row['experiment'],
                                       'participant': row['participant'],
                                       'chars': len(row['text']),
                                       'error': str(error)[:200]}])
            print(f"  OOM on {row['experiment']} p{row['participant']} "
                  f"({len(row['text'])} chars): logged and skipped",
                  flush=True)
    return records

if __name__ == "__main__":
    main()
