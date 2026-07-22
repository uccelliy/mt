"""Score Psych-101-style transcripts with a causal LM (E0 entry point)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from _common import (
    append_records,
    completed_sessions,
    empty_device_cache,
    failure_log_for,
    guard_output,
    is_session_failure,
    load_model,
    load_sessions,
    log_session_failure,
    parse_shard,
    pick_device,
    resolve_dtype,
    session_key,
    skip_log_for,
)
from mt.evaluation.transcript_scoring import (
    ContextLengthError,
    score_session_rows,
)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="HF model name or path")
    parser.add_argument("--data", required=True,
                        help="Path to prompts .jsonl")
    parser.add_argument("--experiment", default=None,
                        help="Filter to one experiment id")
    parser.add_argument("--participant", default=None,
                        help="Filter to one exact participant id")
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
                             "report separately from dense results)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--summary", default=None,
                        help="Optional per-experiment metric CSV path")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu (default: auto)")
    args = parser.parse_args()

    guard_output(args.output, args.resume)
    failures = failure_log_for(args.output)
    rows = load_sessions(args.data, experiment=args.experiment,
                         participant=args.participant,
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

    if not Path(args.output).exists():
        raise SystemExit(f"No choice scores were written to {args.output}.")
    frame = pd.read_csv(args.output)
    summary = summarize_scores(frame)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(summary_path)
        print(f"wrote per-experiment metrics to {summary_path}")
    print(f"total choice scores in {args.output}: {len(frame)}")
    print(f"paper-compatible token NLL over all scored responses: "
          f"{frame['nll'].sum() / frame['num_tokens'].sum():.4f}")
    print(f"macro choice NLL over {len(summary)} experiments: "
          f"{summary['macro_choice_nll'].mean():.4f}")
    if failures.exists():
        n_failed = len(pd.read_csv(failures))
        print(f"note: {n_failed} sessions logged as failed in {failures}")

def score_chunk(model, tokenizer, chunk, device, batch_tokens, failures):
    """Score a chunk, isolating and logging any per-session failures."""

    try:
        return score_session_rows(model, tokenizer, chunk, device=device,
                                  max_batch_tokens=batch_tokens)
    except (RuntimeError, ContextLengthError) as error:
        if not is_session_failure(error):
            raise
        empty_device_cache(device)
    records = []
    for row in chunk:
        try:
            records += score_session_rows(model, tokenizer, [row],
                                          device=device,
                                          max_batch_tokens=batch_tokens)
        except (RuntimeError, ContextLengthError) as error:
            if not is_session_failure(error):
                raise
            log_session_failure(failures, row, error, device)
    return records

def summarize_scores(frame):
    """Return paper-compatible and hierarchical metrics by experiment."""

    required = {'experiment', 'participant', 'nll', 'num_tokens'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    totals = frame.groupby('experiment').agg(
        nll_sum=('nll', 'sum'),
        choice_tokens=('num_tokens', 'sum'),
        choices=('nll', 'size'),
        participants=('participant', 'nunique'),
    )
    totals['paper_token_nll'] = totals['nll_sum'] / totals['choice_tokens']
    participant = frame.groupby(['experiment', 'participant'])['nll'].mean()
    totals['macro_choice_nll'] = participant.groupby('experiment').mean()
    return totals[['paper_token_nll', 'macro_choice_nll', 'choices',
                   'choice_tokens', 'participants']]

if __name__ == "__main__":
    main()
