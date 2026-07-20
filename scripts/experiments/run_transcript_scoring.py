"""Score Psych-101-style transcripts with a causal LM (E0 entry point)."""

from __future__ import annotations

import argparse

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from _common import (
    append_records,
    completed_sessions,
    guard_output,
    load_sessions,
    parse_shard,
    pick_device,
    session_key,
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
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu (default: auto)")
    args = parser.parse_args()

    guard_output(args.output, args.resume)
    rows = load_sessions(args.data, experiment=args.experiment,
                         participants=args.participants,
                         max_participants=args.max_participants,
                         seed=args.seed, shard=parse_shard(args.shard))
    done = completed_sessions(args.output) if args.resume else set()
    pending = [r for r in rows if session_key(r) not in done]
    device = args.device or pick_device()
    print(f"scoring {len(pending)} of {len(rows)} sessions on {device} "
          f"with {args.model} ({len(done)} already done)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model = model.to(device).eval()

    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start:start + args.chunk_size]
        records = score_session_rows(model, tokenizer, chunk, device=device,
                                     max_batch_tokens=args.batch_tokens)
        append_records(args.output, records)
        print(f"progress: {min(start + args.chunk_size, len(pending))}"
              f"/{len(pending)} sessions", flush=True)

    frame = pd.read_csv(args.output)
    per_participant = frame.groupby(['experiment', 'participant'])[
        'nll'].mean()
    per_experiment = per_participant.groupby('experiment').mean()
    print(f"total choice scores in {args.output}: {len(frame)}")
    print(f"macro NLL over {len(per_experiment)} experiments: "
          f"{per_experiment.mean():.4f}")

if __name__ == "__main__":
    main()
