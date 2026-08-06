"""Score Psych-101-style transcripts with a causal LM (Track P entry point).

Writes the three tables of design doc §2.5 into one run directory:

    predictions.csv   one row per choice   -- ① and the derived scalars
    pred_topk.csv     one row per (token position, rank)  -- ②
    pred_options.csv  one row per legal option            -- ③

No metric is computed here beyond what a scalar column literally holds;
everything in §3 is derived from these files offline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

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
    legal_mass,
    predicted_choice,
    score_session_rows,
)
from mt.utils.slurm_progress import ProgressNotifier, is_lead_worker

# §2.5, in order. Track S fills the last three; Track P leaves them empty.
# correct_choice / is_correct stay empty on Psych-101, which has no
# correctness field (§5) -- the columns exist so a dataset that does have
# one writes the same schema.
PREDICTION_COLUMNS = [
    'model', 'dataset', 'condition', 'experiment', 'participant',
    'choice_index', 'pred_choice', 'human_choice', 'correct_choice',
    'is_correct', 'k_options', 'nll', 'num_tokens', 'top1_prob',
    'legal_mass', 'pred_entropy', 'options_status',
    'raw_generation', 'pred_rt_ms', 'format_ok',
]

def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--model", required=True,
                        help="HF model name or path")
    parser.add_argument("--adapter", default=None,
                        help="Optional LoRA adapter applied on the (possibly "
                             "quantized) base, as in the official Centaur "
                             "evaluation")
    parser.add_argument("--data", required=True,
                        help="Path to prompts .jsonl")
    parser.add_argument("--dataset", default=None,
                        help="Dataset name recorded in the output (default: "
                             "the data file's parent directory name)")
    parser.add_argument("--condition", default="full",
                        help="Condition label recorded in the output; keeps "
                             "ablations in the same tables (§2.5)")
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
                        help="Skip sessions already in predictions.csv")
    parser.add_argument("--chunk-size", type=int, default=8,
                        help="Sessions scored between output flushes")
    parser.add_argument("--batch-tokens", type=int, default=16384,
                        help="Max padded tokens per forward batch")
    parser.add_argument("--top-k", type=int, default=20,
                        help="② width: tokens kept per token position, from "
                             "the full vocabulary (0 disables ②)")
    parser.add_argument("--max-options", type=int, default=32,
                        help="③ is skipped for a choice whose legal option "
                             "set is larger than this; on such tasks the "
                             "'<<>> union' approximation is weak anyway "
                             "(§8.5). 0 disables ③")
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
    parser.add_argument("--notify-email", default=None,
                        help="Email progress at 25/50/70/90%%; only the "
                             "shard-0 process sends, so a data-parallel run "
                             "does not deliver four copies")
    parser.add_argument("--notify-label", default="mt-analysis",
                        help="Analysis name shown in the progress emails")
    parser.add_argument("--output-dir", required=True,
                        help="Run directory receiving the three §2.5 tables")
    parser.add_argument("--summary", default=None,
                        help="Optional per-experiment metric CSV path")
    parser.add_argument("--device", default=None,
                        help="cuda / mps / cpu (default: auto)")
    args = parser.parse_args()

    shard = parse_shard(args.shard)
    tables = table_paths(args.output_dir, shard)
    guard_output(tables['predictions'], args.resume)
    failures = failure_log_for(tables['predictions'])
    rows = load_sessions(args.data, experiment=args.experiment,
                         participant=args.participant,
                         participants=args.participants,
                         max_participants=args.max_participants,
                         seed=args.seed, shard=shard,
                         max_chars=args.max_chars,
                         skip_log=skip_log_for(tables['predictions']))
    done = set()
    if args.resume:
        done = completed_sessions(tables['predictions'])
        done |= completed_sessions(failures)
    pending = [r for r in rows if session_key(r) not in done]
    device = args.device or pick_device()
    dtype = resolve_dtype(args.dtype, device)
    print(f"scoring {len(pending)} of {len(rows)} sessions on {device} "
          f"({dtype}) with {args.model} ({len(done)} already done)")

    run = {'model': args.model + (f"+{args.adapter}" if args.adapter else ""),
           'dataset': args.dataset or Path(args.data).parent.name,
           'condition': args.condition}
    write_manifest(args.output_dir, args, run, shard)

    notifier = None
    if args.notify_email and is_lead_worker(args.shard) and pending:
        notifier = ProgressNotifier(len(pending), args.notify_label,
                                    args.notify_email)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, dtype, device, load=args.load,
                       adapter=args.adapter)

    for start in range(0, len(pending), args.chunk_size):
        chunk = pending[start:start + args.chunk_size]
        scored = score_chunk(model, tokenizer, chunk, device,
                             args.batch_tokens, args.top_k,
                             args.max_options, failures)
        write_chunk(tables, scored, run)
        empty_device_cache(device)
        completed = min(start + args.chunk_size, len(pending))
        if notifier:
            notifier.update(completed)
        print(f"progress: {completed}/{len(pending)} sessions", flush=True)

    report(tables, args.summary, failures)

def table_paths(output_dir, shard):
    """The three §2.5 tables, suffixed per shard so merging can glob them."""

    suffix = f"_shard{shard[0]}" if shard else ""
    directory = Path(output_dir)
    return {name: directory / f"{name}{suffix}.csv"
            for name in ('predictions', 'pred_topk', 'pred_options')}

def write_chunk(tables, scored, run):
    """Append one chunk to all three tables, the ledger written last.

    predictions is the resume ledger, so it lands after its own children.
    A crash between the two leaves orphan child rows for at most one chunk;
    merge_shards.slurm drops rows whose choice never made it to
    predictions, which is also where shard-level duplicates are caught.
    """

    predictions, topk, options = [], [], []
    for meta, scores in scored:
        rows = shape_session(meta, scores, run)
        predictions += rows[0]
        topk += rows[1]
        options += rows[2]
    append_records(tables['pred_topk'], topk)
    append_records(tables['pred_options'], options)
    append_records(tables['predictions'], predictions)

def shape_session(meta, scores, run):
    """Turn one session's ChoiceScores into rows of the three tables."""

    predictions, topk, options = [], [], []
    for score in scores:
        key = {**run,
               'experiment': meta['experiment'],
               'participant': meta['participant'],
               'choice_index': score.choice_index}
        record = dict.fromkeys(PREDICTION_COLUMNS, "")
        record.update(key,
                      pred_choice=predicted_choice(score.options),
                      human_choice=score.human_choice,
                      k_options=score.k_options,
                      nll=score.nll,
                      num_tokens=score.num_tokens,
                      top1_prob=score.top1_prob,
                      legal_mass=legal_mass(score.options),
                      pred_entropy=score.pred_entropy,
                      options_status=score.options_status)
        predictions.append(record)
        topk += [{**key, 'token_index': candidate.token_index,
                  'rank': candidate.rank, 'token': candidate.token,
                  'logprob': candidate.logprob}
                 for candidate in score.topk]
        options += [{**key, 'option': option.option,
                     'logprob': option.logprob, 'n_tokens': option.n_tokens,
                     'is_human': option.is_human, 'is_correct': ""}
                    for option in score.options]
    return predictions, topk, options

def write_manifest(output_dir, args, run, shard):
    """Record the run configuration §8.3 requires to be kept with results."""

    path = Path(output_dir) / (f"run_shard{shard[0]}.json" if shard
                               else "run.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**run,
                                'adapter': args.adapter,
                                'data': args.data,
                                'load': args.load,
                                'dtype': args.dtype,
                                'batch_tokens': args.batch_tokens,
                                'top_k': args.top_k,
                                'max_options': args.max_options,
                                'shard': args.shard,
                                'commit': git_commit()}, indent=2))

def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True,
                              text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None

def score_chunk(model, tokenizer, chunk, device, batch_tokens, top_k,
                max_options, failures):
    """Score a chunk, isolating and logging any per-session failures."""

    try:
        return score_session_rows(model, tokenizer, chunk, device=device,
                                  max_batch_tokens=batch_tokens,
                                  top_k=top_k, max_options=max_options)
    except (RuntimeError, ContextLengthError) as error:
        if not is_session_failure(error):
            raise
        empty_device_cache(device)
    scored = []
    for row in chunk:
        try:
            scored += score_session_rows(model, tokenizer, [row],
                                         device=device,
                                         max_batch_tokens=batch_tokens,
                                         top_k=top_k,
                                         max_options=max_options)
        except (RuntimeError, ContextLengthError) as error:
            if not is_session_failure(error):
                raise
            log_session_failure(failures, row, error, device)
    return scored

def report(tables, summary_path, failures):
    """Print pipeline-validation numbers; no §3 metric is computed here."""

    if not tables['predictions'].exists():
        raise SystemExit(f"No choice scores were written to "
                         f"{tables['predictions']}.")
    frame = pd.read_csv(tables['predictions'])
    summary = summarize_scores(frame)
    if summary_path:
        path = Path(summary_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path)
        print(f"wrote per-experiment metrics to {path}")
    print(f"total choice scores in {tables['predictions']}: {len(frame)}")
    print(f"paper-compatible token NLL over all scored responses: "
          f"{frame['nll'].sum() / frame['num_tokens'].sum():.4f}")
    print(f"macro choice NLL over {len(summary)} experiments: "
          f"{summary['macro_choice_nll'].mean():.4f}")
    scored = (frame['options_status'] == "scored").mean()
    print(f"legal-option (③) coverage: {100 * scored:.1f}% of choices")
    for status, count in frame['options_status'].value_counts().items():
        print(f"  {status}: {count}")
    if failures.exists():
        n_failed = len(pd.read_csv(failures))
        print(f"note: {n_failed} sessions logged as failed in {failures}")

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
