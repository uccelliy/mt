"""Cross-check the sparse LM-head path against the standard dense forward.

The scorer normally projects only the scored positions through the LM head,
so the full ``[sequence, vocabulary]`` logits never materialize. That path
has to reproduce whatever final-logit transform the model's own causal-LM
wrapper applies -- for Gemma that is `final_logit_softcapping`, and getting
it wrong shifts every number quietly rather than raising.

Unit tests already pin this on a tiny `Gemma4ForCausalLM`. What they cannot
cover is the real gated checkpoint under NF4 on a V100, which is the gate
design §11 puts in front of unlocking R10-R12. This script closes it, and
reports the peak memory of each path while it is there.

    python scripts/experiments/compare_forward_paths.py \
        --model google/gemma-4-E2B-it --load 4bit --dtype fp16 --sessions 6
"""

from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer

from _common import load_model, load_sessions, pick_device, resolve_dtype
import mt.evaluation.transcript_scoring as transcript_scoring
from mt.evaluation.transcript_scoring import score_session_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model name or path")
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter")
    parser.add_argument("--data",
                        default="data/psych-101-test/prompts_testing_t1.jsonl",
                        help="Path to prompts .jsonl")
    parser.add_argument("--sessions", type=int, default=6,
                        help="Sessions to score through both paths")
    parser.add_argument("--max-participants", type=int, default=1,
                        help="Sessions per experiment before --sessions caps "
                             "the total; 1 spreads the sample over tasks")
    parser.add_argument("--batch-tokens", type=int, default=8192,
                        help="Must match the protocol value being validated")
    parser.add_argument("--max-chars", type=int, default=8000,
                        help="Skip sessions longer than this. The dense "
                             "reference materializes [sequence, vocabulary] "
                             "logits and Gemma's wrapper then allocates three "
                             "more tensors that size for its softcap, so on a "
                             "16 GB card the reference is the side that does "
                             "not fit. Agreement is a property of the "
                             "transform, not of length: short sessions prove "
                             "it just as well")
    parser.add_argument("--dtype", default="fp16",
                        choices=["auto", "fp32", "fp16", "bf16"])
    parser.add_argument("--load", default="4bit",
                        choices=["none", "8bit", "4bit"])
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="Max acceptable absolute NLL difference per choice")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or pick_device()
    dtype = resolve_dtype(args.dtype, device)
    rows = load_sessions(args.data, max_participants=args.max_participants,
                         seed=0)
    rows = [row for row in rows
            if len(row["text"]) <= args.max_chars][:args.sessions]
    if not rows:
        raise SystemExit(f"no session is under --max-chars {args.max_chars}")
    longest = max(len(row["text"]) for row in rows)
    print(f"comparing {len(rows)} sessions (longest {longest:,} chars) on "
          f"{device} ({dtype}) with {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = load_model(args.model, dtype, device, load=args.load,
                       adapter=args.adapter)
    text_config = transcript_scoring._text_config(model)
    model_type = getattr(text_config, "model_type", None)
    print(f"model_type: {model_type}  final_logit_softcapping: "
          f"{getattr(text_config, 'final_logit_softcapping', None)}")
    # Without this the run still prints a reassuring zero: a model outside the
    # allowlist takes the dense branch both times, so the comparison would be
    # a tensor against itself.
    if model_type not in transcript_scoring._SPARSE_PROJECTION_MODEL_TYPES:
        raise SystemExit(
            f"{model_type!r} is not in _SPARSE_PROJECTION_MODEL_TYPES, so both "
            f"runs would take the dense path and the comparison would prove "
            f"nothing. Add it there first if the sparse path is meant to "
            f"apply to this model.")

    sparse, sparse_peak = score_with(model, tokenizer, rows, device,
                                     args.batch_tokens, sparse=True)
    dense, dense_peak = score_with(model, tokenizer, rows, device,
                                   args.batch_tokens, sparse=False)

    if sparse.keys() != dense.keys():
        raise SystemExit(f"the two paths scored different choices: "
                         f"{len(sparse)} vs {len(dense)}")
    if not sparse:
        raise SystemExit("no choices were scored; nothing to compare")

    worst_key, worst = max(((key, abs(sparse[key] - dense[key]))
                            for key in sparse), key=lambda pair: pair[1])
    print(f"\ncompared {len(sparse)} choices")
    print(f"peak memory  sparse: {sparse_peak:6.2f} GiB   "
          f"dense: {dense_peak:6.2f} GiB")
    print(f"max |sparse - dense| NLL: {worst:.3e}  at {worst_key}")
    if worst > args.tolerance:
        raise SystemExit(
            f"FAILED: the sparse LM-head path disagrees with the standard "
            f"forward by {worst:.3e} > {args.tolerance:.1e}. The final-logit "
            f"transform is probably not being reproduced.")
    print(f"PASSED: paths agree within {args.tolerance:.1e}")


def score_with(model, tokenizer, rows, device, batch_tokens, *, sparse):
    """Score every choice through one forward path, keyed for comparison.

    The dense path is forced by emptying the model-type allowlist, which is
    the same switch `_forward` consults -- so this compares the two branches
    of the real scorer rather than a reimplementation of either.
    """

    original = transcript_scoring._SPARSE_PROJECTION_MODEL_TYPES
    if not sparse:
        transcript_scoring._SPARSE_PROJECTION_MODEL_TYPES = frozenset()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        scored = score_session_rows(model, tokenizer, rows, device=device,
                                    max_batch_tokens=batch_tokens)
    finally:
        transcript_scoring._SPARSE_PROJECTION_MODEL_TYPES = original
    peak = (torch.cuda.max_memory_allocated() / 2**30
            if device.startswith("cuda") else float("nan"))
    label = "sparse" if sparse else "dense "
    print(f"  {label} path done, peak {peak:.2f} GiB")
    return {(meta["experiment"], meta["participant"], score.choice_index):
            score.nll
            for meta, scores in scored for score in scores}, peak


if __name__ == "__main__":
    main()
