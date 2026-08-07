"""Pre-submission checks: fail on the login node, not in the queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

FAILURES: list[str] = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B",
        help="HF base-model name expected in the local cache",
    )
    parser.add_argument(
        "--adapter", default="", help="Optional HF adapter name expected in the local cache"
    )
    parser.add_argument(
        "--data",
        default="data/psych-101-test/prompts_testing_t1.jsonl",
        help="Path to prompts .jsonl",
    )
    parser.add_argument(
        "--output-dir", default="outputs/scoring", help="Directory the runners will write to"
    )
    parser.add_argument(
        "--load",
        default="none",
        choices=["none", "8bit", "4bit"],
        help="Validate an optional quantized CUDA runtime",
    )
    args = parser.parse_args()

    check_environment(args.load)
    rows = check_data(args.data)
    if rows:
        check_segmentation(rows)
    snapshot = check_model_cache(args.model)
    if args.adapter:
        check_model_cache(args.adapter, label="adapter")
    if rows and snapshot:
        check_tokenizer(args.model, rows)
    check_output_dir(args.output_dir)
    report_gpu()
    check_attention_backend()

    if FAILURES:
        print(f"\nPREFLIGHT FAILED ({len(FAILURES)} problems):")
        for failure in FAILURES:
            print(f"  - {failure}")
        sys.exit(1)
    print("\nPREFLIGHT PASSED")


def ok(message):
    print(f"  ok: {message}")


def fail(message):
    print(f"  FAIL: {message}")
    FAILURES.append(message)


def check_environment(load="none"):
    print(f"python {sys.version.split()[0]} at {sys.executable}")
    for name in ["torch", "transformers", "pandas"]:
        try:
            module = __import__(name)
            ok(f"{name} {module.__version__}")
        except ImportError as error:
            fail(f"cannot import {name}: {error}")
    try:
        import transformers

        major = int(transformers.__version__.split(".")[0])
        # the runners pass dtype= to from_pretrained, a v5 argument
        if major < 5:
            fail(
                f"transformers {transformers.__version__} < 5; the "
                f"runners use the dtype= loading argument"
            )
    except ImportError:
        pass
    if load != "none":
        try:
            import bitsandbytes  # pyright: ignore[reportMissingImports]

            ok(f"bitsandbytes {bitsandbytes.__version__} for --load {load}")
        except ImportError as error:
            fail(
                f"cannot import bitsandbytes for --load {load}: {error}; "
                f"install the centaur-eval extra"
            )
        try:
            import torch

            if not torch.cuda.is_available():
                fail(f"--load {load} requires a visible CUDA GPU")
        except ImportError:
            pass
    try:
        import mt.evaluation.transcript_scoring  # noqa: F401

        ok("mt package importable")
    except ImportError as error:
        fail(f"mt package not importable ({error}); run 'uv pip install -e .' in this environment")


def check_data(path):
    if not Path(path).exists():
        fail(f"data file missing: {path}")
        return []
    rows = []
    try:
        with open(path) as handle:
            for number, line in enumerate(handle, start=1):
                rows.append(json.loads(line))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        fail(f"data file corrupt at line {number}: {error}")
        return []
    experiments = {r["experiment"] for r in rows}
    ok(f"data: {len(rows)} sessions, {len(experiments)} experiments")

    from mt.models.llm.supervision import find_target_spans

    unmarked = 0
    empty_spans = 0
    for row in rows:
        spans = find_target_spans(row["text"])
        if not spans:
            unmarked += 1
        elif any(start >= end for start, end in spans):
            empty_spans += 1
    if unmarked:
        fail(f"{unmarked} sessions contain no <<>> markers")
    if empty_spans:
        fail(f"{empty_spans} sessions contain empty <<>> spans")
    if not unmarked and not empty_spans:
        ok("every session has non-empty choice markers")
    return rows


def check_segmentation(rows):
    from mt.evaluation.context_windows import segment_transcript

    broken = {}
    for row in rows:
        try:
            segment_transcript(row["text"])
        except (ValueError, RuntimeError) as error:
            broken.setdefault(row["experiment"], str(error))
    if broken:
        for experiment, error in sorted(broken.items()):
            fail(f"segmentation broken for {experiment}: {error}")
    else:
        ok("segmentation lossless for all sessions (marked-text boundaries only)")


def check_model_cache(model, *, label="model"):
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(model, local_files_only=True))
    except Exception as error:
        fail(f"{label} {model} not in local HF cache: {error}")
        return None
    index_files = list(snapshot.glob("*.safetensors.index.json"))
    weight_files = list(snapshot.glob("*.safetensors"))
    if index_files:
        manifest = json.loads(index_files[0].read_text())
        needed = sorted(set(manifest["weight_map"].values()))
        missing = [
            name
            for name in needed
            if not (snapshot / name).exists() or (snapshot / name).stat().st_size == 0
        ]
        if missing:
            fail(f"model cache incomplete, missing shards: {missing}")
        else:
            ok(f"{label} cache complete: {len(needed)} weight shards")
    elif weight_files:
        ok(f"{label} cache has {len(weight_files)} weight files (no index)")
    else:
        fail(f"no safetensors weights for {label} under {snapshot}")
    return snapshot


def check_tokenizer(model, rows):
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    except Exception as error:
        fail(f"tokenizer load failed from cache: {error}")
        return

    from mt.evaluation.transcript_scoring import _prepare_marked_text
    from types import SimpleNamespace

    # context check only; the model itself is never loaded here
    fake = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=131072))
    sampled = {}
    for row in rows:
        sampled.setdefault(row["experiment"], row)
    broken = []
    for experiment, row in sorted(sampled.items()):
        try:
            _prepare_marked_text(fake, tokenizer, row["text"])
        except ValueError as error:
            broken.append(f"{experiment}: {error}")
    if broken:
        for message in broken:
            fail(f"token span mapping failed for {message}")
    else:
        ok(f"token span mapping verified on {len(sampled)} experiments")


def check_output_dir(output_dir):
    path = Path(output_dir)
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path):
            pass
        ok(f"output dir writable: {path}")
    except OSError as error:
        fail(f"cannot write to {path}: {error}")


def report_gpu():
    import torch

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            memory = properties.total_memory / 2**30
            ok(f"gpu {index}: {properties.name}, {memory:.0f} GiB")
    else:
        print("  note: no GPU visible (expected on a login node)")


def check_attention_backend(seq_len=12000, heads=32, kv_heads=8, head_dim=128,
                            budget_gib=1.0):
    """Fail if long-context attention falls back to the quadratic path.

    Uses the roster's real head layout: 32 query heads over 8 key/value
    heads. The previous version passed one tensor as query, key and value,
    so all three had equal head counts and it could never reproduce the GQA
    mismatch that made both fused kernels refuse the job's actual call.
    It passed while the run OOM-killed 39 of 75 sessions.
    """

    import torch

    if not torch.cuda.is_available():
        return

    from mt.evaluation.transcript_scoring import _cuda_sdpa_context
    from transformers.integrations.sdpa_attention import repeat_kv

    device = torch.device("cuda")
    try:
        query = torch.randn((1, heads, seq_len, head_dim),
                            dtype=torch.float16, device=device)
        key_value = torch.randn((1, kv_heads, seq_len, head_dim),
                                dtype=torch.float16, device=device)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        fail(f"the attention probe ran out of memory before it could start: "
             f"{seq_len} tokens of query/key/value do not fit on this GPU")
        return
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def peak_of(call):
        """Return peak GiB above the inputs, or None if the call OOMs."""

        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device) / 2**30
        torch.cuda.reset_peak_memory_stats(device)
        try:
            with torch.no_grad(), _cuda_sdpa_context(device):
                call()
        except torch.cuda.OutOfMemoryError:
            return None
        finally:
            top = torch.cuda.max_memory_allocated(device) / 2**30
            torch.cuda.empty_cache()
        return top - baseline

    # What load_model's force_sdpa_kv_expansion() makes transformers do. This
    # is the path the run actually takes, so it is the pass/fail criterion.
    expanded = peak_of(lambda: sdpa(query, repeat_kv(key_value, heads // kv_heads),
                                    repeat_kv(key_value, heads // kv_heads),
                                    is_causal=True))
    # What transformers does by itself when the mask is None. Informational:
    # on Volta no fused kernel implements enable_gqa, so this is expected to
    # be huge and is precisely why the expansion is forced.
    broadcast = peak_of(lambda: sdpa(query, key_value, key_value,
                                     is_causal=True, enable_gqa=True))

    shape = f"{heads}q/{kv_heads}kv heads at {seq_len} tokens"
    if broadcast is None or broadcast > budget_gib:
        print(f"  note: GQA broadcast is not fused here "
              f"({'OOM' if broadcast is None else f'{broadcast:.2f} GiB'}); "
              f"load_model expands the KV heads instead")
    if expanded is None:
        fail(f"attention over {shape} ran out of memory even with the KV "
             f"heads expanded; no fused kernel is available on this GPU")
    elif expanded > budget_gib:
        fail(f"attention over {shape} peaked at {expanded:.2f} GiB "
             f"(budget {budget_gib} GiB); this is the quadratic MATH "
             f"kernel, not a fused one")
    else:
        ok(f"long-context attention fused: {shape} in {expanded:.2f} GiB")


if __name__ == "__main__":
    main()
