"""Shared plumbing for experiment runner scripts."""

from __future__ import annotations

from collections import Counter, defaultdict
import gc
import json
import os
from pathlib import Path
import random
from statistics import median
import time

import pandas as pd
import torch

from mt.evaluation.transcript_scoring import ContextLengthError


def parse_shard(text):
    """Parse a 'k/n' shard spec into a (k, n) tuple."""

    if text is None:
        return None
    k, n = (int(part) for part in text.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"Invalid shard spec {text!r}: need 0 <= k < n.")
    return k, n


def load_sessions(
    path,
    *,
    experiment=None,
    participant=None,
    participants=None,
    max_participants=None,
    seed=0,
    shard=None,
    max_chars=None,
    skip_log=None,
):
    """Load session rows with optional filtering and seeded sampling."""

    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    if experiment:
        rows = [r for r in rows if r["experiment"] == experiment]
    if participant is not None:
        rows = [r for r in rows if str(r["participant"]) == str(participant)]
    if max_chars:
        rows, skipped = filter_by_max_chars(rows, max_chars)
        report_skips(rows + skipped, skipped, max_chars, skip_log)
    if participants:
        rows = rows[:participants]
    if max_participants:
        by_experiment = defaultdict(list)
        for row in rows:
            by_experiment[row["experiment"]].append(row)
        rng = random.Random(seed)
        rows = []
        for exp in sorted(by_experiment):
            group = by_experiment[exp]
            if len(group) > max_participants:
                group = rng.sample(group, max_participants)
            rows.extend(group)
    if shard:
        k, n = shard
        rows = rows[k::n]
    if not rows:
        raise SystemExit("No sessions matched the filters.")
    return rows


def filter_by_max_chars(rows, max_chars):
    """Partition rows into those within and over a character budget."""

    kept, skipped = [], []
    for row in rows:
        (kept if len(row["text"]) <= max_chars else skipped).append(row)
    return kept, skipped


def report_skips(all_rows, skipped, max_chars, skip_log=None):
    """Print and optionally log which experiments lost sessions."""

    if not skipped:
        return
    totals = Counter(r["experiment"] for r in all_rows)
    by_experiment = defaultdict(list)
    for row in skipped:
        by_experiment[row["experiment"]].append(len(row["text"]))
    print(
        f"max-chars {max_chars}: skipped {len(skipped)}/{len(all_rows)} "
        f"sessions across {len(by_experiment)} experiments"
    )
    for experiment in sorted(by_experiment):
        lengths = by_experiment[experiment]
        total = totals[experiment]
        flag = "  <-- ENTIRE TASK DROPPED" if len(lengths) == total else ""
        print(
            f"  {experiment}: {len(lengths)}/{total} skipped, "
            f"median {int(median(lengths))} chars{flag}"
        )
    if skip_log:
        path = Path(skip_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "experiment": r["experiment"],
                    "participant": r["participant"],
                    "chars": len(r["text"]),
                }
                for r in skipped
            ]
        ).to_csv(path, index=False)
        print(f"  wrote skip log to {path}")


def skip_log_for(output_path):
    path = Path(output_path)
    return path.with_name(path.stem + ".skipped.csv")


def failure_log_for(output_path):
    path = Path(output_path)
    return path.with_name(path.stem + ".failed.csv")


def empty_device_cache(device):
    """Force collection and return freed memory to the device pool."""

    gc.collect()
    device_type = torch.device(device).type
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps":
        torch.mps.empty_cache()


def is_device_out_of_memory(error):
    """Return whether an exception is a recognized device-memory failure."""

    candidates = [
        getattr(torch, "OutOfMemoryError", None),
        getattr(torch.cuda, "OutOfMemoryError", None),
    ]
    mps = getattr(torch, "mps", None)
    candidates.append(getattr(mps, "OutOfMemoryError", None))
    oom_types = tuple({candidate for candidate in candidates if isinstance(candidate, type)})
    if oom_types and isinstance(error, oom_types):
        return True
    message = str(error).lower()
    markers = (
        "cuda out of memory",
        "mps backend out of memory",
        "hip out of memory",
        "defaultcpuallocator: not enough memory",
    )
    return any(marker in message for marker in markers)


def is_session_failure(error):
    """Return whether a per-session error is loggable rather than fatal."""

    return isinstance(error, ContextLengthError) or is_device_out_of_memory(error)


def log_session_failure(failures, row, error, device):
    """Record one unscorable session in the failure log and free memory."""

    empty_device_cache(device)
    append_records(
        failures,
        [
            {
                "experiment": row["experiment"],
                "participant": row["participant"],
                "chars": len(row["text"]),
                "error": str(error)[:200],
            }
        ],
    )
    reason = "context overflow" if isinstance(error, ContextLengthError) else "OOM"
    print(
        f"  {reason} on {row['experiment']} p{row['participant']} "
        f"({len(row['text'])} chars): logged and skipped",
        flush=True,
    )


def session_key(row):
    return (str(row["experiment"]), str(row["participant"]))


def completed_sessions(output_path):
    """Return session keys already present in an output CSV."""

    path = Path(output_path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        frame = pd.read_csv(path, usecols=["experiment", "participant"], dtype=str)
    except pd.errors.EmptyDataError:
        return set()
    return set(map(tuple, frame.drop_duplicates().itertuples(index=False)))


def guard_output(output_path, resume):
    if Path(output_path).exists() and not resume:
        raise SystemExit(
            f"{output_path} exists; pass --resume to continue into it or remove it first."
        )


def append_records(output_path, records):
    if not records:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def fsync_directory(path):
    """Persist directory-entry changes on POSIX filesystems."""

    if os.name == "nt":
        return
    descriptor = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_files(paths):
    """Persist existing files and their directory entries before a receipt."""

    parents = set()
    for value in paths:
        path = Path(value)
        if not path.exists():
            continue
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        parents.add(path.parent)
    for parent in sorted(parents, key=os.fspath):
        fsync_directory(parent)


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(name, device):
    """Map a dtype name to a torch dtype, with a per-device default."""

    if name == "auto":
        device_type = torch.device(device).type
        # bf16 on Apple silicon (stable, no fp16 overflow); fp16 on CUDA;
        # fp32 on CPU where half precision is slow and often unsupported
        if device_type == "mps":
            return torch.bfloat16
        if device_type == "cuda":
            return torch.float16
        return torch.float32
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def prefetch_checkpoint(name, chunk_bytes=8 << 20):
    """Stream the checkpoint once so the loader's mmap faults hit page cache.

    safetensors mmaps each shard and transformers materializes tensor by
    tensor, so every page arrives as a small high-latency fault with no
    readahead -- pathological on a network filesystem. Measured on iris:
    the same shard streams at 906 MB/s under sequential reads while the
    loader crawled at roughly 5 MB/s, which turned a 16 GB checkpoint into
    a 50-minute load and timed out the one-hour smoke gate before it scored
    a single session. One sequential pass costs about 20 seconds and leaves
    the pages cached for the mmap that follows.

    Returns the bytes read, or 0 when the snapshot cannot be resolved --
    this is an optimization, never a precondition, so any failure is silent
    and loading proceeds normally.
    """

    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(name, local_files_only=True))
    except Exception:
        return 0
    total = 0
    for path in sorted(snapshot.glob("*.safetensors")):
        try:
            with open(path, "rb", buffering=0) as handle:
                while True:
                    block = handle.read(chunk_bytes)
                    if not block:
                        break
                    total += len(block)
        except OSError:
            continue
    return total


def library_versions():
    """Record the stack underneath this repo, which drifts on its own.

    §8.3 requires comparable runs to keep their configuration with the
    results. The manifest's `commit` pins this repo; nothing pinned the
    libraries that actually do the arithmetic. That gap is not theoretical:
    the login node had transformers 5.10 and the cluster 5.14, which changed
    the signature of an internal SDPA predicate and killed a run.
    """

    import transformers

    versions = {"torch": torch.__version__,
                "transformers": transformers.__version__}
    for name in ("bitsandbytes", "peft", "pandas"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            continue
    return versions


def force_sdpa_kv_expansion():
    """Materialize GQA key/value heads before SDPA on pre-Ampere GPUs.

    transformers skips `repeat_kv` and asks SDPA for `enable_gqa=True`
    whenever the attention mask is None -- see `use_gqa_in_sdpa` in
    transformers/integrations/sdpa_attention.py. Only the flash and cuDNN
    backends implement `enable_gqa`, and Volta has neither, so SDPA refused
    both fused kernels ("For dense input, both fused kernels require query,
    key and value to have the same num_heads") and dropped to the math
    kernel, which materializes [1, 32, L, L] in fp32: 10.84 GiB at 9,536
    tokens. That is what OOM-killed 39 of the 75 smoke sessions.

    The unpadded single-session forward is exactly the case where the mask
    is None, so the fast path was the one that broke. Expanding 8 KV heads
    to 32 costs ~80 MB and lets the memory-efficient kernel run: measured
    0.09 GiB at 12k tokens on a V100.

    Left alone from sm_80 up, where flash implements `enable_gqa` and
    skipping the expansion is genuinely the faster path.
    """

    if not torch.cuda.is_available():
        return False
    if torch.cuda.get_device_capability()[0] >= 8:
        return False
    from transformers.integrations import sdpa_attention

    # Signature-agnostic on purpose: this predicate took (attention_mask, key)
    # in transformers 5.10 and (attention_mask, key, value) in 5.14, and the
    # answer does not depend on the arguments.
    sdpa_attention.use_gqa_in_sdpa = lambda *_args, **_kwargs: False
    return True


def load_model(name, dtype, device, load="none", adapter=None):
    """Load a causal LM, optionally quantized and/or with a LoRA adapter."""

    from transformers import AutoModelForCausalLM

    if force_sdpa_kv_expansion():
        print("sdpa: expanding GQA key/value heads (pre-Ampere GPU)", flush=True)

    started = time.monotonic()
    warmed = prefetch_checkpoint(name)
    if warmed:
        elapsed = time.monotonic() - started
        rate = warmed / 2**20 / max(elapsed, 1e-6)
        print(f"prefetched {warmed / 2**30:.1f} GiB of weights in "
              f"{elapsed:.0f}s ({rate:.0f} MB/s)", flush=True)

    # Pinned, never left to the default. Volta has neither FlashAttention nor
    # cuDNN attention, so a fall back to eager materializes [batch, heads, L, L]
    # in fp32 for its softmax -- 17 GiB for a 12k-token session, which is what
    # OOM-killed 39 of the first 75 smoke sessions. sdpa lets
    # _cuda_sdpa_context pick the memory-efficient kernel instead.
    if load == "none":
        model = AutoModelForCausalLM.from_pretrained(
            name, dtype=dtype, attn_implementation="sdpa"
        )
        model = model.to(device)
    else:
        if torch.device(device).type != "cuda":
            raise SystemExit(
                f"--load {load} needs a CUDA GPU; this runner "
                f"does not enable bitsandbytes quantization "
                f"on {device}."
            )

        from transformers import BitsAndBytesConfig

        if load == "4bit":
            quant = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
        else:
            quant = BitsAndBytesConfig(load_in_8bit=True)
        # device_map places the quantized weights; no .to() afterwards
        # dtype= is not redundant with bnb_4bit_compute_dtype: it also sets the
        # modules bitsandbytes leaves alone (norms, embeddings, LM head), which
        # would otherwise stay at the checkpoint's bf16 and quietly break the
        # fp16 the protocol pins (design §8.3).
        model = AutoModelForCausalLM.from_pretrained(
            name,
            device_map="auto",
            quantization_config=quant,
            dtype=dtype,
            attn_implementation="sdpa",
        )
    if adapter:
        # adapter-on-quantized-base matches the official Centaur evaluation;
        # merging into BF16 and requantizing degrades the adapter instead
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    # Print it rather than trust it: the eager fall back is invisible until a
    # long session OOMs, and the number in that OOM message is the only other
    # way to tell which kernel ran.
    resolved = getattr(model.config, "_attn_implementation", "unknown")
    print(f"attention implementation: {resolved}", flush=True)
    if resolved != "sdpa":
        raise SystemExit(
            f"refusing to score with attn_implementation={resolved!r}: on Volta "
            f"anything but sdpa materializes the full attention matrix and "
            f"OOMs on long sessions."
        )
    return model.eval()
