"""Shared plumbing for experiment runner scripts."""

from __future__ import annotations

from collections import Counter, defaultdict
import gc
import json
from pathlib import Path
import random
from statistics import median

import pandas as pd
import torch

def parse_shard(text):
    """Parse a 'k/n' shard spec into a (k, n) tuple."""

    if text is None:
        return None
    k, n = (int(part) for part in text.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"Invalid shard spec {text!r}: need 0 <= k < n.")
    return k, n

def load_sessions(path, *, experiment=None, participants=None,
                  max_participants=None, seed=0, shard=None, max_chars=None,
                  skip_log=None):
    """Load session rows with optional filtering and seeded sampling."""

    rows = [json.loads(line)
            for line in open(path)]
    if experiment:
        rows = [r
                for r in rows
                if r['experiment'] == experiment]
    if max_chars:
        rows, skipped = filter_by_max_chars(rows, max_chars)
        report_skips(rows + skipped, skipped, max_chars, skip_log)
    if participants:
        rows = rows[:participants]
    if max_participants:
        by_experiment = defaultdict(list)
        for row in rows:
            by_experiment[row['experiment']].append(row)
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
        (kept if len(row['text']) <= max_chars else skipped).append(row)
    return kept, skipped

def report_skips(all_rows, skipped, max_chars, skip_log=None):
    """Print and optionally log which experiments lost sessions."""

    if not skipped:
        return
    totals = Counter(r['experiment'] for r in all_rows)
    by_experiment = defaultdict(list)
    for row in skipped:
        by_experiment[row['experiment']].append(len(row['text']))
    print(f"max-chars {max_chars}: skipped {len(skipped)}/{len(all_rows)} "
          f"sessions across {len(by_experiment)} experiments")
    for experiment in sorted(by_experiment):
        lengths = by_experiment[experiment]
        total = totals[experiment]
        flag = "  <-- ENTIRE TASK DROPPED" if len(lengths) == total else ""
        print(f"  {experiment}: {len(lengths)}/{total} skipped, "
              f"median {int(median(lengths))} chars{flag}")
    if skip_log:
        path = Path(skip_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{'experiment': r['experiment'],
                       'participant': r['participant'],
                       'chars': len(r['text'])} for r in skipped]).to_csv(
            path, index=False)
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
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()

def session_key(row):
    return (str(row['experiment']), str(row['participant']))

def completed_sessions(output_path):
    """Return session keys already present in an output CSV."""

    path = Path(output_path)
    if not path.exists():
        return set()
    frame = pd.read_csv(path, usecols=['experiment', 'participant'],
                        dtype=str)
    return set(map(tuple, frame.drop_duplicates().itertuples(index=False)))

def guard_output(output_path, resume):
    if Path(output_path).exists() and not resume:
        raise SystemExit(f"{output_path} exists; pass --resume to continue "
                         f"into it or remove it first.")

def append_records(output_path, records):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    frame.to_csv(path, mode='a', header=not path.exists(), index=False)

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def resolve_dtype(name, device):
    """Map a dtype name to a torch dtype, with a per-device default."""

    if name == "auto":
        # bf16 on Apple silicon (stable, no fp16 overflow); fp16 on CUDA;
        # fp32 on CPU where half precision is slow and often unsupported
        if device == "mps":
            return torch.bfloat16
        if device == "cuda":
            return torch.float16
        return torch.float32
    return {"fp32": torch.float32, "fp16": torch.float16,
            "bf16": torch.bfloat16}[name]

def load_model(name, dtype, device, load="none"):
    """Load a causal LM, optionally bitsandbytes-quantized (CUDA only)."""

    from transformers import AutoModelForCausalLM

    if load == "none":
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
        return model.to(device).eval()
    if device != "cuda":
        raise SystemExit(f"--load {load} needs a CUDA GPU; bitsandbytes "
                         f"does not support {device}.")

    from transformers import BitsAndBytesConfig

    if load == "4bit":
        quant = BitsAndBytesConfig(load_in_4bit=True,
                                   bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype)
    else:
        quant = BitsAndBytesConfig(load_in_8bit=True)
    # device_map places the quantized weights; no .to() afterwards
    model = AutoModelForCausalLM.from_pretrained(name, device_map="auto",
                                                 quantization_config=quant)
    return model.eval()
