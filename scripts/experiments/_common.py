"""Shared plumbing for experiment runner scripts."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random

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
                  max_participants=None, seed=0, shard=None):
    """Load session rows with optional filtering and seeded sampling."""

    rows = [json.loads(line) for line in open(path)]
    if experiment:
        rows = [r for r in rows if r['experiment'] == experiment]
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
