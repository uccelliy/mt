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
    parser.add_argument("--model", default="marcelbinz/Llama-3.1-Minitaur-8B",
                        help="HF model name expected in the local cache")
    parser.add_argument("--data",
                        default="data/psych-101-test/prompts_testing_t1.jsonl",
                        help="Path to prompts .jsonl")
    parser.add_argument("--output-dir", default="outputs/scoring",
                        help="Directory the runners will write to")
    parser.add_argument("--load", default="none",
                        choices=["none", "8bit", "4bit"],
                        help="Validate an optional quantized CUDA runtime")
    args = parser.parse_args()

    check_environment(args.load)
    rows = check_data(args.data)
    if rows:
        check_segmentation(rows)
    snapshot = check_model_cache(args.model)
    if rows and snapshot:
        check_tokenizer(args.model, rows)
    check_output_dir(args.output_dir)
    report_gpu()

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
            fail(f"transformers {transformers.__version__} < 5; the "
                 f"runners use the dtype= loading argument")
    except ImportError:
        pass
    if load != "none":
        try:
            import bitsandbytes # pyright: ignore[reportMissingImports]

            ok(f"bitsandbytes {bitsandbytes.__version__} for --load {load}")
        except ImportError as error:
            fail(f"cannot import bitsandbytes for --load {load}: {error}; "
                 f"install the centaur-eval extra")
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
        fail(f"mt package not importable ({error}); "
             f"run 'uv pip install -e .' in this environment")

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
    experiments = {r['experiment'] for r in rows}
    ok(f"data: {len(rows)} sessions, {len(experiments)} experiments")

    from mt.data._llm_supervision import find_target_spans

    unmarked = 0
    empty_spans = 0
    for row in rows:
        spans = find_target_spans(row['text'])
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
            segment_transcript(row['text'])
        except (ValueError, RuntimeError) as error:
            broken.setdefault(row['experiment'], str(error))
    if broken:
        for experiment, error in sorted(broken.items()):
            fail(f"segmentation broken for {experiment}: {error}")
    else:
        ok("segmentation lossless for all sessions (E3 safe)")

def check_model_cache(model):
    try:
        from huggingface_hub import snapshot_download

        snapshot = Path(snapshot_download(model, local_files_only=True))
    except Exception as error:
        fail(f"model {model} not in local HF cache: {error}")
        return None
    index_files = list(snapshot.glob("*.safetensors.index.json"))
    weight_files = list(snapshot.glob("*.safetensors"))
    if index_files:
        manifest = json.loads(index_files[0].read_text())
        needed = sorted(set(manifest['weight_map'].values()))
        missing = [name for name in needed
                   if not (snapshot / name).exists()
                   or (snapshot / name).stat().st_size == 0]
        if missing:
            fail(f"model cache incomplete, missing shards: {missing}")
        else:
            ok(f"model cache complete: {len(needed)} weight shards")
    elif weight_files:
        ok(f"model cache has {len(weight_files)} weight files (no index)")
    else:
        fail(f"no safetensors weights under {snapshot}")
    return snapshot

def check_tokenizer(model, rows):
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model,
                                                  local_files_only=True)
    except Exception as error:
        fail(f"tokenizer load failed from cache: {error}")
        return

    from mt.evaluation.transcript_scoring import _prepare_marked_text
    from types import SimpleNamespace

    # context check only; the model itself is never loaded here
    fake = SimpleNamespace(config=SimpleNamespace(
        max_position_embeddings=131072))
    sampled = {}
    for row in rows:
        sampled.setdefault(row['experiment'], row)
    broken = []
    for experiment, row in sorted(sampled.items()):
        try:
            _prepare_marked_text(fake, tokenizer, row['text'])
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

if __name__ == "__main__":
    main()
