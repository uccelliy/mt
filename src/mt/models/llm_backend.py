"""Utilities for loading causal language models across local backends."""

from __future__ import annotations

import importlib.util
from typing import Any

import torch


def load_causal_lm(
    model_name: str,
    *,
    backend: str = "auto",
    max_seq_length: int = 32768,
    dtype: Any = None,
    load_in_4bit: bool = True,
    for_inference: bool = False,
    trust_remote_code: bool = True,
    hf_model_kwargs: dict[str, Any] | None = None,
    hf_tokenizer_kwargs: dict[str, Any] | None = None,
):
    """Load a causal LM with Unsloth when available, otherwise Transformers.

    ``backend="auto"`` selects Unsloth only when CUDA is available and the
    optional dependency is installed. This keeps Mac/CPU environments importable
    without sprinkling platform checks throughout evaluation scripts.
    """

    backend = backend.lower()
    if backend not in {"auto", "unsloth", "hf", "transformers"}:
        raise ValueError(f"Unknown backend {backend!r}. Use 'auto', 'unsloth', or 'hf'.")

    if backend == "auto":
        backend = "unsloth" if _can_use_unsloth() else "hf"
    elif backend == "transformers":
        backend = "hf"

    if backend == "unsloth":
        return _load_unsloth(
            model_name,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            for_inference=for_inference,
        )

    return _load_transformers(
        model_name,
        trust_remote_code=trust_remote_code,
        hf_model_kwargs=hf_model_kwargs,
        hf_tokenizer_kwargs=hf_tokenizer_kwargs,
    )


def get_is_quantized(model) -> bool:
    return bool(getattr(model, "is_quantized", False))


def set_is_quantized(model, value: bool) -> None:
    if hasattr(model, "is_quantized"):
        model.is_quantized = value


def _can_use_unsloth() -> bool:
    return torch.cuda.is_available() and importlib.util.find_spec("unsloth") is not None


def _load_unsloth(
    model_name: str,
    *,
    max_seq_length: int,
    dtype: Any,
    load_in_4bit: bool,
    for_inference: bool,
):
    if importlib.util.find_spec("unsloth") is None:
        raise ImportError("backend='unsloth' requested, but the unsloth package is not installed.")
    if not torch.cuda.is_available():
        raise RuntimeError("backend='unsloth' requested, but CUDA is not available.")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    if for_inference:
        FastLanguageModel.for_inference(model)
    return model, tokenizer


def _load_transformers(
    model_name: str,
    *,
    trust_remote_code: bool,
    hf_model_kwargs: dict[str, Any] | None,
    hf_tokenizer_kwargs: dict[str, Any] | None,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_kwargs = dict(hf_model_kwargs or {})
    tokenizer_kwargs = dict(hf_tokenizer_kwargs or {})
    model_kwargs.setdefault("trust_remote_code", trust_remote_code)
    tokenizer_kwargs.setdefault("trust_remote_code", trust_remote_code)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    return model, tokenizer
