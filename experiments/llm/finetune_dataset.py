"""Compatibility wrappers for the finetuning dataset helpers."""

from mt.data import (
    find_target_spans,
    has_supervised_token as has_supervised_token,
    make_lm_collate_fn,
    preprocess_marked_text,
    preprocess_supervised_example,
)


def find_choice_spans(text):
    return find_target_spans(text)


def preprocess_example(example, tokenizer, MaxLen=512):
    return preprocess_supervised_example(example, tokenizer, MaxLen)


def preprocess_input(text, tokenizer, MaxLen=512):
    return preprocess_marked_text(text, tokenizer, MaxLen)


def make_collate_fn(tokenizer):
    return make_lm_collate_fn(tokenizer)
