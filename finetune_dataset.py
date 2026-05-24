import random

import torch
from torch.nn.utils.rnn import pad_sequence


def find_choice_spans(text):
    """
    找出 << 和 >> 中间的人类选择。
    返回 char-level spans，例如 [(start_char, end_char), ...]
    """
    spans = []
    pos = 0

    while True:
        left = text.find("<<", pos)
        if left == -1:
            break

        right = text.find(">>", left + 2)
        if right == -1:
            break

        spans.append((left + 2, right))
        pos = right + 2

    return spans

def preprocess_example(example, tokenizer,MaxLen=512):
    text = example["text"]

    if tokenizer.eos_token is not None:
        text = text + tokenizer.eos_token

    choice_spans = find_choice_spans(text)

    encoded = tokenizer(
        text,
        truncation=False,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    full_input_ids = encoded["input_ids"]
    full_offsets = encoded["offset_mapping"]

    full_labels = [-100] * len(full_input_ids)

    for i, (tok_start, tok_end) in enumerate(full_offsets):
        if tok_start == tok_end:
            continue

        for span_start, span_end in choice_spans:
            overlap = tok_start < span_end and tok_end > span_start
            if overlap:
                full_labels[i] = full_input_ids[i]
                break

    if len(full_input_ids) > MaxLen:
        start = random.randint(0, len(full_input_ids) - MaxLen)
        end = start + MaxLen
        input_ids = full_input_ids[start:end]
        labels = full_labels[start:end]
    else:
        input_ids = full_input_ids
        labels = full_labels

    return {
        "input_ids": input_ids,
        "labels": labels,
    }

def has_supervised_token(example):
    return any(x != -100 for x in example["labels"])


def make_collate_fn(tokenizer):
    def collate_fn(batch):
        input_ids = [
            torch.tensor(item["input_ids"], dtype=torch.long)
            for item in batch
        ]

        labels = [
            torch.tensor(item["labels"], dtype=torch.long)
            for item in batch
        ]

        attention_mask = [
            torch.ones_like(x, dtype=torch.long)
            for x in input_ids
        ]

        input_ids = pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id,
        )

        labels = pad_sequence(
            labels,
            batch_first=True,
            padding_value=-100,
        )

        attention_mask = pad_sequence(
            attention_mask,
            batch_first=True,
            padding_value=0,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return collate_fn

#用于处理输入是文本的函数
def preprocess_input(text, tokenizer,MaxLen=512):

    if tokenizer.eos_token is not None:
        text = text + tokenizer.eos_token

    choice_spans = find_choice_spans(text)

    encoded = tokenizer(
        text,
        truncation=False,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    full_input_ids = encoded["input_ids"]
    full_offsets = encoded["offset_mapping"]

    full_labels = [-100] * len(full_input_ids)

    for i, (tok_start, tok_end) in enumerate(full_offsets):
        if tok_start == tok_end:
            continue

        for span_start, span_end in choice_spans:
            overlap = tok_start < span_end and tok_end > span_start
            if overlap:
                full_labels[i] = full_input_ids[i]
                break

    if len(full_input_ids) > MaxLen:
        start = random.randint(0, len(full_input_ids) - MaxLen)
        end = start + MaxLen
        input_ids = full_input_ids[start:end]
        labels = full_labels[start:end]
    else:
        input_ids = full_input_ids
        labels = full_labels

    return {
        "input_ids": input_ids,
        "labels": labels,
    }