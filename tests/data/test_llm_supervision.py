from __future__ import annotations

import torch

from mt.data import (
    find_target_spans,
    format_record_as_marked_text,
    has_supervised_token,
    make_lm_collate_fn,
    preprocess_marked_text,
    truncate_tokenized_example,
)


class CharTokenizer:
    eos_token = ""
    pad_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ):
        assert truncation is False
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [ord(char) for char in text],
            "offset_mapping": [(index, index + 1) for index, _ in enumerate(text)],
        }


def test_find_target_spans_returns_inner_marker_offsets() -> None:
    assert find_target_spans("A <<left>> then <<right>>") == [(4, 8), (18, 23)]


def test_preprocess_marked_text_supervises_only_marked_tokens() -> None:
    result = preprocess_marked_text("choice <<A>>.", CharTokenizer(), max_length=100)

    supervised_chars = [
        chr(input_id)
        for input_id, label in zip(result["input_ids"], result["labels"], strict=True)
        if label != -100
    ]

    assert supervised_chars == ["A"]
    assert has_supervised_token(result)


def test_truncate_tokenized_example_can_keep_tail() -> None:
    input_ids, labels = truncate_tokenized_example(
        [1, 2, 3, 4],
        [-100, -100, 3, 4],
        max_length=2,
        truncation="tail",
    )

    assert input_ids == [3, 4]
    assert labels == [3, 4]


def test_format_record_as_marked_text_lives_in_data_layer() -> None:
    text = format_record_as_marked_text(
        {"participant": 1, "trial": 2, "response": "left", "rt": 0.51},
        target_columns=("response", "rt"),
    )

    assert text.startswith('trial={"participant":1,"trial":2}')
    assert 'target=<<{"response":"left","rt":0.51}>>' in text


def test_make_lm_collate_fn_pads_inputs_and_labels() -> None:
    collate = make_lm_collate_fn(CharTokenizer())
    batch = collate(
        [
            {"input_ids": [1, 2], "labels": [-100, 2]},
            {"input_ids": [3], "labels": [3]},
        ]
    )

    torch.testing.assert_close(batch["input_ids"], torch.tensor([[1, 2], [3, 0]]))
    torch.testing.assert_close(batch["attention_mask"], torch.tensor([[1, 1], [1, 0]]))
    torch.testing.assert_close(batch["labels"], torch.tensor([[-100, 2], [3, -100]]))
