"""Tests for window-truncated prompt reconstruction."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from mt.evaluation.context_windows import (
    build_window_prompt,
    grid_targets,
    score_window_choices,
    segment_transcript,
)

VOCAB_SIZE = 256

TEXT = ("intro one\nintro two\n\n"
        "A <<x>>. out 1\nB <<y>>. out 2\nnote\nC <<z>>. end")

class CharTokenizer:
    """Character-level tokenizer with offset mapping, no special tokens."""

    bos_token_id = None

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=True):
        return {
            'input_ids': [ord(c) % VOCAB_SIZE for c in text],
            'offset_mapping': [(i, i + 1) for i in range(len(text))],
        }

class UniformModel:
    """Callable returning uniform logits over the vocabulary."""

    config = SimpleNamespace(max_position_embeddings=4096)

    def __call__(self, ids, attention_mask=None):
        batch, length = ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, length, VOCAB_SIZE))

def test_segment_transcript_is_lossless_and_splits_on_choice_lines():
    segmented = segment_transcript(TEXT)
    assert segmented.header == "intro one\nintro two\n\n"
    assert len(segmented.segments) == 3
    assert segmented.segments[0] == "A <<x>>. out 1\n"
    assert segmented.segments[2] == "note\nC <<z>>. end"
    assert segmented.tail == ""
    assert segmented.reassemble() == TEXT

def test_segment_transcript_keeps_trailing_newline():
    segmented = segment_transcript(TEXT + "\ndone\n")
    assert segmented.tail == "done\n"
    assert segmented.reassemble() == TEXT + "\ndone\n"

def test_segment_transcript_rejects_unmarked_text():
    try:
        segment_transcript("no markers here")
    except ValueError as error:
        assert "no marked choices" in str(error)
    else:
        raise AssertionError("expected ValueError for unmarked text")

def test_build_window_prompt_zero_window_is_header_plus_target():
    segmented = segment_transcript(TEXT)
    prompt = build_window_prompt(segmented, 2, 0)
    assert prompt == segmented.header + "note\nC <<z>>. end"

def test_build_window_prompt_window_covers_earlier_segments():
    segmented = segment_transcript(TEXT)
    assert build_window_prompt(segmented, 2, 1) == (
        segmented.header + segmented.segments[1] + segmented.segments[2])
    assert build_window_prompt(segmented, 2, 99) == TEXT

def test_build_window_prompt_validates_arguments():
    segmented = segment_transcript(TEXT)
    for target, window in [(3, 0), (-1, 0), (0, -1)]:
        try:
            build_window_prompt(segmented, target, window)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {target}, {window}")

def test_score_window_choices_reports_global_choice_index():
    segmented = segment_transcript(TEXT)
    records = score_window_choices(UniformModel(), CharTokenizer(),
                                   segmented, 2, 0)
    assert len(records) == 1
    assert records[0]['target_index'] == 2
    assert records[0]['choice_index'] == 2
    assert math.isclose(records[0]['nll'], math.log(VOCAB_SIZE),
                        rel_tol=1e-5)

def test_grid_targets_covers_session_bounds():
    grid = grid_targets(100, 5)
    assert grid[0] == 0 and grid[-1] == 99 and len(grid) == 5
    assert grid == sorted(set(grid))
    assert grid_targets(3, 10) == [0, 1, 2]
    assert grid_targets(9, 1) == [4]
