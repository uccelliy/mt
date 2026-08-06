"""Tests for window-truncated prompt reconstruction."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from mt.evaluation.context_windows import (
    SEGMENTATION_PROTOCOL,
    WINDOW_UNIT,
    build_window_prompt,
    grid_targets,
    score_window_grid,
    score_window_choices,
    segment_transcript,
)
from mt.evaluation.transcript_scoring import score_marked_text

VOCAB_SIZE = 256

TEXT = "intro one\nintro two\n\nA <<x>>. out 1\nB <<y>>. out 2\nnote\nC <<z>>. end"


class CharTokenizer:
    """Character-level tokenizer with offset mapping, no special tokens."""

    bos_token_id = None

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=True):
        return {
            "input_ids": [ord(c) % VOCAB_SIZE for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }

    def convert_ids_to_tokens(self, ids):
        return [chr(token_id) for token_id in ids]

    def decode(self, ids, **kwargs):
        return "".join(chr(token_id) for token_id in ids)


class UniformModel:
    """Callable returning uniform logits over the vocabulary."""

    config = SimpleNamespace(max_position_embeddings=4096)

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        batch, length = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, length, VOCAB_SIZE))


class XGreedyModel(UniformModel):
    """Model whose unconstrained greedy token is always ``x``."""

    def __init__(self):
        self.seen_widths = []

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        self.seen_widths.append(input_ids.shape[1])
        output = super().__call__(input_ids, attention_mask=attention_mask, **kwargs)
        output.logits[..., ord("x")] = 2.0
        return output


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


@pytest.mark.parametrize(
    ("text", "target", "old_state", "current_state"),
    [
        (
            "Choose directions using your assigned keys.\n\n"
            "The new starting station is A.\n"
            "You press <<N>>.\n"
            "You are successful.\n\n"
            "The new starting station is B.\n"
            "You press <<S>>.",
            1,
            "starting station is A",
            "starting station is B",
        ),
        (
            "Repeat the digits, then press your assigned end key.\n \t \n"
            "The digits are 1 2.\n"
            "You press <<1>>.\n"
            "You press <<2>>.\n"
            "You press <<#>>.\n\n"
            "The digits are 3.\n"
            "You press <<3>>.\n"
            "You press <<#>>.",
            3,
            "digits are 1 2",
            "digits are 3",
        ),
    ],
)
def test_first_trial_state_is_not_permanently_kept_in_fixed_prefix(
    text, target, old_state, current_state
):
    segmented = segment_transcript(text)
    prompt = build_window_prompt(segmented, target, 0)

    assert old_state not in segmented.header
    assert old_state not in prompt
    assert current_state in prompt
    assert segmented.reassemble() == text


def test_segment_transcript_without_pre_marker_blank_uses_empty_prefix():
    text = "Pre-task question\nYou choose <<A>>.\nNext question\nYou choose <<B>>."
    segmented = segment_transcript(text)

    assert segmented.header == ""
    assert segmented.segments[0] == "Pre-task question\nYou choose <<A>>.\n"
    assert segmented.reassemble() == text


def test_segmentation_protocol_metadata_is_public_and_versioned():
    assert SEGMENTATION_PROTOCOL == "last-pre-marker-blank-v1"
    assert WINDOW_UNIT == "marked-choice-segment"


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
        segmented.header + segmented.segments[1] + segmented.segments[2]
    )
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
    scores = score_window_choices(UniformModel(), CharTokenizer(), segmented, 2, 0)
    assert len(scores) == 1
    assert scores[0].choice_index == 2
    assert math.isclose(scores[0].nll, math.log(VOCAB_SIZE), rel_tol=1e-5)


def test_score_window_grid_accepts_generator_cells():
    segmented = segment_transcript(TEXT)
    cells = ((target, 0) for target in (1, 2))

    results = score_window_grid(UniformModel(), CharTokenizer(), segmented, cells)

    assert [[score.choice_index for score in scores] for scores in results] == [[1], [2]]


def test_window_scores_keep_full_raw_output_and_freeze_full_option_support():
    segmented = segment_transcript(TEXT)
    model = XGreedyModel()
    scores = score_window_choices(
        model,
        CharTokenizer(),
        segmented,
        2,
        0,
        top_k=3,
        max_options=10,
    )

    assert len(scores) == 1
    score = scores[0]
    assert score.choice_index == 2
    assert score.human_choice == "z"
    assert score.pred_choice == score.raw_generation == "x"
    assert score.pred_token_id == ord("x")
    # x is absent from the visible zero-window prompt, so this proves that
    # format validation and option scores use the original full transcript.
    assert score.format_ok
    assert score.k_options == 3
    assert score.options_status == "scored"
    assert {option.option for option in score.options} == {"x", "y", "z"}
    assert {option.option for option in score.options if option.is_human} == {"z"}
    assert len(score.topk) == 3
    assert score.topk[0].token_id == ord("x")
    assert model.seen_widths == [len(build_window_prompt(segmented, 2, 0))]


def test_full_window_target_matches_the_same_choice_from_full_transcript():
    tokenizer = CharTokenizer()
    full = score_marked_text(XGreedyModel(), tokenizer, TEXT, top_k=3, max_options=10)
    window = score_window_choices(
        XGreedyModel(),
        tokenizer,
        segment_transcript(TEXT),
        target=2,
        window=2,
        top_k=3,
        max_options=10,
    )

    assert window == [full[2]]


def test_grid_targets_covers_session_bounds():
    grid = grid_targets(100, 5)
    assert grid[0] == 0 and grid[-1] == 99 and len(grid) == 5
    assert grid == sorted(set(grid))
    assert grid_targets(3, 10) == [0, 1, 2]
    assert grid_targets(9, 1) == [4]


def test_e0_informed_grid_preserves_early_and_longitudinal_anchors():
    assert grid_targets(100, 5, strategy="e0-informed") == [
        0,
        1,
        10,
        50,
        99,
    ]
    assert grid_targets(6, 5, strategy="e0-informed") == [0, 1, 2, 3, 5]
    assert grid_targets(3, 5, strategy="e0-informed") == [0, 1, 2]


def test_grid_targets_validates_strategy_and_e0_point_count():
    with pytest.raises(ValueError, match="Unknown grid strategy"):
        grid_targets(10, 5, strategy="unknown")
    with pytest.raises(ValueError, match="exactly 5"):
        grid_targets(10, 4, strategy="e0-informed")
