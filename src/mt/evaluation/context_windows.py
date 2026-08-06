"""Marked-choice-segment text windows for context-length experiments."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from mt.evaluation.transcript_scoring import ChoiceScore, score_marked_texts
from mt.models.llm.supervision import LEFT_TARGET_MARKER, find_target_spans

SEGMENTATION_PROTOCOL = "last-pre-marker-blank-v1"
WINDOW_UNIT = "marked-choice-segment"

_WHITESPACE_BLANK_RUN = re.compile(r"\n(?:[ \t]*\n)+")


@dataclass(frozen=True)
class SegmentedTranscript:
    """Transcript split into a fixed prefix and marked-choice segments.

    ``header`` is a session-specific fixed prefix detected by the public
    segmentation protocol. It is not guaranteed to be a clean instruction
    block for every experiment. Each segment ends with a line containing one
    or more marked choices, and ``tail`` contains text after the final such
    line.
    """

    header: str
    segments: tuple[str, ...]
    tail: str

    def reassemble(self):
        return self.header + "".join(self.segments) + self.tail


def segment_transcript(text) -> SegmentedTranscript:
    """Split a transcript into lossless marked-choice text-window units.

    The fixed prefix ends after the last whitespace-only blank run before the
    first marked choice. If there is no such run, the prefix is empty and all
    pre-choice text belongs to the first segment. This generic fallback is a
    text-window boundary, not a claim that the segments are clean trials.
    """

    boundaries = []
    cursor = 0
    while True:
        marker = text.find(LEFT_TARGET_MARKER, cursor)
        if marker == -1:
            break
        line_end = text.find("\n", marker)
        line_end = len(text) if line_end == -1 else line_end + 1
        boundaries.append(line_end)
        cursor = line_end
    if not boundaries:
        raise ValueError("Transcript contains no marked choices.")

    first_marker = text.find(LEFT_TARGET_MARKER)
    blank_runs = tuple(_WHITESPACE_BLANK_RUN.finditer(text, 0, first_marker))
    header_end = blank_runs[-1].end() if blank_runs else 0
    starts = [header_end] + boundaries[:-1]
    segments = tuple(text[s:e] for s, e in zip(starts, boundaries))

    result = SegmentedTranscript(text[:header_end], segments, text[boundaries[-1] :])
    if result.reassemble() != text:
        raise RuntimeError("Transcript segmentation is not lossless.")
    return result


def build_window_prompt(segmented, target, window) -> str:
    """Return the fixed prefix plus the target and preceding window segments."""

    if not 0 <= target < len(segmented.segments):
        raise ValueError(f"Target {target} is out of range for {len(segmented.segments)} segments.")
    if window < 0:
        raise ValueError(f"Window must be non-negative, got {window}.")
    start = max(0, target - window)
    return segmented.header + "".join(segmented.segments[start : target + 1])


def score_window_choices(
    model,
    tokenizer,
    segmented,
    target,
    window,
    *,
    device="cpu",
    max_batch_tokens=16384,
    top_k=0,
    max_options=0,
) -> list[ChoiceScore]:
    """Score the target marked-choice segment under a truncated context.

    Choice indices remain global to the original transcript. Candidate-option
    support is also taken from that transcript, rather than from the visible
    window.
    """

    return score_window_grid(
        model,
        tokenizer,
        segmented,
        [(target, window)],
        device=device,
        max_batch_tokens=max_batch_tokens,
        top_k=top_k,
        max_options=max_options,
    )[0]


def score_window_grid(
    model,
    tokenizer,
    segmented,
    cells,
    *,
    device="cpu",
    max_batch_tokens=16384,
    top_k=0,
    max_options=0,
) -> list[list[ChoiceScore]]:
    """Score marked-choice ``(target, window)`` cells in packed batches.

    Each cell contains complete ``ChoiceScore`` objects for the target segment
    only. The option set is frozen to every distinct marked choice in the
    original transcript. The support is passed as scorer metadata, so the
    window prompt itself is not modified and the model processes only the
    requested context.
    """

    cells = tuple(cells)
    full_text = segmented.reassemble()
    full_choices = [full_text[start:end] for start, end in find_target_spans(full_text)]
    span_counts = [len(find_target_spans(s)) for s in segmented.segments]
    prompts = []
    choice_selections = []
    for target, window in cells:
        prompts.append(build_window_prompt(segmented, target, window))
        window_start = max(0, target - window)
        local_base = sum(span_counts[window_start:target])
        choice_selections.append(range(local_base, local_base + span_counts[target]))
    scored = score_marked_texts(
        model,
        tokenizer,
        prompts,
        device=device,
        max_batch_tokens=max_batch_tokens,
        top_k=top_k,
        max_options=max_options,
        option_supports=[full_choices] * len(prompts),
        choice_selections=choice_selections,
    )
    results = []
    for (target, window), scores in zip(cells, scored):
        n_target = span_counts[target]
        window_start = max(0, target - window)
        local_base = sum(span_counts[window_start:target])
        global_base = sum(span_counts[:target])
        expected_local_indices = list(range(local_base, local_base + n_target))
        if [score.choice_index for score in scores] != expected_local_indices:
            raise RuntimeError(
                f"Expected local choice indices {expected_local_indices!r} for target "
                f"{target}, got {[score.choice_index for score in scores]!r}."
            )
        results.append(
            [
                replace(score, choice_index=global_base + offset)
                for offset, score in enumerate(scores)
            ]
        )
    return results


def grid_targets(n_segments, n_points, *, strategy="even") -> list[int]:
    """Return marked-choice-segment targets under a named grid strategy."""

    if n_segments <= 0:
        raise ValueError(f"Need at least one segment, got {n_segments}.")
    if n_points <= 0:
        raise ValueError(f"Need at least one grid point, got {n_points}.")
    if strategy not in {"even", "e0-informed"}:
        raise ValueError(f"Unknown grid strategy {strategy!r}.")
    if strategy == "e0-informed":
        return _e0_informed_targets(n_segments, n_points)
    if n_points >= n_segments:
        return list(range(n_segments))
    if n_points == 1:
        return [n_segments // 2]
    step = (n_segments - 1) / (n_points - 1)
    return sorted({round(i * step) for i in range(n_points)})


def _e0_informed_targets(n_segments, n_points):
    """Return the preregistered five anchors motivated by the E0 curve."""

    if n_points != 5:
        raise ValueError("The e0-informed strategy requires exactly 5 points.")
    if n_segments <= n_points:
        return list(range(n_segments))

    last = n_segments - 1
    raw = [0, 1, 0.10 * last, 0.50 * last, last]
    targets = []
    for index, value in enumerate(raw):
        lower = targets[-1] + 1 if targets else 0
        upper = n_segments - (len(raw) - index)
        targets.append(min(max(round(value), lower), upper))
    return targets
