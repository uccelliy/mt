"""Window-truncated prompt reconstruction for context-length experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mt.data._llm_supervision import LEFT_TARGET_MARKER, find_target_spans
from mt.evaluation.transcript_scoring import score_marked_texts

@dataclass(frozen=True)
class SegmentedTranscript:
    """Transcript split into instruction header and per-choice segments."""

    header: str
    segments: tuple[str, ...]
    tail: str

    def reassemble(self):
        return self.header + "".join(self.segments) + self.tail

def segment_transcript(text) -> SegmentedTranscript:
    """Split a transcript at choice lines, keeping every character."""

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
    header_end = text.rfind("\n", 0, first_marker) + 1
    starts = [header_end] + boundaries[:-1]
    segments = tuple(text[s:e] 
                     for s, e in zip(starts, boundaries))

    result = SegmentedTranscript(text[:header_end], segments,
                                 text[boundaries[-1]:])
    if result.reassemble() != text:
        raise RuntimeError("Transcript segmentation is not lossless.")
    return result

def build_window_prompt(segmented, target, window) -> str:
    """Return header plus the last `window` segments before the target."""

    if not 0 <= target < len(segmented.segments):
        raise ValueError(f"Target {target} is out of range for "
                         f"{len(segmented.segments)} segments.")
    if window < 0:
        raise ValueError(f"Window must be non-negative, got {window}.")
    start = max(0, target - window)
    return segmented.header + "".join(segmented.segments[start:target + 1])

def score_window_choices(model, tokenizer, segmented, target, window, *,
                         device="cpu") -> list[dict[str, Any]]:
    """Score the target segment's choices under a truncated context."""

    return score_window_grid(model, tokenizer, segmented,
                             [(target, window)], device=device)[0]

def score_window_grid(model, tokenizer, segmented, cells, *, device="cpu",
                      max_batch_tokens=16384) -> list[list[dict[str, Any]]]:
    """Score (target, window) cells of one session in packed batches."""

    prompts = [build_window_prompt(segmented, target, window)
               for target, window in cells]
    scored = score_marked_texts(model, tokenizer, prompts, device=device,
                                max_batch_tokens=max_batch_tokens)
    span_counts = [len(find_target_spans(s)) for s in segmented.segments]
    results = []
    for (target, window), scores in zip(cells, scored):
        n_target = span_counts[target]
        base = sum(span_counts[:target])
        records = []
        for offset, score in enumerate(scores[-n_target:]):
            records.append({'target_index': target,
                            'choice_index': base + offset,
                            'nll': score.nll,
                            'num_tokens': score.num_tokens})
        results.append(records)
    return results

def grid_targets(n_segments, n_points) -> list[int]:
    """Return evenly spaced target indices covering the session."""

    if n_points <= 0:
        raise ValueError(f"Need at least one grid point, got {n_points}.")
    if n_points >= n_segments:
        return list(range(n_segments))
    if n_points == 1:
        return [n_segments // 2]
    step = (n_segments - 1) / (n_points - 1)
    return sorted({round(i * step) for i in range(n_points)})
