"""Teacher-forced per-choice NLL scoring for Psych-101-style transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from mt.data._llm_supervision import find_target_spans

@dataclass(frozen=True)
class ChoiceScore:
    """NLL of one human choice within a session transcript."""

    choice_index: int
    nll: float
    num_tokens: int

@torch.no_grad()
def score_marked_text(model, tokenizer, text, *,
                      device: str | torch.device = "cpu"):
    """Score every marked choice of one transcript."""

    return score_marked_texts(model, tokenizer, [text], device=device)[0]

@torch.no_grad()
def score_marked_texts(model, tokenizer, texts, *,
                       device: str | torch.device = "cpu",
                       max_batch_tokens=16384) -> list[list[ChoiceScore]]:
    """Score many transcripts with length-packed padded batches."""

    prepared = [_prepare_marked_text(model, tokenizer, t) 
                for t in texts]
    results: list[list[ChoiceScore]] = [[] for _ in texts]
    scorable = [i 
                for i in range(len(texts)) 
                if prepared[i] is not None]
    # longest first, so a batch's padded width is set by its first member
    scorable.sort(key=
                  lambda i: -len(prepared[i][0]))

    batches = []
    current: list[int] = []
    for index in scorable:
        if current:
            width = len(prepared[current[0]][0])
            if width * (len(current) + 1) > max_batch_tokens:
                batches.append(current)
                current = []
        current.append(index)
    if current:
        batches.append(current)

    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None) or 0
    for batch in batches:
        _score_batch(model, prepared, batch, results, pad_id, device)
    return results

def score_session_rows(model, tokenizer, rows, *, device="cpu",
                       max_batch_tokens=16384,
                       text_column="text") -> list[dict[str, Any]]:
    """Score session records, returning flat per-choice result rows."""

    texts = [row[text_column] for row in rows]
    scored = score_marked_texts(model, tokenizer, texts, device=device,
                                max_batch_tokens=max_batch_tokens)
    results = []
    for row, scores in zip(rows, scored):
        for score in scores:
            record = {k: v 
                      for k, v in row.items() 
                      if k != text_column}
            record.update(choice_index=score.choice_index, nll=score.nll,
                          num_tokens=score.num_tokens)
            results.append(record)
    return results

def map_spans_to_token_indices(offsets: list[tuple[int, int]],
                               spans: list[tuple[int, int]]) -> list[list[int]]:
    """Return, for each character span, indices of tokens overlapping it."""

    token_indices: list[list[int]] = [[] for _ in spans]
    span_cursor = 0
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_start == token_end:
            continue
        while span_cursor < len(spans) and spans[span_cursor][1] <= token_start:
            span_cursor += 1
        for span_index in range(span_cursor, len(spans)):
            span_start, span_end = spans[span_index]
            if span_start >= token_end:
                break
            if token_start < span_end and token_end > span_start:
                token_indices[span_index].append(token_index)
    return token_indices

def _prepare_marked_text(model, tokenizer, text):
    spans = find_target_spans(text)
    if not spans:
        return None

    encoded = tokenizer(text, add_special_tokens=False,
                        return_offsets_mapping=True)
    input_ids = list(encoded['input_ids'])
    offsets = list(encoded['offset_mapping'])

    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None:
        input_ids = [bos] + input_ids
        offsets = [(0, 0)] + offsets

    max_length = getattr(model.config, "max_position_embeddings", None)
    if max_length is not None and len(input_ids) > max_length:
        raise ValueError(f"Transcript has {len(input_ids)} tokens, exceeding "
                         f"the model context of {max_length}.")

    token_indices = map_spans_to_token_indices(offsets, spans)
    for choice_index, indices in enumerate(token_indices):
        # a BOS at index 0 has no preceding position to predict it from
        indices[:] = [i 
                      for i in indices 
                      if i > 0]
        if not indices:
            raise ValueError(f"Choice {choice_index} maps to no tokens.")
    return input_ids, token_indices

def _score_batch(model, prepared, batch, results, pad_id, device):
    width = max(len(prepared[i][0]) for i in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, index in enumerate(batch):
        seq = prepared[index][0]
        ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, :len(seq)] = 1
    ids = ids.to(device)
    mask = mask.to(device)
    hidden, logits = _forward(model, ids, mask)
    head = model.get_output_embeddings() if hidden is not None else None

    for row, index in enumerate(batch):
        seq, token_indices = prepared[index]
        needed = sorted({i - 1
                         for spans in token_indices
                         for i in spans})
        position = {p: n
                    for n, p in enumerate(needed)}
        rows = torch.tensor(needed, device=ids.device)
        # only the positions that predict a target token reach the vocab
        # projection, so the full [seq, vocab] logits never materialize
        if hidden is not None:
            selected = head(hidden[row].index_select(0, rows))
        else:
            selected = logits[row].index_select(0, rows)
        log_probs = torch.log_softmax(selected.float(), dim=-1).cpu()
        for choice_index, indices in enumerate(token_indices):
            nll = -sum(log_probs[position[i - 1], seq[i]].item()
                       for i in indices)
            results[index].append(ChoiceScore(choice_index, nll,
                                              len(indices)))
    del hidden, logits

def _forward(model, ids, mask):
    """Return hidden states (preferred) or full logits (fallback).

    Projecting only the scored positions through the LM head avoids
    materializing the full [seq, vocab] logits, which for long sessions and
    large vocabularies is the dominant memory cost. Models that do not expose
    a base module plus output embeddings fall back to dense logits.
    """

    base = getattr(model, "model", None)
    if base is not None and model.get_output_embeddings() is not None:
        hidden = base(input_ids=ids, attention_mask=mask, use_cache=False)
        return hidden.last_hidden_state, None
    return None, model(input_ids=ids, attention_mask=mask,
                       use_cache=False).logits
