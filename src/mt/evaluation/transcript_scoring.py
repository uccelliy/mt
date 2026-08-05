"""Teacher-forced per-choice scoring for Psych-101-style transcripts.

Scoring records what the model output, never a metric. Three things per
marked choice (design doc §2.1):

  ① the human choice and the model's NLL for it
  ② the model's own top-k tokens at every token position of that choice
  ③ the log-probability of every legal option the task allows

Metrics are derived offline from these, so adding one never costs a GPU
hour. ③ needs every option's tokens scored against the same prefix; an
option that is more than one token would need a continuation pass beyond
the transcript's own tokens, which this module deliberately does not do --
such choices are recorded as unscored with a reason, never partially
scored (a partial option set makes the legal mass silently wrong).
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math

import torch

from mt.models.llm.supervision import LEFT_TARGET_MARKER, find_target_spans

class ContextLengthError(ValueError):
    """A transcript does not fit in the model's context window."""

# Why a choice carries no option scores. §8.5 asks for a per-task coverage
# report, so the reason is recorded rather than left as a silent gap.
OPTIONS_SCORED = "scored"
OPTIONS_OFF = "off"
OPTIONS_TOO_MANY = "too_many_options"
OPTIONS_MULTI_TOKEN = "multi_token_option"
OPTIONS_TOKENIZATION_MISMATCH = "tokenization_mismatch"

@dataclass(frozen=True)
class TokenCandidate:
    """One (token position, rank) entry of the model's own top-k list."""

    token_index: int
    rank: int
    token: str
    logprob: float

@dataclass(frozen=True)
class OptionScore:
    """One legal option and the log-probability the model gives it."""

    option: str
    logprob: float
    n_tokens: int
    is_human: bool

@dataclass(frozen=True)
class ChoiceScore:
    """What the model output at one marked choice."""

    choice_index: int
    nll: float
    num_tokens: int
    human_choice: str = ""
    k_options: int = 0
    top1_prob: float = float("nan")
    pred_entropy: float = float("nan")
    topk: tuple[TokenCandidate, ...] = ()
    options: tuple[OptionScore, ...] = ()
    options_status: str = OPTIONS_OFF

@dataclass
class _Prepared:
    """A tokenized transcript plus everything ③ needs for that session."""

    input_ids: list[int]
    token_indices: list[list[int]]
    choices: list[str]
    k_options: int
    option_ids: dict[str, list[int]] | None
    options_status: str

def legal_mass(options) -> float | None:
    """Total probability the model puts on the legal option set."""

    if not options:
        return None
    return sum(math.exp(option.logprob) for option in options)

def predicted_choice(options) -> str | None:
    """The option the model ranks first, or None when ③ was not scored."""

    if not options:
        return None
    return max(options, key=lambda option: option.logprob).option

@torch.no_grad()
def score_marked_text(model, tokenizer, text, *,
                      device: str | torch.device = "cpu", **kwargs):
    """Score every marked choice of one transcript."""

    return score_marked_texts(model, tokenizer, [text], device=device,
                              **kwargs)[0]

@torch.no_grad()
def score_marked_texts(model, tokenizer, texts, *,
                       device: str | torch.device = "cpu",
                       max_batch_tokens=16384,
                       top_k=0,
                       max_options=0,
                       block_rows=128) -> list[list[ChoiceScore]]:
    """Score many transcripts with length-packed padded batches.

    ② and ③ are off by default: the window-scoring grid (E3) reuses this
    function and neither needs nor should pay for them. The Track P runner
    turns both on explicitly.
    """

    prepared = [_prepare_marked_text(model, tokenizer, t,
                                     max_options=max_options)
                for t in texts]
    results: list[list[ChoiceScore]] = [[] for _ in texts]
    scorable = [i
                for i in range(len(texts))
                if prepared[i] is not None]
    # longest first, so a batch's padded width is set by its first member
    scorable.sort(key=
                  lambda i: -len(prepared[i].input_ids)) # pyright: ignore[reportOptionalMemberAccess]

    batches = []
    current: list[int] = []
    for index in scorable:
        if current:
            width = len(prepared[current[0]].input_ids) # pyright: ignore[reportOptionalMemberAccess]
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
        _score_batch(model, tokenizer, prepared, batch, results, pad_id,
                     device, top_k, block_rows)
    return results

def score_session_rows(model, tokenizer, rows, *, device="cpu",
                       max_batch_tokens=16384, top_k=0, max_options=0,
                       text_column="text") -> list[tuple[dict, list]]:
    """Score session records, pairing each with its session metadata.

    Shaping the three output tables is the runner's job, so this returns
    the metadata (everything but the transcript) beside the scores rather
    than a flattened row.
    """

    texts = [row[text_column] for row in rows]
    scored = score_marked_texts(model, tokenizer, texts, device=device,
                                max_batch_tokens=max_batch_tokens,
                                top_k=top_k, max_options=max_options)
    return [({k: v
              for k, v in row.items()
              if k != text_column}, scores)
            for row, scores in zip(rows, scored)]

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

def _option_token_ids(tokenizer, options, max_options):
    """Tokenize the legal option set as it appears after the left marker.

    Options are scored against the prefix that ends with `<<`, so their
    tokens are taken in that context and the marker's own tokens dropped.
    Returns (ids_by_option, status); ids_by_option is None unless every
    option is a single token, because ③ is all-or-nothing per choice.
    """

    if not max_options:
        return None, OPTIONS_OFF
    if len(options) > max_options:
        return None, OPTIONS_TOO_MANY

    marker = tokenizer(LEFT_TARGET_MARKER, add_special_tokens=False)
    marker_ids = list(marker['input_ids'])
    ids_by_option = {}
    for option in options:
        encoded = tokenizer(LEFT_TARGET_MARKER + option,
                            add_special_tokens=False)
        ids = list(encoded['input_ids'])
        if ids[:len(marker_ids)] != marker_ids:
            return None, OPTIONS_TOKENIZATION_MISMATCH
        ids_by_option[option] = ids[len(marker_ids):]
    if any(len(ids) != 1 for ids in ids_by_option.values()):
        return None, OPTIONS_MULTI_TOKEN
    return ids_by_option, OPTIONS_SCORED

def _prepare_marked_text(model, tokenizer, text, *, max_options=0):
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
        raise ContextLengthError(f"Transcript has {len(input_ids)} tokens, "
                                 f"exceeding the model context of "
                                 f"{max_length}.")

    token_indices = map_spans_to_token_indices(offsets, spans)
    for choice_index, indices in enumerate(token_indices):
        # a BOS at index 0 has no preceding position to predict it from
        indices[:] = [i
                      for i in indices
                      if i > 0]
        if not indices:
            raise ValueError(f"Choice {choice_index} maps to no tokens.")

    choices = [text[start:end]
               for start, end in spans]
    options = sorted(set(choices))
    option_ids, status = _option_token_ids(tokenizer, options, max_options)
    return _Prepared(input_ids=input_ids, token_indices=token_indices,
                     choices=choices, k_options=len(options),
                     option_ids=option_ids, options_status=status)

def _score_batch(model, tokenizer, prepared, batch, results, pad_id, device,
                 top_k, block_rows):
    width = max(len(prepared[i].input_ids) for i in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, index in enumerate(batch):
        seq = prepared[index].input_ids
        ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, :len(seq)] = 1
    ids = ids.to(device)
    mask = mask.to(device)
    hidden, logits = _forward(model, ids, mask)
    head = model.get_output_embeddings() if hidden is not None else None

    for row, index in enumerate(batch):
        prep = prepared[index]
        results[index].extend(
            _score_session(tokenizer, prep, hidden, logits, head, row,
                           ids.device, top_k, block_rows))
    del hidden, logits

def _score_session(tokenizer, prep, hidden, logits, head, row, device,
                   top_k, block_rows):
    """Turn one row's hidden states into that session's ChoiceScores."""

    seq, token_indices = prep.input_ids, prep.token_indices
    needed = sorted({i - 1
                     for spans in token_indices
                     for i in spans})
    # every scored position predicts one known token; the first position of
    # a choice additionally supplies ③, whose options are single-token
    target = {i - 1: seq[i]
              for spans in token_indices
              for i in spans}
    option_ids = prep.option_ids
    option_at = {}
    if option_ids is not None:
        first_positions = {indices[0] - 1
                           for indices in token_indices}
        option_at = {p: option_ids
                     for p in first_positions}

    stats = {}
    for start in range(0, len(needed), block_rows):
        block = needed[start:start + block_rows]
        rows = torch.tensor(block, device=device)
        # only the positions that predict a target token reach the vocab
        # projection, so the full [seq, vocab] logits never materialize
        if hidden is not None:
            selected = head(hidden[row].index_select(0, rows)) # pyright: ignore[reportOptionalCall]
        else:
            selected = logits[row].index_select(0, rows) # pyright: ignore[reportOptionalSubscript]
        log_probs = torch.log_softmax(selected.float(), dim=-1)
        _collect_block(tokenizer, log_probs, block, target, option_at,
                       top_k, stats)
        del selected, log_probs

    scores = []
    for choice_index, indices in enumerate(token_indices):
        first = indices[0] - 1
        candidates = tuple(TokenCandidate(token_index=n, rank=rank,
                                          token=token, logprob=value)
                           for n, i in enumerate(indices)
                           for rank, (token, value)
                           in enumerate(stats[i - 1]['topk']))
        options, status = _session_options(prep, seq, choice_index, indices,
                                           stats[first])
        scores.append(ChoiceScore(
            choice_index=choice_index,
            nll=-sum(stats[i - 1]['target'] for i in indices),
            num_tokens=len(indices),
            human_choice=prep.choices[choice_index],
            k_options=prep.k_options,
            top1_prob=stats[first]['top1_prob'],
            pred_entropy=stats[first]['entropy'],
            topk=candidates,
            options=options,
            options_status=status))
    return scores

def _session_options(prep, seq, choice_index, indices, first_stats):
    """Score the legal option set for one choice, or say why it was not.

    The guard is the human choice: the tokens it actually occupies in the
    transcript must equal the tokens the option set was built from. They
    agree on every Psych-101 task under the Llama tokenizer (`<<` is a
    clean boundary), but a tokenizer that merges across the marker would
    otherwise yield option log-probabilities that are not comparable to
    the human choice's own NLL.
    """

    if prep.option_ids is None:
        return (), prep.options_status
    human = prep.choices[choice_index]
    if [seq[i] for i in indices] != prep.option_ids.get(human):
        return (), OPTIONS_TOKENIZATION_MISMATCH
    values = first_stats['options']
    return tuple(OptionScore(option=option,
                             logprob=values[option],
                             n_tokens=len(prep.option_ids[option]),
                             is_human=option == human)
                 for option in sorted(values)), OPTIONS_SCORED

def _collect_block(tokenizer, log_probs, block, target, option_at, top_k,
                   stats):
    """Reduce one block of vocab rows to the small numbers we keep."""

    wanted = torch.tensor([target[p] for p in block],
                          device=log_probs.device)
    target_lp = log_probs.gather(1, wanted[:, None]).squeeze(1).cpu()
    entropy = -(log_probs.exp() * log_probs).sum(-1).cpu()
    if top_k:
        top_values, top_ids = log_probs.topk(top_k, dim=-1)
        top_values = top_values.cpu()
        top_ids = top_ids.cpu()
        tokens = [tokenizer.convert_ids_to_tokens(ids.tolist())
                  for ids in top_ids]
    else:
        top_values, tokens = None, None

    option_lp = {}
    for n, position in enumerate(block):
        ids_by_option = option_at.get(position)
        if ids_by_option is None:
            continue
        flat = torch.tensor([ids[0] for ids in ids_by_option.values()],
                            device=log_probs.device)
        values = log_probs[n].index_select(0, flat).cpu()
        option_lp[position] = dict(zip(ids_by_option, values.tolist()))

    for n, position in enumerate(block):
        entry = {'target': target_lp[n].item(),
                 'entropy': entropy[n].item(),
                 'options': option_lp.get(position)}
        if top_values is not None:
            entry['topk'] = list(zip(tokens[n], top_values[n].tolist()))
            entry['top1_prob'] = math.exp(top_values[n][0].item())
        else:
            entry['topk'] = []
            entry['top1_prob'] = float("nan")
        stats[position] = entry

def _forward(model, ids, mask):
    """Return hidden states (preferred) or full logits (fallback).

    Projecting only the scored positions through the LM head avoids
    materializing the full [seq, vocab] logits, which for long sessions and
    large vocabularies is the dominant memory cost. Models that do not expose
    a base module plus output embeddings fall back to dense logits.
    """

    base = _unwrap_base(model)
    with _cuda_sdpa_context(ids.device):
        if base is not None and model.get_output_embeddings() is not None:
            hidden = base(input_ids=ids, attention_mask=mask,
                          use_cache=False)
            return hidden.last_hidden_state, None
        return None, model(input_ids=ids, attention_mask=mask,
                           use_cache=False).logits

def _unwrap_base(model):
    """Descend to the bare trunk beneath adapter/causal-LM wrappers.

    A PeftModel nests a full causal LM (which still owns an LM head) under
    .model; scoring must call the trunk below it, or the hidden-state path
    would return logits-shaped output instead of hidden states.
    """

    base = getattr(model, "model", None)
    while base is not None:
        get_head = getattr(base, "get_output_embeddings", None)
        if get_head is None or get_head() is None:
            return base
        base = getattr(base, "model", None)
    return None

def _cuda_sdpa_context(device):
    """Prefer fused CUDA attention before the quadratic math fallback."""

    if device.type != "cuda":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
    return sdpa_kernel(backends, set_priority=True)
