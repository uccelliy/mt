"""Tests for teacher-forced transcript scoring."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from mt.evaluation.transcript_scoring import (
    ChoiceScore,
    map_spans_to_token_indices,
    score_marked_text,
    score_marked_texts,
    score_session_rows,
)

VOCAB_SIZE = 256

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

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        batch, length = input_ids.shape
        return SimpleNamespace(logits=torch.zeros(batch, length, VOCAB_SIZE))

class PrevTokenModel:
    """Content-sensitive fake: each position predicts its own token id."""

    config = SimpleNamespace(max_position_embeddings=4096)

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        logits = torch.zeros(*input_ids.shape, VOCAB_SIZE)
        logits.scatter_(2, input_ids.unsqueeze(-1), 5.0)
        return SimpleNamespace(logits=logits)

class DecomposedModel:
    """Fake CausalLM exposing a base module and LM head, like Llama.

    Triggers the hidden-states scoring path; __call__ composes the same two
    parts, so its dense logits are the fallback-path reference.
    """

    config = SimpleNamespace(max_position_embeddings=4096)

    def __init__(self):
        torch.manual_seed(0)
        embed = torch.nn.Embedding(VOCAB_SIZE, 8)
        self._head = torch.nn.Linear(8, VOCAB_SIZE, bias=False)

        class Base:
            def __call__(self, input_ids, attention_mask=None, **kwargs):
                return SimpleNamespace(last_hidden_state=embed(input_ids))

        self.model = Base()

    def get_output_embeddings(self):
        return self._head

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.model(input_ids).last_hidden_state
        return SimpleNamespace(logits=self._head(hidden))

class CallOnlyModel:
    """Wraps a decomposed model but hides its parts to force the fallback."""

    def __init__(self, inner):
        self._inner = inner
        self.config = inner.config

    def get_output_embeddings(self):
        return None

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        return self._inner(input_ids, attention_mask)

def test_map_spans_to_token_indices_aligns_overlapping_tokens():
    offsets = [(0, 2), (2, 4), (4, 6), (6, 8), (0, 0)]
    spans = [(1, 3), (6, 7)]
    assert map_spans_to_token_indices(offsets, spans) == [[0, 1], [3]]

def test_map_spans_handles_disjoint_spans_and_empty_offsets():
    offsets = [(0, 0), (0, 3), (3, 5), (5, 9)]
    spans = [(3, 5)]
    assert map_spans_to_token_indices(offsets, spans) == [[2]]

def test_score_marked_text_uniform_nll_is_log_vocab_per_token():
    text = "ab <<C>> de <<FG>>"
    scores = score_marked_text(UniformModel(), CharTokenizer(), text)
    assert [s.choice_index for s in scores] == [0, 1]
    assert [s.num_tokens for s in scores] == [1, 2]
    assert math.isclose(scores[0].nll, math.log(VOCAB_SIZE), rel_tol=1e-5)
    assert math.isclose(scores[1].nll, 2 * math.log(VOCAB_SIZE),
                        rel_tol=1e-5)

def test_score_marked_text_no_markers_returns_empty():
    result = score_marked_text(UniformModel(), CharTokenizer(), "no markers")
    assert result == []

def test_score_marked_text_rejects_overlong_transcript():
    model = UniformModel()
    model.config = SimpleNamespace(max_position_embeddings=4)
    try:
        score_marked_text(model, CharTokenizer(), "abc <<d>>")
    except ValueError as error:
        assert "exceeding the model context" in str(error)
    else:
        raise AssertionError("expected ValueError for overlong transcript")

def test_score_session_rows_flattens_metadata():
    rows = [
        {'text': "x <<a>>", 'experiment': "e1", 'participant': "p1"},
        {'text': "y <<b>> z <<c>>", 'experiment': "e1", 'participant': "p2"},
    ]
    results = score_session_rows(UniformModel(), CharTokenizer(), rows)
    assert len(results) == 3
    assert results[0]['participant'] == "p1"
    assert results[0]['choice_index'] == 0
    assert results[2]['participant'] == "p2"
    assert results[2]['choice_index'] == 1
    assert all('text' not in r for r in results)
    assert all(isinstance(r['nll'], float) for r in results)

def test_batched_scoring_matches_single_scoring():
    texts = ["ab <<C>> de <<FG>>", "<<z>>", "longer text here <<q>> tail",
             "no markers at all", "x <<m>> y <<n>> z <<o>>"]
    model = PrevTokenModel()
    tokenizer = CharTokenizer()
    singles = [score_marked_text(model, tokenizer, t) for t in texts]
    for max_batch_tokens in (8, 64, 4096):
        batched = score_marked_texts(model, tokenizer, texts,
                                     max_batch_tokens=max_batch_tokens)
        for one, many in zip(singles, batched):
            assert len(one) == len(many)
            for a, b in zip(one, many):
                assert a.choice_index == b.choice_index
                assert a.num_tokens == b.num_tokens
                assert math.isclose(a.nll, b.nll, rel_tol=1e-6)

def test_hidden_state_path_matches_dense_logits_path():
    texts = ["ab <<C>> de <<FG>>", "<<z>>", "hi <<q>> there <<r>> end",
             "x <<m>> y <<n>> z <<o>>"]
    decomposed = DecomposedModel()
    fallback = CallOnlyModel(decomposed)
    tokenizer = CharTokenizer()
    optimized = score_marked_texts(decomposed, tokenizer, texts)
    dense = score_marked_texts(fallback, tokenizer, texts)
    for opt_scores, dense_scores in zip(optimized, dense):
        assert len(opt_scores) == len(dense_scores)
        for a, b in zip(opt_scores, dense_scores):
            assert a.choice_index == b.choice_index
            assert math.isclose(a.nll, b.nll, rel_tol=1e-6, abs_tol=1e-6)

def test_bos_token_shifts_indices():
    class BosCharTokenizer(CharTokenizer):
        bos_token_id = 1

    scores = score_marked_text(UniformModel(), BosCharTokenizer(), "<<a>>")
    expected = ChoiceScore(choice_index=0, nll=scores[0].nll, num_tokens=1)
    assert scores == [expected]
    assert math.isclose(scores[0].nll, math.log(VOCAB_SIZE), rel_tol=1e-5)
