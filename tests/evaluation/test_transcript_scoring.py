"""Tests for teacher-forced transcript scoring."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from mt.evaluation.transcript_scoring import (
    _cuda_sdpa_context,
    ContextLengthError,
    legal_mass,
    map_spans_to_token_indices,
    predicted_choice,
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

    def convert_ids_to_tokens(self, ids):
        return [chr(i) for i in ids]

class MarkerMergingTokenizer(CharTokenizer):
    """Tokenizes an option differently in isolation than in the transcript.

    Stands in for a real tokenizer that merges across `<<`, which would
    make option log-probabilities incomparable to the human choice's NLL.
    """

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=True):
        encoded = super().__call__(text, add_special_tokens,
                                   return_offsets_mapping)
        if text.startswith("<<"):
            encoded['input_ids'] = (encoded['input_ids'][:2]
                                    + [(i + 1) % VOCAB_SIZE
                                       for i in encoded['input_ids'][2:]])
        return encoded

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

    config = SimpleNamespace(max_position_embeddings=4096,
                             model_type="llama")

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

class AdapterStyleModel:
    """PeftModel-like wrapper: .model is a full causal LM, not a trunk."""

    def __init__(self, inner):
        self.model = inner
        self.config = inner.config

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        raise AssertionError("the dense fallback must not run here")

class SoftcappedDecomposedModel(DecomposedModel):
    """Gemma-like fake whose wrapper soft-caps LM-head logits."""

    def __init__(self, nested_config):
        super().__init__()
        with torch.no_grad():
            self._head.weight.mul_(50)
        text_config = SimpleNamespace(model_type="gemma4_text",
                                      max_position_embeddings=4096,
                                      final_logit_softcapping=2.0)
        if nested_config:
            self.config = SimpleNamespace(
                get_text_config=lambda: text_config)
        else:
            self.config = text_config

    def __call__(self, input_ids, attention_mask=None, **kwargs):
        hidden = self.model(input_ids, attention_mask=attention_mask,
                            **kwargs).last_hidden_state
        logits = self._head(hidden)
        config = (self.config.get_text_config()
                  if hasattr(self.config, "get_text_config")
                  else self.config)
        cap = config.final_logit_softcapping
        return SimpleNamespace(logits=torch.tanh(logits / cap) * cap)

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
    except ContextLengthError as error:
        assert "exceeding the model context" in str(error)
    else:
        raise AssertionError("expected ContextLengthError for overlong "
                             "transcript")

def test_context_limit_uses_nested_text_config():
    text_config = SimpleNamespace(max_position_embeddings=4)
    model = UniformModel()
    model.config = SimpleNamespace(get_text_config=lambda: text_config)
    try:
        score_marked_text(model, CharTokenizer(), "abc <<d>>")
    except ContextLengthError as error:
        assert "exceeding the model context" in str(error)
    else:
        raise AssertionError("expected nested text context limit to apply")

def test_score_session_rows_pairs_metadata_with_scores():
    rows = [
        {'text': "x <<a>>", 'experiment': "e1", 'participant': "p1"},
        {'text': "y <<b>> z <<c>>", 'experiment': "e1", 'participant': "p2"},
    ]
    results = score_session_rows(UniformModel(), CharTokenizer(), rows)
    assert [meta['participant'] for meta, _ in results] == ["p1", "p2"]
    assert [len(scores) for _, scores in results] == [1, 2]
    assert [s.choice_index for s in results[1][1]] == [0, 1]
    assert all('text' not in meta for meta, _ in results)
    assert all(isinstance(s.nll, float)
               for _, scores in results for s in scores)

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

def test_unknown_architecture_uses_its_standard_forward():
    class UnknownModel(DecomposedModel):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(max_position_embeddings=4096,
                                          model_type="unknown")
            self.forward_calls = 0

        def __call__(self, input_ids, attention_mask=None, **kwargs):
            self.forward_calls += 1
            return super().__call__(input_ids, attention_mask, **kwargs)

    model = UnknownModel()
    score_marked_text(model, CharTokenizer(), "ab <<C>>")
    assert model.forward_calls == 1

def test_softcapped_hidden_path_matches_official_wrapper_path():
    tokenizer = CharTokenizer()
    for nested_config in (False, True):
        decomposed = SoftcappedDecomposedModel(nested_config)
        fallback = CallOnlyModel(decomposed)
        optimized = score_marked_text(
            decomposed, tokenizer, "w <<a>> x <<b>>", top_k=3,
            max_options=8)
        dense = score_marked_text(
            fallback, tokenizer, "w <<a>> x <<b>>", top_k=3,
            max_options=8)
        for actual, expected in zip(optimized, dense):
            assert math.isclose(actual.nll, expected.nll,
                                rel_tol=1e-6, abs_tol=1e-6)
            assert math.isclose(actual.top1_prob, expected.top1_prob,
                                rel_tol=1e-6, abs_tol=1e-6)
            assert math.isclose(actual.pred_entropy, expected.pred_entropy,
                                rel_tol=1e-6, abs_tol=1e-6)
            assert [(item.token_index, item.rank, item.token)
                    for item in actual.topk] == [
                        (item.token_index, item.rank, item.token)
                        for item in expected.topk]
            for left, right in zip(actual.topk, expected.topk):
                assert math.isclose(left.logprob, right.logprob,
                                    rel_tol=1e-6, abs_tol=1e-6)
            assert [item.option for item in actual.options] == [
                item.option for item in expected.options]
            for left, right in zip(actual.options, expected.options):
                assert math.isclose(left.logprob, right.logprob,
                                    rel_tol=1e-6, abs_tol=1e-6)

def test_tiny_transformers_gemma_matches_its_standard_forward():
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4TextConfig,
    )
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    config = Gemma4TextConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=128,
        sliding_window=32,
        layer_types=["full_attention"],
        final_logit_softcapping=0.1,
        vocab_size_per_layer_input=VOCAB_SIZE,
        hidden_size_per_layer_input=4,
        num_global_key_value_heads=1,
        global_head_dim=8,
        tie_word_embeddings=False,
    )
    torch.manual_seed(1)
    model = Gemma4ForCausalLM(config).eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(50)
    optimized = score_marked_text(
        model, CharTokenizer(), "w <<a>> x <<b>>", top_k=3,
        max_options=8)
    standard = score_marked_text(
        CallOnlyModel(model), CharTokenizer(), "w <<a>> x <<b>>", top_k=3,
        max_options=8)
    for actual, expected in zip(optimized, standard):
        assert math.isclose(actual.nll, expected.nll,
                            rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(actual.top1_prob, expected.top1_prob,
                            rel_tol=1e-6, abs_tol=1e-6)
        assert math.isclose(actual.pred_entropy, expected.pred_entropy,
                            rel_tol=1e-6, abs_tol=1e-6)
        assert actual.topk == expected.topk
        assert actual.options == expected.options

def test_adapter_wrapper_unwraps_to_trunk_and_matches_inner_model():
    texts = ["ab <<C>> de <<FG>>", "hi <<q>> there <<r>> end"]
    decomposed = DecomposedModel()
    wrapped = AdapterStyleModel(decomposed)
    tokenizer = CharTokenizer()
    direct = score_marked_texts(decomposed, tokenizer, texts)
    via_wrapper = score_marked_texts(wrapped, tokenizer, texts)
    for direct_scores, wrapped_scores in zip(direct, via_wrapper):
        assert len(direct_scores) == len(wrapped_scores)
        for a, b in zip(direct_scores, wrapped_scores):
            assert a.choice_index == b.choice_index
            assert math.isclose(a.nll, b.nll, rel_tol=1e-6, abs_tol=1e-6)

def test_bos_token_shifts_indices():
    class BosCharTokenizer(CharTokenizer):
        bos_token_id = 1

    scores = score_marked_text(UniformModel(), BosCharTokenizer(), "<<a>>")
    assert len(scores) == 1
    assert scores[0].choice_index == 0
    assert scores[0].num_tokens == 1
    assert scores[0].human_choice == "a"
    assert scores[0].k_options == 1
    assert math.isnan(scores[0].top1_prob)
    assert scores[0].topk == ()
    assert scores[0].options == ()
    assert math.isclose(scores[0].nll, math.log(VOCAB_SIZE), rel_tol=1e-5)
    # a uniform model's next-token entropy is log|V| whatever the target is
    assert math.isclose(scores[0].pred_entropy, math.log(VOCAB_SIZE),
                        rel_tol=1e-5)

def test_sdpa_backend_context_is_noop_off_cuda():
    with _cuda_sdpa_context(torch.device("cpu")):
        value = torch.tensor(1)
    assert value.item() == 1

def test_topk_records_every_token_position():
    scores = score_marked_text(PrevTokenModel(), CharTokenizer(),
                               "ab <<CD>>", top_k=3)
    candidates = scores[0].topk
    assert scores[0].num_tokens == 2
    assert [(c.token_index, c.rank) for c in candidates] == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    # PrevTokenModel puts its mass on the position's own input token, so
    # rank 0 at token position i is the token the prefix ends with
    assert [c.token for c in candidates if c.rank == 0] == ["<", "C"]
    for position in (0, 1):
        ranked = [c.logprob for c in candidates if c.token_index == position]
        assert ranked == sorted(ranked, reverse=True)

def test_top1_prob_is_a_probability_not_a_logprob():
    scores = score_marked_text(UniformModel(), CharTokenizer(), "ab <<C>>",
                               top_k=5)
    assert math.isclose(scores[0].top1_prob, 1 / VOCAB_SIZE, rel_tol=1e-5)

def test_single_token_options_are_scored_exactly():
    # the human choice's option log-probability must equal its own -NLL,
    # which is what makes ③ comparable with ① at all
    text = "w <<a>> x <<b>> y <<a>>"
    scores = score_marked_text(PrevTokenModel(), CharTokenizer(), text,
                               max_options=8)
    for score in scores:
        assert score.options_status == "scored"
        assert score.k_options == 2
        assert [o.option for o in score.options] == ["a", "b"]
        assert [o.n_tokens for o in score.options] == [1, 1]
        human = [o for o in score.options if o.is_human]
        assert [o.option for o in human] == [score.human_choice]
        assert math.isclose(human[0].logprob, -score.nll, rel_tol=1e-6)

def test_legal_mass_and_predicted_choice_derive_from_options():
    scores = score_marked_text(PrevTokenModel(), CharTokenizer(),
                               "w <<a>> x <<b>>", max_options=8)
    for score in scores:
        mass = legal_mass(score.options)
        assert math.isclose(mass, sum(math.exp(o.logprob)
                                      for o in score.options), rel_tol=1e-9)
        assert 0 < mass <= 1
        best = max(score.options, key=lambda o: o.logprob).option
        assert predicted_choice(score.options) == best

def test_options_skipped_when_set_exceeds_max_options():
    text = "u <<a>> v <<b>> w <<c>>"
    scores = score_marked_text(UniformModel(), CharTokenizer(), text,
                               max_options=2)
    for score in scores:
        assert score.options_status == "too_many_options"
        assert score.options == ()
        assert score.k_options == 3          # still recorded for coverage

def test_options_skipped_whole_choice_when_any_option_is_multi_token():
    # partial option sets are never written: a legal mass over a subset of
    # the options is silently wrong rather than merely incomplete
    text = "u <<a>> v <<bc>>"
    scores = score_marked_text(UniformModel(), CharTokenizer(), text,
                               max_options=8)
    for score in scores:
        assert score.options_status == "multi_token_option"
        assert score.options == ()

def test_options_dropped_when_tokenization_disagrees_with_transcript():
    scores = score_marked_text(UniformModel(), MarkerMergingTokenizer(),
                               "u <<a>> v <<b>>", max_options=8)
    for score in scores:
        assert score.options_status == "tokenization_mismatch"
        assert score.options == ()

def test_options_off_by_default_so_window_scoring_pays_nothing():
    scores = score_marked_text(UniformModel(), CharTokenizer(), "u <<a>>")
    assert scores[0].options_status == "off"
    assert scores[0].topk == ()

def test_enriched_scoring_leaves_nll_unchanged():
    text = "ab <<C>> de <<FG>>"
    model, tokenizer = PrevTokenModel(), CharTokenizer()
    plain = score_marked_text(model, tokenizer, text)
    rich = score_marked_text(model, tokenizer, text, top_k=4, max_options=8)
    for a, b in zip(plain, rich):
        assert a.num_tokens == b.num_tokens
        assert math.isclose(a.nll, b.nll, rel_tol=1e-9)

def test_block_size_does_not_change_results():
    text = "a <<b>> c <<d>> e <<f>> g <<h>> i <<j>>"
    model, tokenizer = PrevTokenModel(), CharTokenizer()
    whole = score_marked_texts(model, tokenizer, [text], top_k=3,
                               max_options=8, block_rows=64)[0]
    split = score_marked_texts(model, tokenizer, [text], top_k=3,
                               max_options=8, block_rows=2)[0]
    assert whole == split
