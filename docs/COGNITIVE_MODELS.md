# Cognitive Model Formula Implementations

The cognitive models are currently implemented in a formula-first style. Most
new models do not yet define dataframe columns; their `forward(data)` methods
expect tensors with semantically named keys. Data contracts can later map raw
task tables onto these tensors.

## Internal Structure

All cognitive models inherit from `BaseCognitiveModel`. The base class provides
the shared `forward -> compute_logits` convention and parameter persistence via
`save_parameters`, `load_parameters`, and `from_saved`.

Pure equation code lives under `mt.models.cognitive.formulas`. Model modules are
thin wrappers that define learnable parameters, optional preprocessing, and a
`compute_logits(data)` method that calls the formula layer.

## Implemented From The Centaur Supplement

| Model | Module | Expected tensor keys |
| --- | --- | --- |
| Generalized context model | `mt.models.cognitive.generalized_context` | `query_features`, `memory_features`, `memory_labels` |
| Prospect theory model | `mt.models.cognitive.prospect_theory` | `probabilities`, `values` |
| Hyperbolic discounting model | `mt.models.cognitive.hyperbolic_discounting` | `rewards`, `delays` |
| Dual-systems model | `mt.models.cognitive.dual_systems` | currently dataframe-backed |
| Rescorla-Wagner model | `mt.models.cognitive.rescorla_wagner` | currently dataframe-backed |
| Rescorla-Wagner model with context | `mt.models.cognitive.rescorla_wagner_context` | `choice`, `reward`, `context` |
| Online linear regression model | `mt.models.cognitive.linear_regression` | `features`, `reward`, optional `option_values` |
| Weighted-additive model | `mt.models.cognitive.weighted_additive` | `option_features` |
| Decision-updated reference point model | `mt.models.cognitive.reference_point` | `win_values`, `loss_values`, `win_probabilities`, `loss_probabilities` |
| Odd-one-out model | `mt.models.cognitive.odd_one_out` | `option_embeddings` or `object_ids` |
| GP-UCB choice rule | `mt.models.cognitive.gp_ucb` | `means`, `stds` |
| Rational model | `mt.models.cognitive.rational` | currently dataframe-backed |
| Lookup table model | `mt.models.cognitive.lookup_table` | `trial` |

## Not Yet Implemented

The supplement lists a multi-task reinforcement learning model, but the model
details section does not provide its equations. `MultiTaskReinforcementLearningModel`
is therefore an explicit placeholder until the original specification is added.
