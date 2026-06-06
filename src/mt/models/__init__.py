"""Convenient imports for model classes."""

_MODEL_MODULES = {
    'DecisionUpdatedReferencePointModel': 'mt.models.cognitive.reference_point',
    'DunningKruger': 'mt.models.cognitive.dunning_kruger',
    'DualSystemsModel': 'mt.models.cognitive.dual_systems',
    'GPUCBModel': 'mt.models.cognitive.gp_ucb',
    'GeneralizedContextModel': 'mt.models.cognitive.generalized_context',
    'HyperbolicDiscountingModel': 'mt.models.cognitive.hyperbolic_discounting',
    'LookupTableModel': 'mt.models.cognitive.lookup_table',
    'MultiTaskReinforcementLearningModel': 'mt.models.cognitive.multitask_reinforcement_learning',
    'NoiseCeiling': 'mt.models.baselines.noise_ceiling',
    'OddOneOutModel': 'mt.models.cognitive.odd_one_out',
    'OnlineLinearRegressionModel': 'mt.models.cognitive.linear_regression',
    'ProspectTheoryModel': 'mt.models.cognitive.prospect_theory',
    'RationalModel': 'mt.models.cognitive.rational',
    'RescorlaWagnerContextModel': 'mt.models.cognitive.rescorla_wagner_context',
    'RescorlaWagnerModel': 'mt.models.cognitive.rescorla_wagner',
    'Trainer': 'mt.training.trainer',
    'WeightedAdditiveModel': 'mt.models.cognitive.weighted_additive',
}

__all__ = sorted(_MODEL_MODULES)


def __getattr__(name):
    if name not in _MODEL_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name = _MODEL_MODULES[name]
    module = __import__(module_name, fromlist=[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
