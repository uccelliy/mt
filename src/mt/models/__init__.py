"""Convenient imports for model classes."""

_MODEL_MODULES = {
    'DunningKruger': 'mt.models.cognitive.dunning_kruger',
    'DualSystemsModel': 'mt.models.cognitive.dual_systems',
    'NoiseCeiling': 'mt.models.baselines.noise_ceiling',
    'RationalModel': 'mt.models.cognitive.rational',
    'RescorlaWagnerModel': 'mt.models.cognitive.rescorla_wagner',
    'Trainer': 'mt.training.trainer',
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
