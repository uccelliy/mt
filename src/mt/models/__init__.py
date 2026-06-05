"""Convenient imports for model classes."""

_MODEL_MODULES = {
    'DunningKruger': 'mt.models.cog_models.dunning_kruger_model',
    'DualSystemsModel': 'mt.models.cog_models.dual_system_model',
    'NoiseCeiling': 'mt.models.noiseceiling',
    'RationalModel': 'mt.models.cog_models.rational_model',
    'RescorlaWagnerModel': 'mt.models.cog_models.rescola_wagner_model',
    'Trainer': 'mt.models.trainers',
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
