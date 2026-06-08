"""Public baseline model API."""

from mt.models._registry import BASELINE_MODEL_REGISTRY, get_registered_object


__all__ = sorted(BASELINE_MODEL_REGISTRY)


def __getattr__(name):
    if name not in BASELINE_MODEL_REGISTRY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = get_registered_object(name, BASELINE_MODEL_REGISTRY)
    globals()[name] = value
    return value
