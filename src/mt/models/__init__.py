"""Public model API."""

from mt.models._registry import (
    MODEL_REGISTRY,
    get_model_class,
    get_registered_object,
    list_model_names,
)


__all__ = sorted(
    [
        *MODEL_REGISTRY,
        "MODEL_REGISTRY",
        "get_model_class",
        "get_registered_object",
        "list_model_names",
    ]
)


def __getattr__(name):
    if name not in MODEL_REGISTRY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = get_registered_object(name)
    globals()[name] = value
    return value
