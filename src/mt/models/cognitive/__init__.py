"""Public cognitive model API."""

from mt.models._registry import (
    BASE_MODEL_REGISTRY,
    COGNITIVE_MODEL_REGISTRY,
    MODEL_CONTRACT_REGISTRY,
    get_registered_object,
)


_PUBLIC_OBJECTS = {
    **BASE_MODEL_REGISTRY,
    **COGNITIVE_MODEL_REGISTRY,
    **MODEL_CONTRACT_REGISTRY,
}


__all__ = sorted(_PUBLIC_OBJECTS)


def __getattr__(name):
    if name not in _PUBLIC_OBJECTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = get_registered_object(name, _PUBLIC_OBJECTS)
    globals()[name] = value
    return value
