"""Model registry and lazy import helpers."""

from __future__ import annotations

from importlib import import_module


COGNITIVE_MODEL_REGISTRY = {
    "DualSystemsModel": "mt.models.cognitive._dual_systems",
    "GeneralizedContextModel": "mt.models.cognitive._generalized_context",
    "RescorlaWagnerModel": "mt.models.cognitive._rescorla_wagner",
}


BASE_MODEL_REGISTRY = {
    "BaseCognitiveModel": "mt.models.common._base",
    "load_saved_model": "mt.models.common._base",
}


BASELINE_MODEL_REGISTRY = {
    "DunningKruger": "mt.models.baselines._lookup_table_dunning",
    "NoiseCeiling": "mt.models.baselines._noise_ceiling",
    "RationalModel": "mt.models.baselines._rational",
}


MODEL_CONTRACT_REGISTRY = {
    "MODEL_COLUMN_PATTERNS": "mt.models.common._contracts",
    "MODEL_DERIVED_TENSOR_KEYS": "mt.models.common._contracts",
    "MODEL_OPTIONAL_TENSOR_KEYS": "mt.models.common._contracts",
    "MODEL_TENSOR_COLUMNS": "mt.models.common._contracts",
    "ModelDataSpec": "mt.models.common._contracts",
    "data_contract_for_model": "mt.models.common._contracts",
    "model_data_spec": "mt.models.common._contracts",
    "required_columns_for_model": "mt.models.common._contracts",
    "unmapped_tensor_keys_for_model": "mt.models.common._contracts",
}


MODEL_REGISTRY = {
    **BASE_MODEL_REGISTRY,
    **COGNITIVE_MODEL_REGISTRY,
    **BASELINE_MODEL_REGISTRY,
    **MODEL_CONTRACT_REGISTRY,
    "Trainer": "mt.training.trainer",
}


def get_registered_object(name: str, registry: dict[str, str] | None = None):
    """Load a registered object by public name."""

    registry = registry or MODEL_REGISTRY
    if name not in registry:
        raise KeyError(f"Unknown registered model object: {name}")

    module = import_module(registry[name])
    return getattr(module, name)


def get_model_class(name: str):
    """Load a registered model class by name."""

    model_registry = {
        **COGNITIVE_MODEL_REGISTRY,
        **BASELINE_MODEL_REGISTRY,
    }
    return get_registered_object(name, model_registry)


def list_model_names() -> tuple[str, ...]:
    """Return public names of registered model classes."""

    return tuple(sorted([*COGNITIVE_MODEL_REGISTRY, *BASELINE_MODEL_REGISTRY]))
