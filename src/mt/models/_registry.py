"""Model registry and lazy import helpers."""

from __future__ import annotations

from importlib import import_module


COGNITIVE_MODEL_REGISTRY = {
    "DecisionUpdatedReferencePointModel": "mt.models.cognitive._reference_point",
    "DunningKruger": "mt.models.cognitive._dunning_kruger",
    "DualSystemsModel": "mt.models.cognitive._dual_systems",
    "GPUCBModel": "mt.models.cognitive._gp_ucb",
    "GeneralizedContextModel": "mt.models.cognitive._generalized_context",
    "HyperbolicDiscountingModel": "mt.models.cognitive._hyperbolic_discounting",
    "LookupTableModel": "mt.models.cognitive._lookup_table",
    "MultiTaskReinforcementLearningModel": "mt.models.cognitive._multitask_reinforcement_learning",
    "OddOneOutModel": "mt.models.cognitive._odd_one_out",
    "OnlineLinearRegressionModel": "mt.models.cognitive._linear_regression",
    "ProspectTheoryModel": "mt.models.cognitive._prospect_theory",
    "RationalModel": "mt.models.cognitive._rational",
    "RescorlaWagnerContextModel": "mt.models.cognitive._rescorla_wagner_context",
    "RescorlaWagnerModel": "mt.models.cognitive._rescorla_wagner",
    "WeightedAdditiveModel": "mt.models.cognitive._weighted_additive",
}


BASE_MODEL_REGISTRY = {
    "BaseCognitiveModel": "mt.models.cognitive._base",
    "FormulaOnlyCognitiveModel": "mt.models.cognitive._base",
    "load_saved_model": "mt.models.cognitive._base",
}


BASELINE_MODEL_REGISTRY = {
    "NoiseCeiling": "mt.models.baselines._noise_ceiling",
}


MODEL_CONTRACT_REGISTRY = {
    "MODEL_DERIVED_TENSOR_KEYS": "mt.models.cognitive._contracts",
    "MODEL_OPTIONAL_TENSOR_KEYS": "mt.models.cognitive._contracts",
    "MODEL_TENSOR_COLUMNS": "mt.models.cognitive._contracts",
    "ModelDataSpec": "mt.models.cognitive._contracts",
    "data_contract_for_model": "mt.models.cognitive._contracts",
    "model_data_spec": "mt.models.cognitive._contracts",
    "required_columns_for_model": "mt.models.cognitive._contracts",
    "unmapped_tensor_keys_for_model": "mt.models.cognitive._contracts",
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
