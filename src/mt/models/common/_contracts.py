"""Column and tensor-key contracts for model implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# This is the single inventory of data consumed by registered models and the
# shared choice-NLL trainer. Filled tuples are known raw dataframe columns.
# Empty tuples mark inputs whose raw-column mapping still needs a decision.
MODEL_TENSOR_COLUMNS: dict[str, dict[str, tuple[str, ...]]] = {
    "DualSystemsModel": {
        "choice": ("choice",),
        "current_state": ("current_state",),
        "reward": ("reward",),
    },
    "DunningKruger": {
        "choice": ("choice",),
    },
    "GeneralizedContextModel": {
        "choice": ("choice",),
        "ground_truth": ("ground_truth",),
        "features": (),
    },
    "NoiseCeiling": {
        "choice": ("choice",),
        "uid": (),
    },
    "RationalModel": {
        "choice": ("choice",),
        "ground_truth": ("ground_truth",),
    },
    "RescorlaWagnerModel": {
        "choice": ("choice",),
        "reward": ("reward",),
        "forced": ("forced",),
        "choice_for_updating": (),
    },
}


MODEL_OPTIONAL_TENSOR_KEYS: dict[str, tuple[str, ...]] = {
    "RescorlaWagnerModel": ("forced",),
}


MODEL_DERIVED_TENSOR_KEYS: dict[str, tuple[str, ...]] = {
    "RescorlaWagnerModel": ("choice_for_updating",),
    "GeneralizedContextModel": ("features",),
}


MODEL_COLUMN_PATTERNS: dict[str, dict[str, str]] = {
    "GeneralizedContextModel": {
        "features": r"^x\d+$",
    },
}


@dataclass(frozen=True)
class ModelDataSpec:
    """Model-owned view of required tensor keys and dataframe columns."""

    model_name: str
    tensor_columns: dict[str, tuple[str, ...]]
    optional_tensor_keys: tuple[str, ...] = ()
    derived_tensor_keys: tuple[str, ...] = ()
    column_patterns: dict[str, str] = field(default_factory=dict)

    @property
    def required_columns(self) -> tuple[str, ...]:
        columns: list[str] = []
        for key, mapped_columns in self.tensor_columns.items():
            if key in self.optional_tensor_keys or key in self.derived_tensor_keys:
                continue
            columns.extend(mapped_columns)
        return tuple(dict.fromkeys(columns))

    @property
    def unmapped_tensor_keys(self) -> tuple[str, ...]:
        keys = [
            key
            for key, columns in self.tensor_columns.items()
            if not columns and key not in self.derived_tensor_keys
        ]
        return tuple(keys)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "tensor_columns": {key: list(columns) for key, columns in self.tensor_columns.items()},
            "optional_tensor_keys": list(self.optional_tensor_keys),
            "derived_tensor_keys": list(self.derived_tensor_keys),
            "column_patterns": dict(self.column_patterns),
            "required_columns": list(self.required_columns),
            "unmapped_tensor_keys": list(self.unmapped_tensor_keys),
        }


def model_data_spec(model_or_name: object) -> ModelDataSpec:
    """Return the editable model data specification for a model class/name."""

    model_name = _model_name(model_or_name)
    return ModelDataSpec(
        model_name=model_name,
        tensor_columns=dict(MODEL_TENSOR_COLUMNS.get(model_name, {})),
        optional_tensor_keys=MODEL_OPTIONAL_TENSOR_KEYS.get(model_name, ()),
        derived_tensor_keys=MODEL_DERIVED_TENSOR_KEYS.get(model_name, ()),
        column_patterns=dict(MODEL_COLUMN_PATTERNS.get(model_name, {})),
    )


def required_columns_for_model(model_or_name: object) -> tuple[str, ...]:
    """Return known dataframe columns required by a model."""

    return model_data_spec(model_or_name).required_columns


def unmapped_tensor_keys_for_model(model_or_name: object) -> tuple[str, ...]:
    """Return tensor keys whose raw columns still need to be filled in."""

    return model_data_spec(model_or_name).unmapped_tensor_keys


def _model_name(model_or_name: object) -> str:
    if isinstance(model_or_name, str):
        return model_or_name
    if isinstance(model_or_name, type):
        return model_or_name.__name__
    return model_or_name.__class__.__name__
