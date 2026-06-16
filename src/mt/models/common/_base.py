"""Shared base classes for model implementations."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn


LEGACY_MODEL_MODULES = {
    "mt.models.cognitive.base": "mt.models.common._base",
    "mt.models.cognitive._base": "mt.models.common._base",
    "mt.models.cognitive.contracts": "mt.models.common._contracts",
    "mt.models.cognitive._contracts": "mt.models.common._contracts",
    "mt.models.cognitive.rescorla_wagner": "mt.models.cognitive._rescorla_wagner",
    "mt.models.cognitive.rational": "mt.models.baselines._rational",
    "mt.models.cognitive._rational": "mt.models.baselines._rational",
    "mt.models.cognitive.dual_systems": "mt.models.cognitive._dual_systems",
    "mt.models.cognitive.dunning_kruger": "mt.models.baselines._lookup_table_dunning",
    "mt.models.cognitive._dunning_kruger": "mt.models.baselines._lookup_table_dunning",
    "mt.models.cognitive.generalized_context": "mt.models.cognitive._generalized_context",
    "mt.models.cognitive.prospect_theory": "mt.models.cognitive._prospect_theory",
    "mt.models.cognitive.hyperbolic_discounting": "mt.models.cognitive._hyperbolic_discounting",
    "mt.models.cognitive.linear_regression": "mt.models.cognitive._linear_regression",
    "mt.models.cognitive.reference_point": "mt.models.cognitive._reference_point",
    "mt.models.cognitive.weighted_additive": "mt.models.cognitive._weighted_additive",
    "mt.models.cognitive.odd_one_out": "mt.models.cognitive._odd_one_out",
    "mt.models.cognitive.gp_ucb": "mt.models.cognitive._gp_ucb",
    "mt.models.cognitive.lookup_table": "mt.models.baselines._lookup_table",
    "mt.models.cognitive._lookup_table": "mt.models.baselines._lookup_table",
    "mt.models.cognitive.multitask_reinforcement_learning": (
        "mt.models.cognitive._multitask_reinforcement_learning"
    ),
    "mt.models.baselines.noise_ceiling": "mt.models.baselines._noise_ceiling",
}


class BaseCognitiveModel(nn.Module):
    """Base class for cognitive models.

    Subclasses own learnable parameters and formula calls. The data layer owns
    dataframe contracts, splits, and transforms.
    """

    required_columns: ClassVar[list[str]] = []
    config_keys: ClassVar[tuple[str, ...]] = ()

    def preprocess_data(self, train_df, eval_df):
        """Convert train/eval dataframes into tensors for this model."""

        raise NotImplementedError(f"{self.__class__.__name__} does not define a data contract.")

    def compute_logits(self, data):
        """Compute model logits from already-prepared tensor data."""

        raise NotImplementedError(f"{self.__class__.__name__} must implement compute_logits().")

    def forward(self, data):
        return self.compute_logits(data)

    def get_config(self) -> dict[str, Any]:
        """Return constructor arguments needed to recreate this model."""

        return {key: getattr(self, key) for key in self.config_keys}

    @classmethod
    def data_spec(cls):
        """Return the model's tensor-key to dataframe-column specification."""

        from mt.models.common._contracts import model_data_spec

        return model_data_spec(cls)

    @classmethod
    def data_contract(cls, *, allow_incomplete: bool = False):
        """Return a data contract built from the model's known column mapping."""

        from mt.models.common._contracts import data_contract_for_model

        return data_contract_for_model(cls, allow_incomplete=allow_incomplete)

    @classmethod
    def known_required_columns(cls) -> tuple[str, ...]:
        """Return known dataframe columns required by this model."""

        return cls.data_spec().required_columns

    @classmethod
    def unmapped_tensor_keys(cls) -> tuple[str, ...]:
        """Return tensor keys whose raw dataframe columns are not decided yet."""

        return cls.data_spec().unmapped_tensor_keys

    def parameter_payload(
        self,
        *,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        data_contract: dict[str, Any] | None = None,
        split_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a serializable parameter payload."""

        return {
            "schema_version": 1,
            "model_class": self.__class__.__name__,
            "model_module": self.__class__.__module__,
            "config": self.get_config(),
            "state_dict": self.state_dict(),
            "metadata": metadata or {},
            "metrics": metrics or {},
            "data_contract": data_contract or self.data_spec().to_dict(),
            "split_metadata": split_metadata or {},
        }

    def save_parameters(
        self,
        path,
        *,
        metadata: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        data_contract: dict[str, Any] | None = None,
        split_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save model configuration and fitted parameters."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            self.parameter_payload(
                metadata=metadata,
                metrics=metrics,
                data_contract=data_contract,
                split_metadata=split_metadata,
            ),
            path,
        )

    def load_parameters(self, path, *, map_location=None, strict: bool = True):
        """Load parameters into an existing model instance."""

        payload = torch.load(path, map_location=map_location)
        self.load_state_dict(payload["state_dict"], strict=strict)
        return payload

    @classmethod
    def from_saved(cls, path, *, map_location=None, strict: bool = True, **config_overrides):
        """Instantiate a model from a payload produced by `save_parameters`."""

        payload = torch.load(path, map_location=map_location)
        config = dict(payload.get("config", {}))
        config.update(config_overrides)
        model = cls(**config)
        model.load_state_dict(payload["state_dict"], strict=strict)
        return model


class FormulaOnlyCognitiveModel(BaseCognitiveModel):
    """Model with implemented equations but no dataframe contract yet."""

    required_columns: ClassVar[list[str]] = []


def load_saved_model(path, *, map_location=None, strict: bool = True, **config_overrides):
    """Load a saved model payload without manually importing its class."""

    payload = torch.load(path, map_location=map_location)
    module_name = LEGACY_MODEL_MODULES.get(payload["model_module"], payload["model_module"])
    module = importlib.import_module(
        module_name if module_name is not None else payload["model_module"]
    )
    model_class = getattr(module, payload["model_class"])
    config = dict(payload.get("config", {}))
    config.update(config_overrides)
    model = model_class(**config)
    model.load_state_dict(payload["state_dict"], strict=strict)
    return model, payload
