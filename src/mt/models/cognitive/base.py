"""Shared base classes for cognitive models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import torch
from torch import nn


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

    def parameter_payload(self, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build a serializable parameter payload."""

        return {
            "model_class": self.__class__.__name__,
            "model_module": self.__class__.__module__,
            "config": self.get_config(),
            "state_dict": self.state_dict(),
            "metadata": metadata or {},
        }

    def save_parameters(self, path, *, metadata: dict[str, Any] | None = None) -> None:
        """Save model configuration and fitted parameters."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.parameter_payload(metadata=metadata), path)

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
