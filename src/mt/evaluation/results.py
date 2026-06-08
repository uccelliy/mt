"""Result containers for model training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingResult:
    """Summary of a model fitting run."""

    final_loss: float
    loss_history: tuple[float, ...]
    num_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_loss": self.final_loss,
            "loss_history": list(self.loss_history),
            "num_steps": self.num_steps,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationResult:
    """Summary of a model evaluation run."""

    loss: Any
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def loss_value(self) -> float:
        if hasattr(self.loss, "item"):
            return float(self.loss.item())
        return float(self.loss)

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss_value,
            "metrics": dict(self.metrics),
            "metadata": self.metadata,
        }


@dataclass
class FitEvaluationResult:
    """Combined result from fitting a model and evaluating it."""

    training: TrainingResult
    evaluation: EvaluationResult
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "metadata": self.metadata,
        }
