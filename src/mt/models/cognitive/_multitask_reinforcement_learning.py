"""Multi-task reinforcement learning model placeholder.

The Centaur supplementary model-details section lists this model and its tasks,
but does not provide equations. The data contract and equations should be added
once the original model specification is available.
"""

from __future__ import annotations

from mt.models.common._base import FormulaOnlyCognitiveModel


class MultiTaskReinforcementLearningModel(FormulaOnlyCognitiveModel):
    def compute_logits(self, data):
        raise NotImplementedError(
            "The Centaur supplement lists this model but does not provide equations."
        )
