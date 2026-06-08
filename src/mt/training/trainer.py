"""Training utilities for cognitive models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import schedulefree
import torch
from tqdm import tqdm

from mt.evaluation.metrics import choice_nll
from mt.evaluation.results import EvaluationResult, FitEvaluationResult, TrainingResult


class Trainer:
    def __init__(self, model, num_iter: int = 1000, lr: float = 0.1, show_progress: bool = True):
        self.model = model
        self.num_iter = num_iter
        self.show_progress = show_progress
        self.optimizer = schedulefree.AdamWScheduleFree(self.model.parameters(), lr=lr)

    def preprocess_dataframes(self, train_df, eval_df):
        """Convert train/eval dataframes into model-ready tensor dictionaries."""

        return self.model.preprocess_data(train_df, eval_df)

    def fit(
        self,
        train_data: dict[str, Any],
        *,
        save_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        split_metadata: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """Fit the model on already-prepared tensor data."""

        self.model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()

        loss_history: list[float] = []
        iterator = tqdm(range(self.num_iter), disable=not self.show_progress)
        for _ in iterator:
            self.optimizer.zero_grad()
            logits = self.model(train_data)
            loss = choice_nll(logits, train_data["choice"])
            loss.backward()
            self.optimizer.step()
            loss_value = float(loss.detach().item())
            loss_history.append(loss_value)
            iterator.set_postfix(loss=loss_value)

        result = TrainingResult(
            final_loss=loss_history[-1] if loss_history else float("nan"),
            loss_history=tuple(loss_history),
            num_steps=self.num_iter,
            metadata=metadata or {},
        )

        if save_path is not None:
            self.model.save_parameters(
                save_path,
                metadata=metadata,
                metrics={"train": result.to_dict()},
                split_metadata=split_metadata,
            )

        return result

    def evaluate(
        self,
        eval_data: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Evaluate the current model on already-prepared tensor data."""

        self.model.eval()
        if hasattr(self.optimizer, "eval"):
            self.optimizer.eval()

        with torch.no_grad():
            logits = self.model(eval_data)
            loss = choice_nll(logits, eval_data["choice"])

        return EvaluationResult(
            loss=loss,
            metrics={"choice_nll": float(loss.item())},
            metadata=metadata or {},
        )

    def fit_and_evaluate_result(
        self,
        train_df,
        eval_df,
        *,
        save_path: str | Path | None = None,
        metadata: dict[str, Any] | None = None,
        split_metadata: dict[str, Any] | None = None,
    ) -> FitEvaluationResult:
        """Fit on dataframes and return structured train/eval results."""

        train_data, eval_data = self.preprocess_dataframes(train_df, eval_df)
        training = self.fit(train_data, metadata=metadata, split_metadata=split_metadata)
        evaluation = self.evaluate(eval_data, metadata=metadata)
        result = FitEvaluationResult(
            training=training,
            evaluation=evaluation,
            metadata=metadata or {},
        )

        if save_path is not None:
            self.model.save_parameters(
                save_path,
                metadata=metadata,
                metrics=result.to_dict(),
                split_metadata=split_metadata,
            )

        return result

    def fit_and_evaluate(self, train_df, eval_df):
        """Backward-compatible API returning only the evaluation loss tensor."""

        return self.fit_and_evaluate_result(train_df, eval_df).evaluation.loss
