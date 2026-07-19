"""
Shared evaluation metrics for the logistic and tree models.

Both model trainers evaluate through this one class, so their metrics
(confusion matrix, macro-F1, per-class ROC/AUC) are computed identically
and are directly comparable, per `docs/architecture_and_design.md`.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize


class ModelEvaluator:
    """Computes standard classification metrics for a fitted pipeline.

    Entry points:
        evaluate: Run all metrics for a fitted pipeline against a data split.

    Args:
        class_labels: The target's class labels, in the fixed order used
            for the confusion matrix rows/columns and per-class ROC/AUC.
    """

    def __init__(self, class_labels: list[str]) -> None:
        self._class_labels = class_labels

    def _confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> list[list[int]]:
        matrix = confusion_matrix(y_true, y_pred, labels=self._class_labels)
        return matrix.tolist()

    def _macro_f1(self, y_true: pd.Series, y_pred: np.ndarray) -> float:
        return float(f1_score(y_true, y_pred, average="macro"))

    def _roc_auc_per_class(
        self, y_true: pd.Series, y_proba: np.ndarray, proba_columns: np.ndarray
    ) -> dict[str, dict[str, Any]]:
        # One-vs-rest: binarize the multiclass target into one 0/1 column
        # per class, so each class gets its own ROC curve against "all
        # other classes" — the standard way to extend ROC/AUC beyond binary.
        # `proba_columns` (the pipeline's own `classes_`) fixes which
        # `y_proba` column belongs to which label — it is not guaranteed to
        # match `self._class_labels`'s order.
        y_true_binarized = label_binarize(y_true, classes=list(proba_columns))
        per_class: dict[str, dict[str, Any]] = {}
        for index, label in enumerate(proba_columns):
            fpr, tpr, _ = roc_curve(y_true_binarized[:, index], y_proba[:, index])
            auc = roc_auc_score(y_true_binarized[:, index], y_proba[:, index])
            per_class[label] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": float(auc),
            }
        return per_class

    def evaluate(
        self, pipeline: Pipeline, features: pd.DataFrame, target: pd.Series
    ) -> dict[str, Any]:
        """Compute confusion matrix, macro-F1, and per-class ROC/AUC.

        Args:
            pipeline: A fitted scikit-learn Pipeline exposing `predict` and
                `predict_proba`.
            features: Raw feature DataFrame for this split (train/val/test).
            target: True labels for this split.

        Returns:
            Dict with keys `confusion_matrix` (list of lists, ordered per
            `class_labels`), `macro_f1` (float), and `roc_auc` (dict mapping
            class label to its `fpr`/`tpr`/`auc`).
        """
        predictions = pipeline.predict(features)
        probabilities = pipeline.predict_proba(features)
        return {
            "confusion_matrix": self._confusion_matrix(target, predictions),
            "macro_f1": self._macro_f1(target, predictions),
            "roc_auc": self._roc_auc_per_class(
                target, probabilities, pipeline.classes_
            ),
        }
