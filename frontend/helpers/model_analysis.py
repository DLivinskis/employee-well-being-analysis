"""
Shared rendering for the logistic and tree model pages.

Confusion matrix and ROC/AUC rendering are identical between the two
models (per `docs/architecture_and_design.md`'s "standard evaluation
visuals" note) — only feature importance differs, so it's left to each
page's own subclass of `ModelAnalysisView`.
"""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


class ModelAnalysisView:
    """Renders the shared parts of a model-analysis response.

    Entry points:
        render_macro_f1: Show the model's macro-F1 as a metric.
        render_confusion_matrix: Show the confusion matrix as a table.
        render_roc_curves: Plot one ROC curve per class, with AUC in the legend.

    Args:
        analysis: A `GET /analysis/{model}-importance` response — must
            contain `macro_f1`, `confusion_matrix`, and `roc_auc`.
    """

    def __init__(self, analysis: dict[str, Any]) -> None:
        self._analysis = analysis
        # Confusion matrix rows/columns and ROC/AUC's per-class keys are
        # both ordered by the model's own `classes_` (see
        # `training/helpers/evaluation.py`), which scikit-learn keeps
        # sorted for string labels — so sorting roc_auc's keys recovers
        # the same order the confusion matrix was built with.
        self._class_labels = sorted(analysis["roc_auc"].keys())

    def render_macro_f1(self) -> None:
        """Show macro-F1 as a Streamlit metric."""
        st.metric("Macro-F1 (test set)", f"{self._analysis['macro_f1']:.3f}")

    def render_confusion_matrix(self) -> None:
        """Show the confusion matrix as a labeled table."""
        st.subheader("Confusion matrix (test set)")
        matrix = pd.DataFrame(
            self._analysis["confusion_matrix"],
            index=[f"true: {label}" for label in self._class_labels],
            columns=[f"pred: {label}" for label in self._class_labels],
        )
        st.table(matrix)

    def render_roc_curves(self) -> None:
        """Plot one-vs-rest ROC curves, one per class, with AUC in the legend."""
        st.subheader("ROC curves (test set, one-vs-rest)")
        figure, axes = plt.subplots()
        for label in self._class_labels:
            curve = self._analysis["roc_auc"][label]
            axes.plot(curve["fpr"], curve["tpr"], label=f"{label} (AUC={curve['auc']:.3f})")
        axes.plot([0, 1], [0, 1], linestyle="--", color="gray", label="chance")
        axes.set_xlabel("False positive rate")
        axes.set_ylabel("True positive rate")
        axes.legend()
        st.pyplot(figure)
