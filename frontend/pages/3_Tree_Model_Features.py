"""
Tree Model — Top Features page.

Renders permutation feature importance (one value per raw feature, not
per class — see `training/train_tree.py`), plus the shared confusion
matrix / ROC-AUC visuals, from `GET /analysis/tree-importance`.
"""

import pandas as pd
import streamlit as st

from frontend.api_client import ApiClient
from frontend.helpers.model_analysis import ModelAnalysisView


class TreeFeaturesView(ModelAnalysisView):
    """Adds permutation-importance rendering to the shared model-analysis view.

    Entry points:
        render: Draw the full page.
    """

    def _render_importances(self) -> None:
        st.subheader("Permutation importance")
        rows = self._analysis["feature_importance"]
        importances = pd.Series(
            {row["feature"]: row["importance"] for row in rows}, name="importance"
        ).sort_values()
        st.bar_chart(importances)

    def render(self) -> None:
        """Draw macro-F1, importances, confusion matrix, and ROC curves."""
        self.render_macro_f1()
        self._render_importances()
        self.render_confusion_matrix()
        self.render_roc_curves()


st.set_page_config(page_title="Tree Model Features", page_icon="🌳")
st.title("🌳 Tree Model — Top Features")

analysis = ApiClient().get_tree_importance()
TreeFeaturesView(analysis).render()
