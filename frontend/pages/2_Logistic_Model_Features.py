"""
Logistic Model — Top Features page.

Renders standardized coefficients per class, plus the shared confusion
matrix / ROC-AUC visuals, from `GET /analysis/logistic-importance`.
"""

import pandas as pd
import streamlit as st

from api_client import ApiClient
from helpers.model_analysis import ModelAnalysisView


class LogisticFeaturesView(ModelAnalysisView):
    """Adds per-class coefficient rendering to the shared model-analysis view.

    Entry points:
        render: Draw the full page.
    """

    def _render_coefficients(self) -> None:
        st.subheader("Standardized coefficients")
        class_label = st.selectbox("Class", self._class_labels)
        rows = self._analysis["feature_importance"][class_label]
        coefficients = pd.Series(
            {row["feature"]: row["coefficient"] for row in rows}, name="coefficient"
        ).sort_values()
        st.bar_chart(coefficients)

    def render(self) -> None:
        """Draw macro-F1, coefficients, confusion matrix, and ROC curves."""
        self.render_macro_f1()
        self._render_coefficients()
        self.render_confusion_matrix()
        self.render_roc_curves()


st.set_page_config(page_title="Logistic Model Features", page_icon="📊")
st.title("📊 Logistic Model — Top Features")

analysis = ApiClient().get_logistic_importance()
LogisticFeaturesView(analysis).render()
