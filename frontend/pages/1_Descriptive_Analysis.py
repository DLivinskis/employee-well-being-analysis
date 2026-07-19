"""
Descriptive Analysis page.

Renders target/feature distributions and feature-vs-target breakdowns from
`GET /analysis/descriptive`.
"""

from typing import Any

import pandas as pd
import streamlit as st

from api_client import ApiClient

NUMERIC_SUMMARY_KEYS = {"mean", "std", "min", "25%", "50%", "75%", "max"}


class DescriptiveAnalysisView:
    """Renders the descriptive-analysis response as Streamlit charts/tables.

    Entry points:
        render: Draw the full page.

    Args:
        analysis: The parsed `GET /analysis/descriptive` response.
    """

    def __init__(self, analysis: dict[str, Any]) -> None:
        self._analysis = analysis

    def _is_numeric_summary(self, distribution: dict[str, Any]) -> bool:
        return NUMERIC_SUMMARY_KEYS.issubset(distribution.keys())

    def _render_target_distribution(self) -> None:
        st.subheader("Target distribution")
        series = pd.Series(self._analysis["target_distribution"], name="count")
        st.bar_chart(series)

    def _render_feature_distributions(self) -> None:
        st.subheader("Feature distributions")
        column = st.selectbox(
            "Feature", sorted(self._analysis["feature_distributions"].keys())
        )
        distribution = self._analysis["feature_distributions"][column]
        if self._is_numeric_summary(distribution):
            st.table(pd.Series(distribution, name=column))
        else:
            st.bar_chart(pd.Series(distribution, name="count"))

    def _render_feature_vs_target(self) -> None:
        st.subheader("Feature vs. target")
        p_values = {
            column: breakdown["p_value"]
            for column, breakdown in self._analysis["feature_vs_target"].items()
        }
        p_value_table = pd.Series(p_values, name="p_value").sort_values()
        st.write(
            "Chi-square (categorical) / ANOVA (numeric) p-value per feature — "
            "see `exploratory_analysis/statistics_concepts.md` for how to "
            "read these. With many features tested independently, treat "
            "borderline values cautiously (see `docs/eda_findings.md`)."
        )
        st.table(p_value_table)

        column = st.selectbox(
            "Inspect one feature's breakdown",
            sorted(self._analysis["feature_vs_target"].keys()),
        )
        breakdown = self._analysis["feature_vs_target"][column]
        if breakdown["test"] == "chi-square":
            st.write(f"{column}: proportion of each satisfaction class, by category")
            st.bar_chart(pd.DataFrame(breakdown["proportions"]).T)
        else:
            st.write(f"{column}: mean value, by satisfaction class")
            st.bar_chart(pd.Series(breakdown["group_means"]))

    def render(self) -> None:
        """Draw the target distribution, feature distributions, and
        feature-vs-target sections."""
        self._render_target_distribution()
        self._render_feature_distributions()
        self._render_feature_vs_target()


st.set_page_config(page_title="Descriptive Analysis", page_icon="📈")
st.title("📈 Descriptive Analysis")

analysis = ApiClient().get_descriptive()
DescriptiveAnalysisView(analysis).render()
