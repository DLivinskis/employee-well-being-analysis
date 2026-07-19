"""
Reusable descriptive-statistics computations for the employee well-being dataset.

This is the single source of truth for the numbers shown on the Streamlit
"Descriptive Analysis" page and served by the API's `GET /analysis/descriptive`
endpoint. The companion notebook (`ad_hoc_exploratory_analysis.ipynb`) is for
personal, throwaway exploration only — nothing it computes should be treated
as authoritative or re-derived by hand elsewhere.
"""

from typing import Any, ClassVar

import pandas as pd
from scipy.stats import chi2_contingency, f_oneway


class DescriptiveStats:
    """Computes distributions and feature-vs-target breakdowns for the dataset.

    Entry points:
        target_distribution: Row counts for each satisfaction class.
        feature_distributions: Distribution (value counts or summary stats)
            for every categorical and numeric feature.
        feature_vs_target: For every feature, its relationship to the target
            (crosstab or group means) plus a significance test p-value.

    Args:
        data: Raw employee records, as loaded from the CSV (must be read with
            `keep_default_na=False, na_values=[]` — the dataset uses the
            literal string "None" as a real category, e.g. "no mental health
            condition", not as a missing value).
        target_col: Name of the target column to break every feature down
            against.
    """

    # ClassVar: shared by every instance rather than re-listed in __init__,
    # since which columns are categorical/numeric is a property of the
    # dataset schema, not of any particular DescriptiveStats instance.
    CATEGORICAL_COLUMNS: ClassVar[list[str]] = [
        "Gender",
        "Job_Role",
        "Industry",
        "Work_Location",
        "Stress_Level",
        "Mental_Health_Condition",
        "Access_to_Mental_Health_Resources",
        "Productivity_Change",
        "Physical_Activity",
        "Sleep_Quality",
        "Region",
    ]

    NUMERIC_COLUMNS: ClassVar[list[str]] = [
        "Age",
        "Years_of_Experience",
        "Hours_Worked_Per_Week",
        "Number_of_Virtual_Meetings",
        "Work_Life_Balance_Rating",
        "Social_Isolation_Rating",
        "Company_Support_for_Remote_Work",
    ]

    def __init__(
        self,
        data: pd.DataFrame,
        target_col: str = "Satisfaction_with_Remote_Work",
    ) -> None:
        self._data = data
        self._target_col = target_col

    def _categorical_vs_target(self, column: str) -> dict[str, Any]:
        crosstab = pd.crosstab(self._data[column], self._data[self._target_col])
        proportions = pd.crosstab(
            self._data[column], self._data[self._target_col], normalize="index"
        )
        _, p_value, _, _ = chi2_contingency(crosstab)
        return {
            "counts": crosstab,
            "proportions": proportions,
            "p_value": float(p_value),
            "test": "chi-square",
        }

    def _numeric_vs_target(self, column: str) -> dict[str, Any]:
        # Comprehension: builds one sample array per target class so f_oneway
        # can compare all classes' means for this column in a single call.
        groups = [
            self._data.loc[self._data[self._target_col] == level, column]
            for level in self._data[self._target_col].unique()
        ]
        group_means = self._data.groupby(self._target_col)[column].mean()
        _, p_value = f_oneway(*groups)
        return {
            "group_means": group_means,
            "p_value": float(p_value),
            "test": "anova",
        }

    def target_distribution(self) -> pd.Series:
        """Row count for each satisfaction class.

        Returns:
            Series indexed by satisfaction class, values are row counts.
        """
        return self._data[self._target_col].value_counts()

    def feature_distributions(self) -> dict[str, pd.Series]:
        """Distribution of every categorical and numeric feature.

        Returns:
            Dict mapping column name to its value counts (categorical) or
            `describe()` summary (numeric).
        """
        distributions: dict[str, pd.Series] = {}
        for column in self.CATEGORICAL_COLUMNS:
            distributions[column] = self._data[column].value_counts()
        for column in self.NUMERIC_COLUMNS:
            distributions[column] = self._data[column].describe()
        return distributions

    def feature_vs_target(self) -> dict[str, dict[str, Any]]:
        """Relationship between every feature and the target, with a p-value.

        Categorical features are compared via a chi-square test on their
        crosstab with the target; numeric features via one-way ANOVA across
        target groups. The p-value is a triage signal only — for deciding
        which features are worth carrying into modeling — not a substitute
        for the trained models' own feature importance.

        How to interpret the result for a given column, e.g. `result["Gender"]`:
            - `test`: which test produced the p-value below — `"chi-square"`
              for a categorical column, `"anova"` for a numeric one. Read the
              other keys accordingly (`counts`/`proportions` only exist for
              chi-square columns, `group_means` only for anova columns).
            - `p_value`: probability of seeing an association this strong (or
              stronger) between the column and the target if there were truly
              no relationship. Small (conventionally < 0.05) suggests the
              column is genuinely related to the target; large means the
              observed differences are consistent with random noise. With 17
              features tested independently here, treat this as a rough
              triage signal rather than a strict cutoff — at p < 0.05 you'd
              expect roughly one feature in 20 to clear that bar by chance
              alone, so a single borderline p-value across many tests is
              weaker evidence than the same p-value would be from one test in
              isolation (a Bonferroni-style threshold of 0.05/17 ≈ 0.003 is
              the more conservative bar for "this alone is convincing").
            - `counts` (categorical only): raw crosstab of column category ×
              target class — read this for sample size per cell before
              trusting `proportions` (a proportion from a tiny cell is noisy).
            - `proportions` (categorical only): the same crosstab normalized
              per row, i.e. "of employees in this category, what fraction
              landed in each target class" — the shape to eyeball for whether
              a category skews toward one satisfaction level.
            - `group_means` (numeric only): the column's mean value within
              each target class — a large spread across classes suggests the
              numeric feature separates the target; near-identical means
              suggest it doesn't.

        Returns:
            Dict mapping column name to its breakdown dict (see
            `_categorical_vs_target` / `_numeric_vs_target` for keys).
        """
        breakdowns: dict[str, dict[str, Any]] = {}
        for column in self.CATEGORICAL_COLUMNS:
            breakdowns[column] = self._categorical_vs_target(column)
        for column in self.NUMERIC_COLUMNS:
            breakdowns[column] = self._numeric_vs_target(column)
        return breakdowns
