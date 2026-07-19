"""
Shared feature encoding for both the logistic and tree models.

One `FeatureEncoder`, fit once on the training split, is reused by both
model trainers and later by the API's inference path — so a caller only
ever needs to supply raw feature values, never pre-encoded ones.
"""

from typing import ClassVar

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


class FeatureEncoder:
    """Encodes raw employee features into a model-ready numeric matrix.

    Entry points:
        build: Construct the (unfitted) scikit-learn ColumnTransformer.
        to_pipeline: Wrap the ColumnTransformer in a single-step Pipeline.

    Ordinal columns are encoded with an explicit category order so their
    numeric encoding reflects the real ordering (e.g. "Low" < "Medium" <
    "High"); nominal columns are one-hot encoded since they have no natural
    order. `Employee_ID` and the target column are excluded — they're
    handled by the caller, not by this transformer.
    """

    # ClassVar: the column roles are a property of the dataset schema, not
    # of any particular FeatureEncoder instance.
    ORDINAL_COLUMNS: ClassVar[dict[str, list[str]]] = {
        "Stress_Level": ["Low", "Medium", "High"],
        "Sleep_Quality": ["Poor", "Average", "Good"],
        "Physical_Activity": ["None", "Weekly", "Daily"],
        "Productivity_Change": ["Decrease", "No Change", "Increase"],
        "Access_to_Mental_Health_Resources": ["No", "Yes"],
    }

    NOMINAL_COLUMNS: ClassVar[list[str]] = [
        "Gender",
        "Job_Role",
        "Industry",
        "Work_Location",
        "Mental_Health_Condition",
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

    def _build_ordinal_transformer(self) -> OrdinalEncoder:
        # `categories` fixes the encoded integer order per column (e.g.
        # Low=0, Medium=1, High=2) instead of alphabetical/appearance order.
        ordered_categories = [
            self.ORDINAL_COLUMNS[column] for column in self.ORDINAL_COLUMNS
        ]
        return OrdinalEncoder(categories=ordered_categories)

    def _build_nominal_transformer(self) -> OneHotEncoder:
        # sparse_output=False: keeps the ColumnTransformer's output dense so
        # it can feed straight into StandardScaler (the logistic model's
        # pipeline), which does not support sparse input with centering.
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    def feature_names(self) -> list[str]:
        """All raw column names this encoder expects, in a fixed order.

        Returns:
            Column names covering every ordinal, nominal, and numeric
            feature (excludes `Employee_ID` and the target column).
        """
        return (
            list(self.ORDINAL_COLUMNS.keys())
            + self.NOMINAL_COLUMNS
            + self.NUMERIC_COLUMNS
        )

    def build(self) -> ColumnTransformer:
        """Construct the unfitted ColumnTransformer for all feature groups.

        Returns:
            A ColumnTransformer with ordinal, one-hot, and passthrough
            (numeric) branches, ready to be fit on a training DataFrame
            inside a model `Pipeline`.
        """
        return ColumnTransformer(
            transformers=[
                (
                    "ordinal",
                    self._build_ordinal_transformer(),
                    list(self.ORDINAL_COLUMNS.keys()),
                ),
                ("nominal", self._build_nominal_transformer(), self.NOMINAL_COLUMNS),
                ("numeric", "passthrough", self.NUMERIC_COLUMNS),
            ]
        )

    def to_pipeline(self) -> Pipeline:
        """Wrap this encoder's ColumnTransformer in a single-step Pipeline.

        Returns:
            A Pipeline with one "preprocessing" step, so model trainers can
            append their estimator as a second step and fit/predict
            end-to-end on raw feature DataFrames.
        """
        return Pipeline(steps=[("preprocessing", self.build())])
