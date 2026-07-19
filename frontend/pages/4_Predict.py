"""
Predict page.

Covers both input modes from `docs/architecture_and_design.md`: fetch an
existing employee's features by ID, or enter them manually — either path
ends in the same call to `POST /predict`. If a looked-up ID doesn't exist,
offers to save the manually entered features as a new employee via
`POST /employees` before predicting, simulating a new hire appearing.
"""

from typing import Any, ClassVar

import pandas as pd
import requests
import streamlit as st

from api_client import ApiClient


class PredictPage:
    """Renders the client-ID-lookup / manual-entry predict form.

    Entry points:
        render: Draw the full page.

    Args:
        client: API client used for employee lookup/creation and predicting.
    """

    CATEGORICAL_OPTIONS: ClassVar[dict[str, list[str]]] = {
        "Gender": ["Non-binary", "Female", "Male", "Prefer not to say"],
        "Job_Role": [
            "HR", "Data Scientist", "Software Engineer", "Sales",
            "Marketing", "Designer", "Project Manager",
        ],
        "Industry": [
            "Healthcare", "IT", "Education", "Finance",
            "Consulting", "Manufacturing", "Retail",
        ],
        "Work_Location": ["Hybrid", "Remote", "Onsite"],
        "Stress_Level": ["Low", "Medium", "High"],
        "Mental_Health_Condition": ["None", "Anxiety", "Burnout", "Depression"],
        "Access_to_Mental_Health_Resources": ["No", "Yes"],
        "Productivity_Change": ["Decrease", "No Change", "Increase"],
        "Physical_Activity": ["None", "Weekly", "Daily"],
        "Sleep_Quality": ["Poor", "Average", "Good"],
        "Region": [
            "Europe", "Asia", "North America",
            "South America", "Oceania", "Africa",
        ],
    }

    NUMERIC_BOUNDS: ClassVar[dict[str, tuple[int, int]]] = {
        "Age": (22, 60),
        "Years_of_Experience": (0, 40),
        "Hours_Worked_Per_Week": (20, 60),
        "Number_of_Virtual_Meetings": (0, 20),
        "Work_Life_Balance_Rating": (1, 5),
        "Social_Isolation_Rating": (1, 5),
        "Company_Support_for_Remote_Work": (1, 5),
    }

    def __init__(self, client: ApiClient) -> None:
        self._client = client

    def _lookup_by_id(self) -> tuple[str, dict[str, Any] | None]:
        employee_id = st.text_input("Employee ID (optional — leave blank for a new employee)")
        prefill: dict[str, Any] | None = None
        if employee_id:
            found = self._client.get_employee(employee_id)
            if found is None:
                st.info(f"No employee {employee_id!r} found — fill in the form below.")
            else:
                st.success(f"Loaded {employee_id!r} — fields below are pre-filled.")
                prefill = found
        return employee_id, prefill

    def _feature_form(self, prefill: dict[str, Any] | None) -> dict[str, Any] | None:
        prefill = prefill or {}
        with st.form("predict_form"):
            features: dict[str, Any] = {}
            for column, options in self.CATEGORICAL_OPTIONS.items():
                default = prefill.get(column, options[0])
                features[column] = st.selectbox(
                    column, options, index=options.index(default)
                )
            for column, (low, high) in self.NUMERIC_BOUNDS.items():
                features[column] = st.number_input(
                    column, min_value=low, max_value=high,
                    value=int(prefill.get(column, low)),
                )
            model_name = st.radio("Model", ["tree", "logistic"], horizontal=True)
            submitted = st.form_submit_button("Predict")
        if not submitted:
            return None
        features["model_name"] = model_name
        return features

    def _render_prediction(self, result: dict[str, Any]) -> None:
        st.subheader(f"Prediction ({result['model_used']} model)")
        st.metric("Predicted satisfaction", result["label"])
        st.bar_chart(pd.Series(result["class_probabilities"], name="probability"))

    def _offer_save_new_employee(
        self, employee_id: str, features: dict[str, Any]
    ) -> None:
        if not employee_id:
            return
        if self._client.get_employee(employee_id) is not None:
            return
        if st.checkbox(f"Save {employee_id!r} as a new employee record"):
            try:
                self._client.create_employee({"Employee_ID": employee_id, **features})
                st.success(f"Saved {employee_id!r}.")
            except requests.HTTPError as error:
                st.error(f"Could not save employee: {error}")

    def render(self) -> None:
        """Draw the ID-lookup field, feature form, and prediction result."""
        employee_id, prefill = self._lookup_by_id()
        submission = self._feature_form(prefill)
        if submission is None:
            return
        model_name = submission.pop("model_name")
        result = self._client.predict(submission, model_name)
        self._render_prediction(result)
        self._offer_save_new_employee(employee_id, submission)


st.set_page_config(page_title="Predict", page_icon="🔮")
st.title("🔮 Predict")

PredictPage(ApiClient()).render()
