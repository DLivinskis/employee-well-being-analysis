"""
Thin HTTP client wrapping every FastAPI endpoint the Streamlit app calls.

Every page goes through this client rather than making `requests` calls
directly, so the app stays a presentation layer over one source of truth
(the API), per `docs/architecture_and_design.md`.
"""

import os
from typing import Any, Literal

import requests


class ApiClient:
    """Wraps HTTP calls to the FastAPI backend.

    Entry points:
        get_employee: Look up an employee's features by ID, or `None` if
            not found.
        create_employee: Insert a new employee record.
        predict: Predict satisfaction for a feature set, using a named model.
        get_descriptive: Descriptive statistics over `employees`.
        get_logistic_importance: Logistic model's coefficients + metrics.
        get_tree_importance: Tree model's importances + metrics.

    Args:
        base_url: Base URL of the API. Defaults to the `API_BASE_URL`
            environment variable, or `http://localhost:8000` for local dev.
    """

    def __init__(self, base_url: str | None = None) -> None:
        self._base_url = base_url or os.environ.get(
            "API_BASE_URL", "http://localhost:8000"
        )

    def _get(self, path: str) -> dict[str, Any]:
        response = requests.get(f"{self._base_url}{path}")
        response.raise_for_status()
        return response.json()

    def get_employee(self, employee_id: str) -> dict[str, Any] | None:
        """Look up an employee's stored features by ID.

        Args:
            employee_id: The `Employee_ID` to look up.

        Returns:
            The employee's feature dict, or `None` if no such ID exists.
        """
        response = requests.get(f"{self._base_url}/employees/{employee_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def create_employee(self, employee: dict[str, Any]) -> dict[str, Any]:
        """Insert a new employee record.

        Args:
            employee: Feature dict plus `Employee_ID`, matching
                `EmployeeCreate`.

        Returns:
            The created employee's response dict.

        Raises:
            requests.HTTPError: If the ID already exists (409) or the
                payload is invalid (422).
        """
        response = requests.post(f"{self._base_url}/employees", json=employee)
        response.raise_for_status()
        return response.json()

    def predict(
        self, features: dict[str, Any], model_name: Literal["logistic", "tree"]
    ) -> dict[str, Any]:
        """Predict satisfaction class + probabilities for one feature set.

        Args:
            features: Raw feature values (no `Employee_ID` or target).
            model_name: Which trained model to use.

        Returns:
            Dict with `model_used`, `label`, and `class_probabilities`.
        """
        payload = {**features, "model_name": model_name}
        response = requests.post(f"{self._base_url}/predict", json=payload)
        response.raise_for_status()
        return response.json()

    def get_descriptive(self) -> dict[str, Any]:
        """Return descriptive statistics over the current `employees` table."""
        return self._get("/analysis/descriptive")

    def get_logistic_importance(self) -> dict[str, Any]:
        """Return the logistic model's coefficients, confusion matrix, ROC/AUC."""
        return self._get("/analysis/logistic-importance")

    def get_tree_importance(self) -> dict[str, Any]:
        """Return the tree model's importances, confusion matrix, ROC/AUC."""
        return self._get("/analysis/tree-importance")
