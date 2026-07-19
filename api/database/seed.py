"""
One-time database seeding: raw CSV into `employees`, training metadata into
`models`.

Both seed steps are idempotent so re-running the API's startup hook never
duplicates rows: `employees` is only loaded from the CSV once (later
`POST /employees` inserts are left untouched), and `models` rows are
replaced by name rather than accumulated.
"""

import json
from datetime import datetime

import pandas as pd
from sqlalchemy import select

from api.config import APIConfig
from api.database.connection import Database
from api.database.models import Employee, TrainedModel

# Maps the CSV's column names to Employee's snake_case attribute names.
CSV_COLUMN_TO_ATTRIBUTE = {
    "Employee_ID": "employee_id",
    "Age": "age",
    "Gender": "gender",
    "Job_Role": "job_role",
    "Industry": "industry",
    "Years_of_Experience": "years_of_experience",
    "Work_Location": "work_location",
    "Hours_Worked_Per_Week": "hours_worked_per_week",
    "Number_of_Virtual_Meetings": "number_of_virtual_meetings",
    "Work_Life_Balance_Rating": "work_life_balance_rating",
    "Stress_Level": "stress_level",
    "Mental_Health_Condition": "mental_health_condition",
    "Access_to_Mental_Health_Resources": "access_to_mental_health_resources",
    "Productivity_Change": "productivity_change",
    "Social_Isolation_Rating": "social_isolation_rating",
    "Satisfaction_with_Remote_Work": "satisfaction_with_remote_work",
    "Company_Support_for_Remote_Work": "company_support_for_remote_work",
    "Physical_Activity": "physical_activity",
    "Sleep_Quality": "sleep_quality",
    "Region": "region",
}


class DatabaseSeeder:
    """Loads the CSV into `employees` and training metadata into `models`.

    Entry points:
        seed_employees: Load the CSV into `employees`, only if empty.
        seed_models: Upsert each trained model's metadata into `models`.

    Args:
        config: API configuration holding the CSV and artifacts paths.
        database: Database instance to seed.
    """

    def __init__(self, config: APIConfig, database: Database) -> None:
        self._config = config
        self._database = database

    def _load_csv(self) -> pd.DataFrame:
        # keep_default_na=False, na_values=[]: the CSV uses the literal
        # string "None" as a real category (e.g. no mental health
        # condition), which pandas would otherwise silently parse as a
        # missing value — see docs/eda_findings.md.
        return pd.read_csv(
            self._config.raw_data_path, keep_default_na=False, na_values=[]
        )

    def _row_to_employee(self, row: pd.Series) -> Employee:
        attributes = {
            CSV_COLUMN_TO_ATTRIBUTE[column]: value for column, value in row.items()
        }
        return Employee(**attributes)

    def seed_employees(self) -> None:
        """Load the CSV into `employees`, but only if the table is empty."""
        with self._database.session() as session:
            already_seeded = session.execute(select(Employee.employee_id).limit(1))
            if already_seeded.first() is not None:
                return
            data = self._load_csv()
            session.add_all(
                self._row_to_employee(row) for _, row in data.iterrows()
            )

    def seed_models(self) -> None:
        """Upsert each trained model's metadata from `metadata.json`.

        Replaces any existing row for the same model name — only the
        latest training run's metrics/importances are ever kept.
        """
        metadata_path = self._config.artifacts_dir / "metadata.json"
        with metadata_path.open() as f:
            metadata = json.load(f)

        with self._database.session() as session:
            for name, model_metadata in metadata.items():
                existing = session.get(TrainedModel, name)
                if existing is not None:
                    session.delete(existing)
                    session.flush()
                session.add(
                    TrainedModel(
                        name=name,
                        trained_at=datetime.now(),
                        artifact_path=model_metadata["artifact_path"],
                        macro_f1=model_metadata["test_metrics"]["macro_f1"],
                        feature_importance=model_metadata["feature_importance"],
                        confusion_matrix=model_metadata["test_metrics"][
                            "confusion_matrix"
                        ],
                        roc_auc=model_metadata["test_metrics"]["roc_auc"],
                    )
                )
