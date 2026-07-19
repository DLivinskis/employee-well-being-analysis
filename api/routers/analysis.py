"""
`GET /analysis/descriptive`, `GET /analysis/logistic-importance`,
`GET /analysis/tree-importance`.
"""

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from api.database.models import Employee, TrainedModel
from api.endpoint_helpers.dependencies import get_session
from api.endpoint_helpers.schemas import DescriptiveAnalysisResponse, ModelAnalysisResponse
from exploratory_analysis.descriptive_stats import DescriptiveStats

router = APIRouter(prefix="/analysis", tags=["analysis"])


class DescriptiveAnalysisBuilder:
    """Builds a JSON-safe descriptive-analysis response from the DB.

    Entry points:
        build: Load `employees` and return a `DescriptiveAnalysisResponse`.

    Args:
        session: DB session to read `employees` from.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def _load_employees_dataframe(self) -> pd.DataFrame:
        employees = self._session.execute(select(Employee)).scalars().all()
        rows = [
            {
                csv_column: getattr(employee, attribute)
                for csv_column, attribute in Employee.CSV_COLUMN_TO_ATTRIBUTE.items()
            }
            for employee in employees
        ]
        return pd.DataFrame(rows)

    def _serialize_feature_vs_target(
        self, breakdowns: dict[str, dict]
    ) -> dict[str, dict]:
        serialized: dict[str, dict] = {}
        for column, breakdown in breakdowns.items():
            entry = {"test": breakdown["test"], "p_value": breakdown["p_value"]}
            if breakdown["test"] == "chi-square":
                entry["proportions"] = breakdown["proportions"].to_dict(orient="index")
            else:
                entry["group_means"] = breakdown["group_means"].to_dict()
            serialized[column] = entry
        return serialized

    def build(self) -> DescriptiveAnalysisResponse:
        """Compute descriptive statistics over the current `employees` table.

        Returns:
            A `DescriptiveAnalysisResponse` with plain dicts (no pandas
            objects), safe to serialize as JSON.
        """
        data = self._load_employees_dataframe()
        stats = DescriptiveStats(data, target_col="Satisfaction_with_Remote_Work")
        return DescriptiveAnalysisResponse(
            target_distribution=stats.target_distribution().to_dict(),
            feature_distributions={
                column: series.to_dict()
                for column, series in stats.feature_distributions().items()
            },
            feature_vs_target=self._serialize_feature_vs_target(
                stats.feature_vs_target()
            ),
        )


def _model_analysis_response(name: str, session: Session) -> ModelAnalysisResponse:
    model = session.get(TrainedModel, name)
    if model is None:
        raise HTTPException(status_code=404, detail=f"No trained model named {name!r}")
    return ModelAnalysisResponse(
        macro_f1=model.macro_f1,
        feature_importance=model.feature_importance,
        confusion_matrix=model.confusion_matrix,
        roc_auc=model.roc_auc,
    )


@router.get("/descriptive", response_model=DescriptiveAnalysisResponse)
def get_descriptive_analysis(
    session: Session = Depends(get_session),
) -> DescriptiveAnalysisResponse:
    """Return distributions and feature-vs-target breakdowns for `employees`."""
    return DescriptiveAnalysisBuilder(session).build()


@router.get("/logistic-importance", response_model=ModelAnalysisResponse)
def get_logistic_importance(
    session: Session = Depends(get_session),
) -> ModelAnalysisResponse:
    """Return the logistic model's coefficients, confusion matrix, and ROC/AUC."""
    return _model_analysis_response("logistic", session)


@router.get("/tree-importance", response_model=ModelAnalysisResponse)
def get_tree_importance(
    session: Session = Depends(get_session),
) -> ModelAnalysisResponse:
    """Return the tree model's importances, confusion matrix, and ROC/AUC."""
    return _model_analysis_response("tree", session)
