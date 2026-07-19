"""
Pydantic request/response models for the FastAPI service.

Field names on `EmployeeFeatures` intentionally match the raw CSV's column
names (e.g. `Stress_Level`, not `stress_level`) rather than following
typical snake_case — these are the exact names
`training.helpers.preprocessing.FeatureEncoder.feature_names()` expects,
so a request body can be turned into a model-ready DataFrame with no
renaming step in between.
"""

from typing import Literal

from pydantic import BaseModel


class EmployeeFeatures(BaseModel):
    """The raw feature set a trained pipeline expects, one field per column."""

    Age: int
    Gender: str
    Job_Role: str
    Industry: str
    Years_of_Experience: int
    Work_Location: str
    Hours_Worked_Per_Week: int
    Number_of_Virtual_Meetings: int
    Work_Life_Balance_Rating: int
    Stress_Level: str
    Mental_Health_Condition: str
    Access_to_Mental_Health_Resources: str
    Productivity_Change: str
    Social_Isolation_Rating: int
    Company_Support_for_Remote_Work: int
    Physical_Activity: str
    Sleep_Quality: str
    Region: str


class EmployeeCreate(EmployeeFeatures):
    """Body for `POST /employees`.

    `Satisfaction_with_Remote_Work` is deliberately absent: a newly
    appearing employee (the case this endpoint simulates) is, by
    definition, someone whose satisfaction is not yet known — that's what
    `/predict` is for. Requiring it here would contradict the endpoint's
    own purpose.
    """

    Employee_ID: str


class EmployeeResponse(EmployeeFeatures):
    """Body for `GET /employees/{employee_id}`."""

    Employee_ID: str
    Satisfaction_with_Remote_Work: str | None


class PredictRequest(EmployeeFeatures):
    """Body for `POST /predict`."""

    model_name: Literal["logistic", "tree"] = "tree"


class PredictResponse(BaseModel):
    """Response for `POST /predict`."""

    model_used: Literal["logistic", "tree"]
    label: str
    class_probabilities: dict[str, float]


class ModelAnalysisResponse(BaseModel):
    """Response for `/analysis/logistic-importance` and `/analysis/tree-importance`."""

    macro_f1: float
    feature_importance: dict | list
    confusion_matrix: list[list[int]]
    roc_auc: dict[str, dict[str, float | list[float]]]


class DescriptiveAnalysisResponse(BaseModel):
    """Response for `GET /analysis/descriptive`."""

    target_distribution: dict[str, int]
    feature_distributions: dict[str, dict]
    feature_vs_target: dict[str, dict]
