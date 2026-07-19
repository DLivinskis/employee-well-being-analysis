"""
SQLAlchemy ORM model for the `models` table.

Caches each trained model's metadata and evaluation results, populated
from `training/run_training.py`'s `metadata.json`, per
`docs/architecture_and_design.md`.
"""

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, String
from sqlalchemy.orm import Mapped, mapped_column

from api.database.models.base import Base


class TrainedModel(Base):
    """Metadata and evaluation results for one trained model.

    One row per model name (`logistic`, `tree`); re-running training
    upserts (replaces) that model's row rather than accumulating history,
    since only the latest trained artifact is ever served.
    """

    __tablename__ = "models"

    name: Mapped[str] = mapped_column(String, primary_key=True)
    trained_at: Mapped[datetime] = mapped_column(DateTime)
    artifact_path: Mapped[str] = mapped_column(String)
    macro_f1: Mapped[float] = mapped_column(Float)
    feature_importance: Mapped[dict] = mapped_column(JSON)
    confusion_matrix: Mapped[dict] = mapped_column(JSON)
    roc_auc: Mapped[dict] = mapped_column(JSON)
