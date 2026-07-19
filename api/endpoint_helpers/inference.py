"""
Loads trained model pipelines and runs predictions against them.

Both pipelines (`logistic`, `tree`) are loaded once at API startup, per
`docs/architecture_and_design.md`, rather than re-loaded from disk on every
`/predict` call.
"""

from typing import Any

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from api.endpoint_helpers.config import APIConfig


class InferenceService:
    """Holds loaded model pipelines and runs predictions against them.

    Entry points:
        load: Load every model artifact from `config.artifacts_dir` into
            memory. Call once, at API startup.
        predict: Predict the satisfaction class + probabilities for one
            employee's features, using a named model.

    Args:
        config: API configuration holding the artifacts directory.
    """

    MODEL_NAMES = ("logistic", "tree")

    def __init__(self, config: APIConfig) -> None:
        self._config = config
        self._pipelines: dict[str, Pipeline] = {}

    def load(self) -> None:
        """Load every model artifact in `MODEL_NAMES` from disk into memory."""
        for name in self.MODEL_NAMES:
            artifact_path = self._config.artifacts_dir / f"{name}.joblib"
            self._pipelines[name] = joblib.load(artifact_path)

    def predict(self, model_name: str, features: dict[str, Any]) -> dict[str, Any]:
        """Predict the satisfaction class and per-class probabilities.

        Args:
            model_name: Which loaded pipeline to use — must be a key
                populated by `load` (one of `MODEL_NAMES`).
            features: Raw feature values, keyed by the same column names
                `training.helpers.preprocessing.FeatureEncoder.feature_names()`
                returns (e.g. `Age`, `Work_Location`).

        Returns:
            Dict with `label` (the predicted class) and
            `class_probabilities` (dict mapping each class to its predicted
            probability).

        Raises:
            KeyError: If `model_name` was never loaded (call `load` first,
                or check `model_name` is one of `MODEL_NAMES`).
        """
        pipeline = self._pipelines[model_name]
        features_df = pd.DataFrame([features])
        probabilities = pipeline.predict_proba(features_df)[0]
        label = pipeline.classes_[probabilities.argmax()]
        return {
            "label": label,
            "class_probabilities": dict(zip(pipeline.classes_, probabilities.tolist())),
        }
