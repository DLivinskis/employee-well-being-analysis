"""
Multinomial logistic regression baseline.

Provides an interpretable first pass — standardized coefficients give an
immediate read on whether there's linear signal at all, per
`docs/architecture_and_design.md`'s modeling section.
"""

from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from training.helpers.preprocessing import FeatureEncoder


class LogisticModelTrainer:
    """Fits a multinomial logistic regression and reports its coefficients.

    Entry points:
        train: Fit the pipeline on a training split.
        feature_importance: Standardized coefficients per class, ranked by
            absolute magnitude.

    Args:
        encoder: Shared FeatureEncoder describing how raw features are
            encoded.
        random_seed: Seed for the solver, for reproducibility.
    """

    def __init__(self, encoder: FeatureEncoder, random_seed: int) -> None:
        self._encoder = encoder
        self._random_seed = random_seed

    def _build_pipeline(self) -> Pipeline:
        pipeline = self._encoder.to_pipeline()
        # Standardizing after encoding puts one-hot and numeric columns on
        # the same scale, so coefficient magnitudes are comparable across
        # feature types rather than dominated by whichever had a larger
        # raw range.
        pipeline.steps.append(("scaling", StandardScaler()))
        pipeline.steps.append(
            ("model", LogisticRegression(max_iter=1000, random_state=self._random_seed))
        )
        return pipeline

    def train(self, features: pd.DataFrame, target: pd.Series) -> Pipeline:
        """Fit a logistic regression pipeline on the training split.

        Args:
            features: Raw training features.
            target: Training labels.

        Returns:
            The fitted Pipeline (preprocessing + scaling + model).
        """
        pipeline = self._build_pipeline()
        pipeline.fit(features, target)
        return pipeline

    def feature_importance(self, pipeline: Pipeline) -> dict[str, list[dict[str, Any]]]:
        """Standardized coefficients per class, ranked by absolute magnitude.

        Args:
            pipeline: A pipeline fitted by `train`.

        Returns:
            Dict mapping class label to a list of `{"feature",
            "coefficient"}` dicts, sorted by descending absolute
            coefficient.
        """
        feature_names = pipeline.named_steps["preprocessing"].get_feature_names_out()
        model: LogisticRegression = pipeline.named_steps["model"]
        importances: dict[str, list[dict[str, Any]]] = {}
        for class_index, class_label in enumerate(model.classes_):
            coefficients = model.coef_[class_index]
            ranked = sorted(
                zip(feature_names, coefficients),
                key=lambda pair: abs(pair[1]),
                reverse=True,
            )
            importances[class_label] = [
                {"feature": name, "coefficient": float(coefficient)}
                for name, coefficient in ranked
            ]
        return importances
