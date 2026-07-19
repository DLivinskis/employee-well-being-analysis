"""
Random forest comparison model.

Captures non-linear/interaction effects the logistic baseline can't, per
`docs/architecture_and_design.md`'s modeling section. Feature importance is
computed via permutation importance rather than the model's built-in
impurity-based importances, since permutation importance isn't biased
toward high-cardinality features the way impurity-based importance is.
"""

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from training.helpers.preprocessing import FeatureEncoder


class TreeModelTrainer:
    """Fits a random forest classifier and reports permutation importances.

    Entry points:
        train: Fit the pipeline on a training split.
        feature_importance: Permutation importance per raw feature, ranked
            by magnitude.

    Args:
        encoder: Shared FeatureEncoder describing how raw features are
            encoded.
        random_seed: Seed for the forest's bootstrap sampling and for
            permutation shuffling.
    """

    def __init__(self, encoder: FeatureEncoder, random_seed: int) -> None:
        self._encoder = encoder
        self._random_seed = random_seed

    def _build_pipeline(self) -> Pipeline:
        pipeline = self._encoder.to_pipeline()
        pipeline.steps.append(
            (
                "model",
                RandomForestClassifier(n_estimators=300, random_state=self._random_seed),
            )
        )
        return pipeline

    def train(self, features: pd.DataFrame, target: pd.Series) -> Pipeline:
        """Fit a random forest pipeline on the training split.

        Args:
            features: Raw training features.
            target: Training labels.

        Returns:
            The fitted Pipeline (preprocessing + model).
        """
        pipeline = self._build_pipeline()
        pipeline.fit(features, target)
        return pipeline

    def feature_importance(
        self, pipeline: Pipeline, features: pd.DataFrame, target: pd.Series
    ) -> list[dict[str, Any]]:
        """Permutation importance for each raw feature, ranked by magnitude.

        Computed on held-out (e.g. validation) data rather than the
        training data the model was fit on, since permutation importance
        measured on training data can overstate a feature the model has
        overfit to.

        Args:
            pipeline: A pipeline fitted by `train`.
            features: Raw features to permute — should be a split the
                pipeline was NOT trained on.
            target: True labels matching `features`.

        Returns:
            List of `{"feature", "importance"}` dicts, sorted by descending
            importance. Ranked over the raw (pre-encoding) feature names —
            permutation shuffles a whole raw column before it reaches the
            encoder, so importances stay interpretable without needing to
            re-aggregate one-hot-expanded columns.
        """
        result = permutation_importance(
            pipeline,
            features,
            target,
            n_repeats=10,
            random_state=self._random_seed,
            scoring="f1_macro",
        )
        ranked = sorted(
            zip(features.columns, result.importances_mean),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [{"feature": name, "importance": float(value)} for name, value in ranked]
