"""
Training pipeline entry point.

Orchestrates loading the raw data, splitting it, fitting both models,
evaluating them, and serializing artifacts + metadata to disk. The
`models` table in Postgres is populated from this metadata in a later
phase (see `docs/execution_plan.md` Phase 3) — this script only needs the
filesystem.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from training.helpers.config import TrainingConfig
from training.helpers.data_split import DataSplitter
from training.helpers.evaluation import ModelEvaluator
from training.helpers.preprocessing import FeatureEncoder
from training.train_logistic import LogisticModelTrainer
from training.train_tree import TreeModelTrainer


class TrainingPipeline:
    """Runs the full training pipeline end to end.

    Entry points:
        run: Load data, split, train both models, evaluate, and write
            artifacts + metadata to `config.artifacts_dir`.

    Args:
        config: Shared training configuration.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config
        self._encoder = FeatureEncoder()

    def _load_data(self) -> pd.DataFrame:
        # keep_default_na=False, na_values=[]: the CSV uses the literal
        # string "None" as a real category (e.g. no mental health
        # condition), which pandas would otherwise silently parse as a
        # missing value — see docs/eda_findings.md.
        return pd.read_csv(
            self._config.raw_data_path, keep_default_na=False, na_values=[]
        )

    def _features_and_target(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        feature_columns = self._encoder.feature_names()
        return data[feature_columns], data[self._config.target_column]

    def _save_artifact(self, pipeline: Any, name: str) -> Path:
        self._config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self._config.artifacts_dir / f"{name}.joblib"
        joblib.dump(pipeline, artifact_path)
        return artifact_path

    def _write_metadata(self, metadata: dict[str, Any]) -> Path:
        self._config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self._config.artifacts_dir / "metadata.json"
        with metadata_path.open("w") as f:
            json.dump(metadata, f, indent=2)
        return metadata_path

    def run(self) -> dict[str, Any]:
        """Run the full pipeline and write artifacts + metadata to disk.

        Returns:
            The same metadata dict written to `metadata.json`, containing
            per-model artifact paths, evaluation metrics (on validation and
            test), and feature importances.
        """
        data = self._load_data()
        splitter = DataSplitter(self._config)
        train_data, val_data, test_data = splitter.split(data)

        train_features, train_target = self._features_and_target(train_data)
        val_features, val_target = self._features_and_target(val_data)
        test_features, test_target = self._features_and_target(test_data)

        class_labels = sorted(train_target.unique())
        evaluator = ModelEvaluator(class_labels)

        logistic_trainer = LogisticModelTrainer(self._encoder, self._config.random_seed)
        logistic_pipeline = logistic_trainer.train(train_features, train_target)

        tree_trainer = TreeModelTrainer(self._encoder, self._config.random_seed)
        tree_pipeline = tree_trainer.train(train_features, train_target)

        metadata = {
            "logistic": {
                "artifact_path": str(
                    self._save_artifact(logistic_pipeline, "logistic")
                ),
                "feature_importance": logistic_trainer.feature_importance(
                    logistic_pipeline
                ),
                "validation_metrics": evaluator.evaluate(
                    logistic_pipeline, val_features, val_target
                ),
                "test_metrics": evaluator.evaluate(
                    logistic_pipeline, test_features, test_target
                ),
            },
            "tree": {
                "artifact_path": str(self._save_artifact(tree_pipeline, "tree")),
                "feature_importance": tree_trainer.feature_importance(
                    tree_pipeline, val_features, val_target
                ),
                "validation_metrics": evaluator.evaluate(
                    tree_pipeline, val_features, val_target
                ),
                "test_metrics": evaluator.evaluate(
                    tree_pipeline, test_features, test_target
                ),
            },
        }
        self._write_metadata(metadata)
        return metadata


if __name__ == "__main__":
    result = TrainingPipeline(TrainingConfig()).run()
    print(
        "logistic macro-F1 (test):",
        result["logistic"]["test_metrics"]["macro_f1"],
    )
    print(
        "tree macro-F1 (test):",
        result["tree"]["test_metrics"]["macro_f1"],
    )
