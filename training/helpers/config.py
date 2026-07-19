"""
Configuration for the model training pipeline.

Centralizes paths, the random seed, and split ratios so every training
script reads the same values instead of hardcoding them independently.
"""

import os
from pathlib import Path


class TrainingConfig:
    """Holds paths and hyperparameters shared across the training pipeline.

    Entry points:
        This class has no methods beyond `__init__` — it is a plain
        settings holder read by `DataSplitter`, `FeatureEncoder`, the model
        trainers, and `TrainingPipeline`.

    Args:
        raw_data_path: Path to the source CSV.
        artifacts_dir: Directory serialized model pipelines are written to.
        random_seed: Seed shared by the split and both models, so results
            are reproducible and the two models are compared on identical
            data.
        train_size: Fraction of rows in the training split.
        val_size: Fraction of rows in the validation split.
        test_size: Fraction of rows in the held-out test split.
        target_column: Name of the target column to predict.
    """

    def __init__(
        self,
        raw_data_path: Path | None = None,
        artifacts_dir: Path | None = None,
        random_seed: int = 42,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        target_column: str = "Satisfaction_with_Remote_Work",
    ) -> None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        self.raw_data_path = raw_data_path or Path(
            os.environ.get(
                "RAW_DATA_PATH",
                repo_root / "raw_data" / "Impact_of_Remote_Work_on_Mental_Health.csv",
            )
        )
        self.artifacts_dir = artifacts_dir or Path(
            os.environ.get("ARTIFACTS_DIR", repo_root / "data" / "models")
        )
        self.random_seed = random_seed
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.target_column = target_column

        if abs(train_size + val_size + test_size - 1.0) > 1e-9:
            raise ValueError("train_size + val_size + test_size must sum to 1.0")
