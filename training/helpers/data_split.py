"""
Stratified train/validation/test split for the employee well-being dataset.

Implements the split logic described in
`docs/architecture_and_design.md`'s "Train / validation / test split"
section: two stratified splits (test off first, then train/validation from
the remainder) so every partition preserves the target's class balance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from training.helpers.config import TrainingConfig


class DataSplitter:
    """Splits a DataFrame into stratified train/validation/test partitions.

    Entry points:
        split: Return the (train, validation, test) DataFrames.

    Args:
        config: Shared training configuration (split ratios, random seed,
            target column).
    """

    def __init__(self, config: TrainingConfig) -> None:
        self._config = config

    def _split_off_test(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        remainder, test = train_test_split(
            data,
            test_size=self._config.test_size,
            stratify=data[self._config.target_column],
            random_state=self._config.random_seed,
        )
        return remainder, test

    def _split_train_val(
        self, remainder: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Validation's share of `remainder` (not of the original data) —
        # remainder is (train_size + val_size) of the original, so scale
        # val_size back up to that reduced denominator.
        val_fraction_of_remainder = self._config.val_size / (
            self._config.train_size + self._config.val_size
        )
        train, val = train_test_split(
            remainder,
            test_size=val_fraction_of_remainder,
            stratify=remainder[self._config.target_column],
            random_state=self._config.random_seed,
        )
        return train, val

    def split(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into stratified train/validation/test partitions.

        Args:
            data: Full raw dataset, including the target column.

        Returns:
            Tuple of (train, validation, test) DataFrames, each stratified
            on `config.target_column` and using `config.random_seed`.
        """
        remainder, test = self._split_off_test(data)
        train, val = self._split_train_val(remainder)
        return train, val, test
