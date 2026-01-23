"""Categorical metadata encoding for conditioning target predictions."""

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


class CategoricalEncoder:
    """Encodes categorical metadata columns to integer indices.

    Index 0 is reserved for unknown/missing values across all columns.
    Known categories are mapped to indices 1 through N where N is the
    number of unique categories seen during fit.
    """

    def __init__(self) -> None:
        """Initialize encoder."""
        self._mappings: Optional[dict[str, dict[str, int]]] = None
        self._column_names: Optional[list[str]] = None

    def _check_fitted(self) -> None:
        """Raise if encoder is not fitted."""
        if not self.is_fitted:
            raise RuntimeError("Encoder has not been fit. Call fit() first.")

    def fit(
        self,
        metadata: pd.DataFrame,
        columns: list[str],
    ) -> "CategoricalEncoder":
        """Learn category mappings from training data.

        Args:
            metadata: DataFrame containing categorical columns.
            columns: Column names to encode.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If required column missing from metadata.
        """
        col_names = columns

        available_cols = list(metadata.columns)
        for col in col_names:
            if col not in available_cols:
                raise ValueError(f"Column '{col}' not found in metadata. Available columns: {available_cols}")

        self._mappings = {}
        self._column_names = list(col_names)

        for col in col_names:
            unique_values = metadata[col].dropna().unique()
            str_values = sorted(str(v) for v in unique_values)
            self._mappings[col] = {v: i + 1 for i, v in enumerate(str_values)}

        return self

    def transform(
        self,
        metadata: pd.DataFrame,
        sample_ids: Optional[list[str]] = None,
    ) -> dict[str, np.ndarray]:
        """Transform categorical values to integer indices.

        Unknown categories (not seen during fit) are mapped to index 0.
        Missing values (NaN, None) are also mapped to index 0.

        Args:
            metadata: DataFrame containing categorical columns.
            sample_ids: Optional list of sample IDs to extract. If None,
                uses all rows in order. If provided, metadata must have
                'sample_id' column or index.

        Returns:
            Dictionary mapping column name to numpy array of indices.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        self._check_fitted()

        if sample_ids is not None:
            if "sample_id" in metadata.columns:
                metadata = metadata.set_index("sample_id")
            elif metadata.index.name != "sample_id":
                metadata = metadata.copy()
                metadata.index.name = "sample_id"
            metadata = metadata.loc[sample_ids]

        result: dict[str, np.ndarray] = {}
        for col in self._column_names:  # type: ignore[union-attr]
            mapping = self._mappings[col]  # type: ignore[index]
            values = metadata[col]
            indices = np.array(
                [mapping.get(str(v), 0) if pd.notna(v) else 0 for v in values],
                dtype=np.int64,
            )
            result[col] = indices

        return result

    def fit_transform(
        self,
        metadata: pd.DataFrame,
        columns: list[str],
        sample_ids: Optional[list[str]] = None,
    ) -> dict[str, np.ndarray]:
        """Fit encoder and transform data in one step.

        Args:
            metadata: DataFrame containing categorical columns.
            columns: Column names to encode.
            sample_ids: Sample IDs to extract (passed to transform).

        Returns:
            Dictionary mapping column name to numpy array of indices.
        """
        self.fit(metadata, columns)
        return self.transform(metadata, sample_ids=sample_ids)

    def get_cardinalities(self) -> dict[str, int]:
        """Get cardinality (number of categories + 1 for unknown) per column.

        The cardinality includes the reserved index 0 for unknown/missing,
        so for a column with N unique values, cardinality is N + 1.

        Returns:
            Dictionary mapping column name to cardinality.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        self._check_fitted()
        return {col: len(mapping) + 1 for col, mapping in self._mappings.items()}  # type: ignore[union-attr]

    def get_mappings(self) -> dict[str, dict[str, int]]:
        """Get category-to-index mappings for all columns.

        Returns:
            Dictionary mapping column name to category-to-index dict.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        self._check_fitted()
        return {col: dict(mapping) for col, mapping in self._mappings.items()}  # type: ignore[union-attr]

    def get_reverse_mappings(self) -> dict[str, dict[int, str]]:
        """Get index-to-category mappings for all columns.

        Returns:
            Dictionary mapping column name to index-to-category dict.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        self._check_fitted()
        return {
            col: {idx: category for category, idx in mapping.items()}
            for col, mapping in self._mappings.items()  # type: ignore[union-attr]
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save encoder state to JSON file.

        Args:
            path: File path for saving encoder state.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        self._check_fitted()
        state = self.to_dict()
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CategoricalEncoder":
        """Load encoder from JSON file.

        Args:
            path: File path to load encoder from.

        Returns:
            CategoricalEncoder with restored state.
        """
        with open(path) as f:
            state = json.load(f)
        return cls.from_dict(state)

    @property
    def is_fitted(self) -> bool:
        """Whether the encoder has been fit on data."""
        return self._mappings is not None

    @property
    def column_names(self) -> list[str]:
        """List of column names this encoder handles.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        self._check_fitted()
        return list(self._column_names)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        """Convert encoder state to dictionary for checkpoint serialization.

        Returns:
            Dictionary containing encoder state.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        self._check_fitted()
        return {
            "mappings": self._mappings,
            "column_names": self._column_names,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CategoricalEncoder":
        """Restore encoder from dictionary state.

        Args:
            data: Dictionary containing encoder state from to_dict().

        Returns:
            CategoricalEncoder with restored state.
        """
        encoder = cls()
        encoder._mappings = data["mappings"]
        encoder._column_names = data["column_names"]
        return encoder
