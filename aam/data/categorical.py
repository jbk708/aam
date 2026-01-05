"""Categorical metadata schema and encoding for conditioning target predictions."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


@dataclass
class CategoricalColumnConfig:
    """Configuration for a single categorical metadata column.

    Args:
        name: Column name in metadata DataFrame.
        cardinality: Number of categories. If None, auto-detected from data.
        embed_dim: Embedding dimension for this column. If None, uses schema default.
        required: If True, raise error when column missing from metadata.
    """

    name: str
    cardinality: Optional[int] = None
    embed_dim: Optional[int] = None
    required: bool = True


@dataclass
class CategoricalSchema:
    """Schema defining categorical columns for model conditioning.

    Index 0 is reserved for unknown/missing categories across all columns.
    Known categories are mapped to indices 1 through cardinality.

    Args:
        columns: List of categorical column configurations.
        default_embed_dim: Default embedding dimension when not specified per-column.
    """

    columns: list[CategoricalColumnConfig] = field(default_factory=list)
    default_embed_dim: int = 16
    _column_lookup: dict[str, CategoricalColumnConfig] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        """Validate schema after initialization."""
        seen_names: set[str] = set()

        for config in self.columns:
            if not config.name:
                raise ValueError("Column name cannot be empty")

            if config.name in seen_names:
                raise ValueError(f"Duplicate column name: {config.name}")
            seen_names.add(config.name)

            if config.cardinality is not None and config.cardinality <= 0:
                raise ValueError(
                    f"Column '{config.name}' has invalid cardinality {config.cardinality}. "
                    "Must be positive or None for auto-detection."
                )

            if config.embed_dim is not None and config.embed_dim <= 0:
                raise ValueError(
                    f"Column '{config.name}' has invalid embed_dim {config.embed_dim}. "
                    "Must be positive or None to use schema default."
                )

        self._column_lookup = {c.name: c for c in self.columns}

    def get_column(self, name: str) -> CategoricalColumnConfig:
        """Get configuration for a column by name.

        Args:
            name: Column name to look up.

        Returns:
            Column configuration.

        Raises:
            KeyError: If column not found in schema.
        """
        if name not in self._column_lookup:
            raise KeyError(f"Column '{name}' not found in schema")
        return self._column_lookup[name]

    def get_embed_dim(self, name: str) -> int:
        """Get effective embedding dimension for a column.

        Uses column-specific embed_dim if set, otherwise schema default.

        Args:
            name: Column name.

        Returns:
            Embedding dimension for the column.
        """
        config = self.get_column(name)
        if config.embed_dim is not None:
            return config.embed_dim
        return self.default_embed_dim

    @property
    def total_embed_dim(self) -> int:
        """Total embedding dimension across all columns."""
        return sum(self.get_embed_dim(c.name) for c in self.columns)

    @property
    def column_names(self) -> list[str]:
        """List of all column names in schema."""
        return [c.name for c in self.columns]

    def validate_metadata_columns(self, available_columns: list[str]) -> None:
        """Validate that required columns exist in metadata.

        Args:
            available_columns: Columns available in metadata DataFrame.

        Raises:
            ValueError: If required column missing from available columns.
        """
        available_set = set(available_columns)
        for config in self.columns:
            if config.required and config.name not in available_set:
                raise ValueError(
                    f"Required categorical column '{config.name}' not found in metadata. "
                    f"Available columns: {sorted(available_columns)}"
                )

    @classmethod
    def from_column_names(
        cls,
        names: list[str],
        default_embed_dim: int = 16,
    ) -> "CategoricalSchema":
        """Create schema from column names with default settings.

        Convenience factory for simple schemas where all columns use
        auto-detected cardinality and default embedding dimension.

        Args:
            names: List of categorical column names.
            default_embed_dim: Embedding dimension for all columns.

        Returns:
            CategoricalSchema with one config per name.
        """
        columns = [CategoricalColumnConfig(name=name) for name in names]
        return cls(columns=columns, default_embed_dim=default_embed_dim)


class CategoricalEncoder:
    """Encodes categorical metadata columns to integer indices.

    Index 0 is reserved for unknown/missing values across all columns.
    Known categories are mapped to indices 1 through N where N is the
    number of unique categories seen during fit.

    Args:
        schema: Optional CategoricalSchema defining column configurations.
            If None, schema is inferred from data during fit.
    """

    def __init__(self, schema: Optional[CategoricalSchema] = None) -> None:
        """Initialize encoder with optional schema."""
        self._schema = schema
        self._mappings: Optional[dict[str, dict[str, int]]] = None
        self._column_names: Optional[list[str]] = None

    def _check_fitted(self) -> None:
        """Raise if encoder is not fitted."""
        if not self.is_fitted:
            raise RuntimeError("Encoder has not been fit. Call fit() first.")

    def fit(
        self,
        metadata: pd.DataFrame,
        columns: Optional[list[str]] = None,
    ) -> "CategoricalEncoder":
        """Learn category mappings from training data.

        Args:
            metadata: DataFrame containing categorical columns.
            columns: Column names to encode. If None, uses schema column names.
                Required if encoder was created without schema.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If no columns specified and no schema provided.
            ValueError: If required column missing from metadata.
        """
        if columns is not None:
            col_names = columns
        elif self._schema is not None:
            col_names = self._schema.column_names
        else:
            raise ValueError(
                "No columns specified. Provide columns argument or create "
                "encoder with a CategoricalSchema."
            )

        available_cols = list(metadata.columns)
        for col in col_names:
            if col not in available_cols:
                raise ValueError(
                    f"Column '{col}' not found in metadata. "
                    f"Available columns: {available_cols}"
                )

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
        columns: Optional[list[str]] = None,
        sample_ids: Optional[list[str]] = None,
    ) -> dict[str, np.ndarray]:
        """Fit encoder and transform data in one step.

        Args:
            metadata: DataFrame containing categorical columns.
            columns: Column names to encode (passed to fit).
            sample_ids: Sample IDs to extract (passed to transform).

        Returns:
            Dictionary mapping column name to numpy array of indices.
        """
        self.fit(metadata, columns=columns)
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
