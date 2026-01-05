"""Categorical metadata schema and encoding for conditioning target predictions."""

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def get_cardinalities(self) -> dict[str, int]:
        """Get cardinality (number of categories + 1 for unknown) per column.

        The cardinality includes the reserved index 0 for unknown/missing,
        so for a column with N unique values, cardinality is N + 1.

        Returns:
            Dictionary mapping column name to cardinality.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        raise NotImplementedError

    def get_mappings(self) -> dict[str, dict[str, int]]:
        """Get category-to-index mappings for all columns.

        Returns:
            Dictionary mapping column name to category-to-index dict.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        raise NotImplementedError

    def save(self, path: Union[str, Path]) -> None:
        """Save encoder state to JSON file.

        Args:
            path: File path for saving encoder state.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        raise NotImplementedError

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CategoricalEncoder":
        """Load encoder from JSON file.

        Args:
            path: File path to load encoder from.

        Returns:
            CategoricalEncoder with restored state.
        """
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        """Whether the encoder has been fit on data."""
        raise NotImplementedError

    @property
    def column_names(self) -> list[str]:
        """List of column names this encoder handles.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Convert encoder state to dictionary for checkpoint serialization.

        Returns:
            Dictionary containing encoder state.

        Raises:
            RuntimeError: If encoder has not been fit.
        """
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CategoricalEncoder":
        """Restore encoder from dictionary state.

        Args:
            data: Dictionary containing encoder state from to_dict().

        Returns:
            CategoricalEncoder with restored state.
        """
        raise NotImplementedError
