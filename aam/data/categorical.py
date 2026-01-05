"""Categorical metadata schema and encoding for conditioning target predictions."""

from dataclasses import dataclass, field
from typing import Optional


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
