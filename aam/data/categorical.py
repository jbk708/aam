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

    def __post_init__(self) -> None:
        """Validate schema after initialization."""
        raise NotImplementedError("CAT-1: Implement schema validation")

    def get_column(self, name: str) -> CategoricalColumnConfig:
        """Get configuration for a column by name.

        Args:
            name: Column name to look up.

        Returns:
            Column configuration.

        Raises:
            KeyError: If column not found in schema.
        """
        raise NotImplementedError("CAT-1: Implement column lookup")

    def get_embed_dim(self, name: str) -> int:
        """Get effective embedding dimension for a column.

        Uses column-specific embed_dim if set, otherwise schema default.

        Args:
            name: Column name.

        Returns:
            Embedding dimension for the column.
        """
        raise NotImplementedError("CAT-1: Implement embed_dim resolution")

    @property
    def total_embed_dim(self) -> int:
        """Total embedding dimension across all columns."""
        raise NotImplementedError("CAT-1: Implement total_embed_dim")

    @property
    def column_names(self) -> list[str]:
        """List of all column names in schema."""
        raise NotImplementedError("CAT-1: Implement column_names property")

    def validate_metadata_columns(self, available_columns: list[str]) -> None:
        """Validate that required columns exist in metadata.

        Args:
            available_columns: Columns available in metadata DataFrame.

        Raises:
            ValueError: If required column missing from available columns.
        """
        raise NotImplementedError("CAT-1: Implement metadata column validation")

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
        raise NotImplementedError("CAT-1: Implement from_column_names factory")
