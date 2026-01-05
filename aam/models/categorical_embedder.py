"""Categorical embedding layer for conditioning target predictions."""

from typing import Optional, Union

import torch
import torch.nn as nn

from aam.data.categorical import CategoricalSchema


class CategoricalEmbedder(nn.Module):
    """Embeds categorical features for conditioning target predictions.

    Creates one nn.Embedding per categorical column and concatenates outputs.
    Index 0 is reserved as padding index for unknown/missing categories.

    Args:
        column_cardinalities: Dict mapping column name to number of categories
            (including the reserved index 0 for unknown).
        embed_dim: Embedding dimension per column. Can be int for shared dimension
            or dict mapping column name to dimension.
        dropout: Dropout probability applied to embeddings.
    """

    def __init__(
        self,
        column_cardinalities: dict[str, int],
        embed_dim: Union[int, dict[str, int]] = 16,
        dropout: float = 0.1,
    ) -> None:
        """Initialize CategoricalEmbedder."""
        raise NotImplementedError

    @classmethod
    def from_schema(
        cls,
        schema: CategoricalSchema,
        cardinalities: dict[str, int],
        dropout: float = 0.1,
    ) -> "CategoricalEmbedder":
        """Create embedder from CategoricalSchema.

        Args:
            schema: CategoricalSchema defining column configurations.
            cardinalities: Dict mapping column name to cardinality
                (from CategoricalEncoder.get_cardinalities()).
            dropout: Dropout probability.

        Returns:
            CategoricalEmbedder instance configured from schema.
        """
        raise NotImplementedError

    def forward(
        self,
        categorical_ids: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Embed categorical indices and concatenate.

        Args:
            categorical_ids: Dict mapping column name to index tensor [B].
                Missing columns use padding index 0.

        Returns:
            Concatenated embeddings [B, total_embed_dim].
        """
        raise NotImplementedError

    def broadcast_to_sequence(
        self,
        embeddings: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Broadcast embeddings to sequence length.

        Args:
            embeddings: Categorical embeddings [B, total_embed_dim].
            seq_len: Sequence length to broadcast to.

        Returns:
            Broadcast embeddings [B, seq_len, total_embed_dim].
        """
        raise NotImplementedError

    @property
    def total_embed_dim(self) -> int:
        """Total embedding dimension across all columns."""
        raise NotImplementedError

    @property
    def column_names(self) -> list[str]:
        """List of column names in order."""
        raise NotImplementedError

    @property
    def num_columns(self) -> int:
        """Number of categorical columns."""
        raise NotImplementedError
