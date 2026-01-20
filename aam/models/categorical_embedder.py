"""Categorical embedding layer for conditioning target predictions."""

from typing import Union

import torch
import torch.nn as nn


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
        super().__init__()

        self._column_names = sorted(column_cardinalities.keys())
        self._embed_dims: dict[str, int] = {}

        # Create embedding for each column
        self.embeddings = nn.ModuleDict()
        for col in self._column_names:
            cardinality = column_cardinalities[col]
            if isinstance(embed_dim, dict):
                col_embed_dim = embed_dim[col]
            else:
                col_embed_dim = embed_dim

            self._embed_dims[col] = col_embed_dim

            # padding_idx=0 for unknown/missing categories
            self.embeddings[col] = nn.Embedding(
                num_embeddings=cardinality,
                embedding_dim=col_embed_dim,
                padding_idx=0,
            )

        self.dropout = nn.Dropout(p=dropout)

        # Compute total embedding dimension
        self._total_embed_dim = sum(self._embed_dims.values())

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
        if not self._column_names:
            # No columns configured - return empty tensor
            # Determine batch size from input if available
            if categorical_ids:
                first_tensor = next(iter(categorical_ids.values()))
                batch_size = first_tensor.shape[0]
                device = first_tensor.device
            else:
                batch_size = 0
                device = torch.device("cpu")
            return torch.empty(batch_size, 0, device=device)

        embeddings_list = []
        batch_size = None
        device = None

        for col in self._column_names:
            if col in categorical_ids:
                indices = categorical_ids[col]
                if batch_size is None:
                    batch_size = indices.shape[0]
                    device = indices.device
                emb = self.embeddings[col](indices)  # [B, embed_dim]
            else:
                # Missing column - use padding index 0
                if batch_size is None:
                    # Get batch size from another column
                    for other_col in categorical_ids:
                        batch_size = categorical_ids[other_col].shape[0]
                        device = categorical_ids[other_col].device
                        break
                if batch_size is None:
                    raise ValueError("Cannot determine batch size from empty categorical_ids")

                padding_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
                emb = self.embeddings[col](padding_indices)  # [B, embed_dim]

            embeddings_list.append(emb)

        # Concatenate all embeddings
        concatenated = torch.cat(embeddings_list, dim=-1)  # [B, total_embed_dim]

        # Apply dropout
        return self.dropout(concatenated)

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
        # [B, embed_dim] -> [B, 1, embed_dim] -> [B, seq_len, embed_dim]
        return embeddings.unsqueeze(1).expand(-1, seq_len, -1)

    @property
    def total_embed_dim(self) -> int:
        """Total embedding dimension across all columns."""
        return self._total_embed_dim

    @property
    def column_names(self) -> list[str]:
        """List of column names in order."""
        return list(self._column_names)

    @property
    def num_columns(self) -> int:
        """Number of categorical columns."""
        return len(self._column_names)
