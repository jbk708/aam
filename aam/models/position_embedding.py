"""Position embedding layer for sequence embeddings."""

import torch
import torch.nn as nn


class PositionEmbedding(nn.Module):
    """Add learned position information to embeddings."""

    def __init__(self, max_length: int, hidden_dim: int):
        """Initialize PositionEmbedding.

        Args:
            max_length: Maximum sequence length
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.max_length = max_length
        self.hidden_dim = hidden_dim
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Input embeddings [batch_size, seq_len, hidden_dim]

        Returns:
            Embeddings with position information [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = embeddings.shape

        positions = torch.arange(seq_len, device=embeddings.device)
        position_embeds = self.position_embedding(positions)
        position_embeds = position_embeds.unsqueeze(0).expand(batch_size, -1, -1)

        return embeddings + position_embeds
