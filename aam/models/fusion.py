"""Multimodal fusion modules for categorical conditioning.

This module provides fusion strategies for combining sequence representations
with categorical embeddings, enabling learned modality weighting.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class GMU(nn.Module):
    """Gated Multimodal Unit for adaptive modality weighting.

    Implements learned gating between sequence and categorical modalities.
    The gate learns to weight the contribution of each modality based on
    the input representations.

    Formula:
        h_seq_t = tanh(seq_transform(h_seq))
        h_cat_t = tanh(cat_transform(h_cat))
        z = sigmoid(gate(concat([h_seq, h_cat])))
        output = z * h_seq_t + (1 - z) * h_cat_t

    Reference:
        Arevalo et al., "Gated Multimodal Units for Information Fusion"
        (arXiv:1702.01992)
    """

    def __init__(self, seq_dim: int, cat_dim: int) -> None:
        """Initialize GMU.

        Args:
            seq_dim: Dimension of sequence representation (embedding_dim).
            cat_dim: Dimension of categorical embeddings (total_cat_dim).
        """
        super().__init__()
        self.seq_dim = seq_dim
        self.cat_dim = cat_dim

        self.seq_transform = nn.Linear(seq_dim, seq_dim)
        self.cat_transform = nn.Linear(cat_dim, seq_dim)
        self.gate = nn.Linear(seq_dim + cat_dim, seq_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.seq_transform.weight)
        nn.init.zeros_(self.seq_transform.bias)
        nn.init.xavier_uniform_(self.cat_transform.weight)
        nn.init.zeros_(self.cat_transform.bias)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        h_seq: torch.Tensor,
        h_cat: torch.Tensor,
        return_gate: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through GMU.

        Args:
            h_seq: Pooled sequence representation [batch_size, seq_dim].
            h_cat: Categorical embeddings [batch_size, cat_dim].
            return_gate: If True, return gate values for logging.

        Returns:
            Tuple of:
                - Fused representation [batch_size, seq_dim]
                - Gate values [batch_size, seq_dim] if return_gate=True, else None
        """
        h_seq_t = torch.tanh(self.seq_transform(h_seq))
        h_cat_t = torch.tanh(self.cat_transform(h_cat))
        z = torch.sigmoid(self.gate(torch.cat([h_seq, h_cat], dim=-1)))
        output = z * h_seq_t + (1 - z) * h_cat_t
        gate_output = z if return_gate else None
        return output, gate_output
