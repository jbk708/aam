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
        raise NotImplementedError("GMU stub - implementation pending")

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
        raise NotImplementedError("GMU stub - implementation pending")
