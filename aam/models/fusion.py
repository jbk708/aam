"""Multimodal fusion modules for categorical conditioning.

This module provides fusion strategies for combining sequence representations
with categorical embeddings, enabling learned modality weighting.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union


class GMU(nn.Module):
    """Gated Multimodal Unit for adaptive modality weighting.

    Implements learned gating between sequence and categorical modalities
    using a residual formulation that preserves sequence information.

    Formula:
        h_cat_t = tanh(cat_transform(h_cat))
        z = sigmoid(gate(concat([h_seq, h_cat])))
        output = h_seq + z * h_cat_t

    The residual formulation (h_seq + z * h_cat_t) preserves the scale and
    information in the sequence representation, while allowing categorical
    information to modulate it. This avoids the compression issues of the
    original GMU formula which applies tanh to both branches.

    Reference:
        Inspired by Arevalo et al., "Gated Multimodal Units for Information
        Fusion" (arXiv:1702.01992), adapted with residual connection.
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

        self.cat_transform = nn.Linear(cat_dim, seq_dim)
        self.gate = nn.Linear(seq_dim + cat_dim, seq_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
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
        """Forward pass through GMU with residual connection.

        Uses a residual formulation to preserve sequence information:
            z = sigmoid(gate(concat([h_seq, h_cat])))
            output = h_seq + z * tanh(cat_transform(h_cat))

        The gate controls how much categorical information is added to the
        sequence representation, rather than interpolating between two
        tanh-compressed representations.

        Args:
            h_seq: Pooled sequence representation [batch_size, seq_dim].
            h_cat: Categorical embeddings [batch_size, cat_dim].
            return_gate: If True, return gate values for logging.

        Returns:
            Tuple of:
                - Fused representation [batch_size, seq_dim]
                - Gate values [batch_size, seq_dim] if return_gate=True, else None
        """
        h_cat_t = torch.tanh(self.cat_transform(h_cat))
        z = torch.sigmoid(self.gate(torch.cat([h_seq, h_cat], dim=-1)))
        output = h_seq + z * h_cat_t
        gate_output = z if return_gate else None
        return output, gate_output


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion for position-specific categorical conditioning.

    Enables each sequence position (ASV) to attend differently to categorical
    metadata, allowing position-specific modulation. This contrasts with
    broadcast-based fusion where all positions receive identical conditioning.

    Architecture:
        Sequence [B, S, D] --query--> MultiHeadAttention <--key/value-- Metadata [B, K, E]
                                              |
                                    Position-specific update [B, S, D]

    Reference:
        Cross-attention mechanism adapted from transformer architectures,
        applied to multimodal fusion following FT-Transformer principles.
    """

    def __init__(
        self,
        seq_dim: int,
        cat_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize CrossAttentionFusion.

        Args:
            seq_dim: Dimension of sequence representations (embedding_dim).
            cat_dim: Total dimension of categorical embeddings (total_cat_dim).
            num_heads: Number of attention heads (default: 8).
            dropout: Dropout rate for attention (default: 0.1).
        """
        super().__init__()
        self.seq_dim = seq_dim
        self.cat_dim = cat_dim
        self.num_heads = num_heads

        self.cat_projection = nn.Linear(cat_dim, seq_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=seq_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(seq_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.cat_projection.weight)
        nn.init.zeros_(self.cat_projection.bias)

    def forward(
        self,
        seq_repr: torch.Tensor,
        cat_emb: torch.Tensor,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through cross-attention fusion.

        Each sequence position attends to categorical metadata tokens,
        enabling position-specific conditioning.

        Args:
            seq_repr: Sequence representations [batch_size, seq_len, seq_dim].
            cat_emb: Categorical embeddings [batch_size, cat_dim].
            return_weights: If True, return attention weights for logging.

        Returns:
            If return_weights=False:
                Fused representations [batch_size, seq_len, seq_dim]
            If return_weights=True:
                Tuple of:
                    - Fused representations [batch_size, seq_len, seq_dim]
                    - Attention weights [batch_size, num_heads, seq_len, num_cat_tokens]
        """
        metadata = self.cat_projection(cat_emb).unsqueeze(1)  # [B, 1, seq_dim]

        attn_out, attn_weights = self.cross_attn(
            query=seq_repr,
            key=metadata,
            value=metadata,
            need_weights=return_weights,
            average_attn_weights=False,
        )

        output = self.norm(seq_repr + attn_out)

        if return_weights:
            return output, attn_weights
        return output
