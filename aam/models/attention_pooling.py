"""Attention pooling layer for sequence embeddings."""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Pool sequence-level embeddings using learned attention weights."""

    def __init__(self, hidden_dim: int):
        """Initialize AttentionPooling.

        Args:
            hidden_dim: Hidden dimension size
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Input embeddings [batch_size, seq_len, hidden_dim]
            mask: Optional mask [batch_size, seq_len] where 1 is valid, 0 is padding

        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = embeddings.shape

        scores = self.query(embeddings)
        scores = scores.squeeze(-1)
        scores = scores / (hidden_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
            
            # Handle all-padding sequences (all positions masked)
            # softmax(all -inf) = NaN, so we need special handling
            mask_sum = mask.sum(dim=-1, keepdim=True)  # [batch_size, 1]
            all_padding = (mask_sum == 0)  # [batch_size, 1]
            
            # For all-padding sequences, use uniform attention weights
            # This prevents NaN from softmax(all -inf)
            if all_padding.any():
                # Set scores to 0 for all-padding sequences (will give uniform softmax)
                scores = scores.masked_fill(all_padding, 0.0)

        attention_weights = torch.softmax(scores, dim=-1)

        if mask is not None:
            attention_weights = attention_weights * mask
            attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            # For all-padding sequences, set to zero (will be handled by normalization)
            if all_padding.any():
                attention_weights = attention_weights.masked_fill(all_padding, 0.0)
                # Re-normalize (all-padding sequences will have sum=0, so use uniform)
                weight_sum = attention_weights.sum(dim=-1, keepdim=True)
                attention_weights = attention_weights / (weight_sum + 1e-8)
                # For truly all-padding, use uniform weights
                attention_weights = attention_weights.masked_fill(all_padding, 1.0 / seq_len)

        pooled = torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=1)
        pooled = self.norm(pooled)

        return pooled


def float_mask(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to float mask (1 for nonzero, 0 for zero).

    Args:
        tensor: Input tensor

    Returns:
        Float mask tensor
    """
    return (tensor != 0).float()


def create_mask_from_tokens(tokens: torch.Tensor) -> torch.Tensor:
    """Create mask from token tensor (1 for nonzero tokens, 0 for padding).

    Args:
        tokens: Token tensor [batch_size, seq_len]

    Returns:
        Mask tensor [batch_size, seq_len]
    """
    return (tokens != 0).long()


def apply_mask(embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply mask to embeddings.

    Args:
        embeddings: Embeddings [batch_size, seq_len, hidden_dim]
        mask: Mask [batch_size, seq_len]

    Returns:
        Masked embeddings
    """
    mask = mask.unsqueeze(-1)
    return embeddings * mask
