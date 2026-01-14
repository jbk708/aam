"""Attention pooling layer for sequence embeddings."""

from typing import Optional

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

    def forward(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Input embeddings [batch_size, seq_len, hidden_dim]
            mask: Optional mask [batch_size, seq_len] where 1 is valid, 0 is padding

        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = embeddings.shape

        # Check for NaN in embeddings before processing
        if torch.any(torch.isnan(embeddings)):
            import sys

            print("ERROR: NaN in embeddings before attention pooling", file=sys.stderr, flush=True)
            print(f"embeddings shape={embeddings.shape}", file=sys.stderr, flush=True)
            raise ValueError("NaN values found in embeddings before attention pooling")

        scores = self.query(embeddings)
        scores = scores.squeeze(-1)
        scores = scores / (hidden_dim**0.5)

        # Initialize all_padding outside the if block to ensure it's available later
        all_padding = None
        if mask is not None:
            # Handle all-padding sequences (all positions masked)
            # softmax(all -inf) = NaN, so we need special handling
            mask_sum = mask.sum(dim=-1, keepdim=True)  # [batch_size, 1]
            all_padding = mask_sum == 0  # [batch_size, 1]

            # For sequences with valid positions, mask padding with -inf
            # For all-padding sequences, set scores to 0 (will give uniform softmax)
            if all_padding.any():
                # Expand all_padding for proper broadcasting
                all_padding_expanded = all_padding.expand(-1, seq_len)
                # Set scores to 0 for all-padding sequences (prevents NaN from softmax(all -inf))
                scores = scores.masked_fill(all_padding_expanded, 0.0)
                # For sequences with valid positions, mask padding with -inf
                valid_mask_expanded = (~all_padding).expand(-1, seq_len)
                scores = scores.masked_fill(valid_mask_expanded & (mask == 0), float("-inf"))
            else:
                # No all-padding sequences, normal masking
                scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)

        if mask is not None:
            # Handle all-padding sequences: set uniform weights and skip mask/normalization
            if all_padding is not None and all_padding.any():
                # Expand all_padding to match attention_weights shape [batch_size, seq_len]
                all_padding_expanded = all_padding.expand(-1, seq_len)
                # For all-padding sequences, set uniform attention weights (1/seq_len for each position)
                uniform_weights = torch.full_like(attention_weights, 1.0 / seq_len)
                # Store uniform weights for all-padding sequences
                attention_weights_all_padding = uniform_weights
            else:
                attention_weights_all_padding = None
                all_padding_expanded = None

            # Apply mask to zero out padding positions for sequences with valid positions
            attention_weights = attention_weights * mask

            # Normalize attention weights for sequences with valid positions
            attention_weights_sum = attention_weights.sum(dim=-1, keepdim=True)
            attention_weights = attention_weights / (attention_weights_sum + 1e-8)

            # For all-padding sequences, replace with uniform weights (skip mask and normalization)
            if all_padding_expanded is not None:
                assert attention_weights_all_padding is not None  # Set together with all_padding_expanded
                attention_weights = torch.where(all_padding_expanded, attention_weights_all_padding, attention_weights)

        # Check for NaN in attention_weights before pooling
        if torch.any(torch.isnan(attention_weights)):
            import sys

            print("ERROR: NaN in attention_weights before pooling", file=sys.stderr, flush=True)
            print(f"attention_weights shape={attention_weights.shape}", file=sys.stderr, flush=True)
            if mask is not None:
                print(f"mask sum per sample: {mask.sum(dim=-1)}", file=sys.stderr, flush=True)
                print(f"all_padding: {all_padding if all_padding is not None else 'N/A'}", file=sys.stderr, flush=True)
                if all_padding is not None:
                    print(
                        f"attention_weights sum for all-padding: {attention_weights[all_padding.squeeze()].sum(dim=-1) if all_padding.any() else 'N/A'}",
                        file=sys.stderr,
                        flush=True,
                    )
            raise ValueError("NaN values found in attention_weights before pooling")

        pooled = torch.sum(embeddings * attention_weights.unsqueeze(-1), dim=1)

        # Check for NaN before LayerNorm (helps identify where NaN originates)
        if torch.any(torch.isnan(pooled)):
            import sys

            print("ERROR: NaN in pooled embeddings before LayerNorm", file=sys.stderr, flush=True)
            print(f"pooled shape={pooled.shape}, embeddings shape={embeddings.shape}", file=sys.stderr, flush=True)
            if mask is not None:
                print(f"mask sum per sample: {mask.sum(dim=-1)}", file=sys.stderr, flush=True)
                print(f"all_padding: {all_padding if all_padding is not None else 'N/A'}", file=sys.stderr, flush=True)
            raise ValueError("NaN values found in pooled embeddings before LayerNorm")

        pooled = self.norm(pooled)

        # Final check for NaN after LayerNorm
        if torch.any(torch.isnan(pooled)):
            import sys

            print("ERROR: NaN in pooled embeddings after LayerNorm", file=sys.stderr, flush=True)
            print(f"pooled shape={pooled.shape}, embeddings shape={embeddings.shape}", file=sys.stderr, flush=True)
            if mask is not None:
                print(f"mask sum per sample: {mask.sum(dim=-1)}", file=sys.stderr, flush=True)
            raise ValueError("NaN values found in pooled embeddings after LayerNorm")

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
