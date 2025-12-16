"""Transformer encoder layer for processing sequences with self-attention."""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence processing."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        intermediate_size: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        gradient_checkpointing: bool = False,
    ):
        """Initialize TransformerEncoder.

        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            hidden_dim: Embedding dimension
            intermediate_size: FFN intermediate size (defaults to 4 * hidden_dim)
            dropout: Dropout rate
            activation: Activation function ('gelu' or 'relu')
            gradient_checkpointing: Whether to use gradient checkpointing to save memory
        """
        super().__init__()
        if intermediate_size is None:
            intermediate_size = 4 * hidden_dim

        if activation == "gelu":
            activation_fn = nn.GELU()
        elif activation == "relu":
            activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'gelu' or 'relu'.")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            dropout=dropout,
            activation=activation_fn,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(
        self, embeddings: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            embeddings: Input embeddings [batch_size, seq_len, hidden_dim]
            mask: Optional mask [batch_size, seq_len] where 1 is valid, 0 is padding

        Returns:
            Processed embeddings [batch_size, seq_len, hidden_dim]
        """
        src_key_padding_mask = None
        if mask is not None:
            src_key_padding_mask = (mask == 0)

        if self.gradient_checkpointing and self.training:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            
            output = checkpoint(
                create_custom_forward(self.encoder),
                embeddings,
                src_key_padding_mask,
                use_reentrant=False,
            )
        else:
            output = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        output = self.norm(output)

        return output
