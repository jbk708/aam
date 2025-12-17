"""Transformer encoder layer for processing sequences with self-attention."""

from contextlib import contextmanager
from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

# Type alias for attention implementation options
AttnImplementation = Literal["sdpa", "flash", "mem_efficient", "math"]


@contextmanager
def sdpa_kernel_context(attn_implementation: Optional[AttnImplementation]):
    """Context manager to configure SDPA backend.

    Args:
        attn_implementation: Which SDPA backend to use:
            - "sdpa" or None: Use PyTorch's default SDPA (auto-selects best backend)
            - "flash": Force Flash Attention (requires compatible GPU, falls back to math)
            - "mem_efficient": Force memory-efficient attention (falls back to math)
            - "math": Force standard math implementation (always available)

    Note:
        Flash and mem_efficient backends require CUDA. On CPU or incompatible GPUs,
        they will fall back to the math implementation.
    """
    if attn_implementation is None or attn_implementation == "sdpa":
        # Use default SDPA behavior (auto-select best backend)
        yield
        return

    # For specific backends, disable others and enable the requested one
    # Always keep math enabled as fallback for CPU or when preferred backend unavailable
    original_flash = torch.backends.cuda.flash_sdp_enabled()
    original_mem_eff = torch.backends.cuda.mem_efficient_sdp_enabled()
    original_math = torch.backends.cuda.math_sdp_enabled()

    try:
        if attn_implementation == "flash":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        elif attn_implementation == "mem_efficient":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        elif attn_implementation == "math":
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        yield
    finally:
        torch.backends.cuda.enable_flash_sdp(original_flash)
        torch.backends.cuda.enable_mem_efficient_sdp(original_mem_eff)
        torch.backends.cuda.enable_math_sdp(original_math)


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence processing.

    Uses PyTorch's nn.TransformerEncoder which leverages scaled_dot_product_attention
    (SDPA) for optimized attention computation in PyTorch 2.0+.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        intermediate_size: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        gradient_checkpointing: bool = False,
        attn_implementation: Optional[AttnImplementation] = "sdpa",
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
            attn_implementation: Which attention implementation to use:
                - "sdpa" (default): Use scaled_dot_product_attention with auto backend selection
                - "flash": Force Flash Attention (requires compatible GPU, falls back to math)
                - "mem_efficient": Force memory-efficient attention (falls back to math)
                - "math": Force standard math implementation (always available)
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
            enable_nested_tensor=True,
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.gradient_checkpointing = gradient_checkpointing
        self.attn_implementation = attn_implementation

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
            src_key_padding_mask = mask == 0

        with sdpa_kernel_context(self.attn_implementation):
            if self.gradient_checkpointing and self.training:

                def custom_forward(embeddings, src_key_padding_mask):
                    return self.encoder(
                        embeddings, src_key_padding_mask=src_key_padding_mask
                    )

                output = checkpoint(
                    custom_forward,
                    embeddings,
                    src_key_padding_mask,
                    use_reentrant=False,
                )
            else:
                output = self.encoder(
                    embeddings, src_key_padding_mask=src_key_padding_mask
                )

        output = self.norm(output)

        return output
