"""FiLM (Feature-wise Linear Modulation) layers for categorical conditioning.

FiLM allows categorical embeddings to modulate intermediate representations via
learned scale (γ) and shift (β) parameters: h_out = γ * h + β

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)
"""

from typing import List, Optional

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """Generates FiLM parameters (γ, β) from categorical embeddings."""

    def __init__(self, categorical_dim: int, hidden_dim: int) -> None:
        """Initialize FiLMGenerator.

        Args:
            categorical_dim: Dimension of categorical embedding input.
            hidden_dim: Dimension of hidden layer to modulate.
        """
        raise NotImplementedError("FiLMGenerator not yet implemented")

    def forward(self, categorical_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate γ and β from categorical embedding.

        Args:
            categorical_emb: Categorical embeddings [batch_size, categorical_dim].

        Returns:
            Tuple of (gamma, beta) each [batch_size, hidden_dim].
        """
        raise NotImplementedError("FiLMGenerator.forward not yet implemented")


class FiLMLayer(nn.Module):
    """MLP layer with FiLM conditioning."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        categorical_dim: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize FiLMLayer.

        Args:
            in_dim: Input dimension.
            out_dim: Output dimension.
            categorical_dim: Dimension of categorical embedding for FiLM.
            dropout: Dropout rate after activation.
        """
        raise NotImplementedError("FiLMLayer not yet implemented")

    def forward(
        self,
        x: torch.Tensor,
        categorical_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with FiLM modulation.

        Args:
            x: Input tensor [batch_size, in_dim].
            categorical_emb: Categorical embeddings [batch_size, categorical_dim].

        Returns:
            Modulated output [batch_size, out_dim].
        """
        raise NotImplementedError("FiLMLayer.forward not yet implemented")


class FiLMTargetHead(nn.Module):
    """MLP regression head with FiLM conditioning at each layer."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        categorical_dim: int,
        dropout: float = 0.0,
    ) -> None:
        """Initialize FiLMTargetHead.

        Args:
            in_dim: Input dimension (embedding_dim).
            out_dim: Output dimension (number of targets).
            hidden_dims: List of hidden layer dimensions.
            categorical_dim: Dimension of categorical embedding for FiLM.
            dropout: Dropout rate between layers.
        """
        raise NotImplementedError("FiLMTargetHead not yet implemented")

    def forward(
        self,
        x: torch.Tensor,
        categorical_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through FiLM-conditioned MLP.

        Args:
            x: Input tensor [batch_size, in_dim].
            categorical_emb: Categorical embeddings [batch_size, categorical_dim].
                If None, uses identity transform (gamma=1, beta=0).

        Returns:
            Output tensor [batch_size, out_dim].
        """
        raise NotImplementedError("FiLMTargetHead.forward not yet implemented")
