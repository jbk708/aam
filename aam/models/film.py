"""FiLM (Feature-wise Linear Modulation) layers for categorical conditioning.

FiLM allows categorical embeddings to modulate intermediate representations via
learned scale (γ) and shift (β) parameters: h_out = γ * h + β

Reference: Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer" (2018)
"""

from typing import List, Optional

import torch
import torch.nn as nn


class FiLMGenerator(nn.Module):
    """Generates FiLM parameters (γ, β) from categorical embeddings.

    Initialized to produce identity transform (gamma=1, beta=0) when given
    zero input, allowing gradual learning of modulation.
    """

    def __init__(self, categorical_dim: int, hidden_dim: int) -> None:
        """Initialize FiLMGenerator.

        Args:
            categorical_dim: Dimension of categorical embedding input.
            hidden_dim: Dimension of hidden layer to modulate.
        """
        super().__init__()
        self.gamma_proj = nn.Linear(categorical_dim, hidden_dim)
        self.beta_proj = nn.Linear(categorical_dim, hidden_dim)

        # Initialize to identity transform: gamma=1, beta=0 for zero input
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, categorical_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate γ and β from categorical embedding.

        Args:
            categorical_emb: Categorical embeddings [batch_size, categorical_dim].

        Returns:
            Tuple of (gamma, beta) each [batch_size, hidden_dim].
        """
        gamma = self.gamma_proj(categorical_emb)
        beta = self.beta_proj(categorical_emb)
        return gamma, beta


class FiLMLayer(nn.Module):
    """MLP layer with FiLM conditioning.

    Applies: output = dropout(relu(gamma * linear(x) + beta))
    """

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
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.film = FiLMGenerator(categorical_dim, out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

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
        h = self.linear(x)
        gamma, beta = self.film(categorical_emb)
        h = gamma * h + beta  # FiLM modulation
        h = self.activation(h)
        h = self.dropout(h)
        return h


class FiLMTargetHead(nn.Module):
    """MLP regression head with FiLM conditioning at each layer.

    Structure: [FiLMLayer, FiLMLayer, ..., Linear]
    FiLM modulation is applied at each hidden layer but not the output layer.
    """

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
        super().__init__()

        self.categorical_dim = categorical_dim
        self.film_layers = nn.ModuleList()

        current_dim = in_dim
        for hidden_dim in hidden_dims:
            self.film_layers.append(
                FiLMLayer(
                    in_dim=current_dim,
                    out_dim=hidden_dim,
                    categorical_dim=categorical_dim,
                    dropout=dropout,
                )
            )
            current_dim = hidden_dim

        self.output_layer = nn.Linear(current_dim, out_dim)

        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(
        self,
        x: torch.Tensor,
        categorical_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through FiLM-conditioned MLP.

        Args:
            x: Input tensor [batch_size, in_dim].
            categorical_emb: Categorical embeddings [batch_size, categorical_dim].
                If None, uses zero tensor (identity transform).

        Returns:
            Output tensor [batch_size, out_dim].
        """
        if categorical_emb is None:
            # Use zero tensor for identity FiLM transform
            categorical_emb = torch.zeros(
                x.size(0), self.categorical_dim, device=x.device, dtype=x.dtype
            )

        for film_layer in self.film_layers:
            x = film_layer(x, categorical_emb)

        return self.output_layer(x)
