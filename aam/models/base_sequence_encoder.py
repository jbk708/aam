"""Base sequence encoder for processing ASV embeddings at the sample level."""

import torch
import torch.nn as nn

from aam.models.asv_encoder import ASVEncoder
from aam.models.position_embedding import PositionEmbedding
from aam.models.transformer import TransformerEncoder


class BaseSequenceEncoder(nn.Module):
    """Encoder that processes ASV embeddings at the sample level."""

    def __init__(
        self,
        vocab_size: int = 5,
        embedding_dim: int = 128,
        max_bp: int = 150,
        token_limit: int = 1024,
        asv_num_layers: int = 2,
        asv_num_heads: int = 4,
        asv_intermediate_size: int = None,
        asv_dropout: float = 0.1,
        asv_activation: str = "gelu",
        sample_num_layers: int = 2,
        sample_num_heads: int = 4,
        sample_intermediate_size: int = None,
        sample_dropout: float = 0.1,
        sample_activation: str = "gelu",
        predict_nucleotides: bool = False,
    ):
        """Initialize BaseSequenceEncoder.

        Args:
            vocab_size: Vocabulary size (default: 5 for pad, A, C, G, T)
            embedding_dim: Embedding dimension
            max_bp: Maximum sequence length (base pairs)
            token_limit: Maximum number of ASVs per sample
            asv_num_layers: Number of transformer layers for ASV encoder
            asv_num_heads: Number of attention heads for ASV encoder
            asv_intermediate_size: FFN intermediate size for ASV encoder
            asv_dropout: Dropout rate for ASV encoder
            asv_activation: Activation function for ASV encoder ('gelu' or 'relu')
            sample_num_layers: Number of transformer layers for sample-level transformer
            sample_num_heads: Number of attention heads for sample-level transformer
            sample_intermediate_size: FFN intermediate size for sample-level transformer
            sample_dropout: Dropout rate for sample-level transformer
            sample_activation: Activation function for sample-level transformer ('gelu' or 'relu')
            predict_nucleotides: Whether to include nucleotide prediction head
        """
        super().__init__()
        pass

    def forward(
        self, tokens: torch.Tensor, return_nucleotides: bool = False
    ):
        """Forward pass.

        Args:
            tokens: Input tokens [batch_size, num_asvs, seq_len]
            return_nucleotides: Whether to return nucleotide predictions

        Returns:
            If return_nucleotides=False: Base embeddings [batch_size, num_asvs, embedding_dim]
            If return_nucleotides=True: Tuple of (base_embeddings, nucleotide_predictions)
                where nucleotide_predictions is [batch_size, num_asvs, seq_len, vocab_size]
        """
        pass
