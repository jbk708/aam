"""Main regression model that composes SequenceEncoder as base model."""

import torch
import torch.nn as nn
from typing import Dict, Optional

from aam.models.sequence_encoder import SequenceEncoder
from aam.models.attention_pooling import AttentionPooling
from aam.models.transformer import TransformerEncoder


class SequenceRegressor(nn.Module):
    """Main model for predicting sample-level targets and ASV counts."""

    def __init__(
        self,
        base_model: Optional[SequenceEncoder] = None,
        encoder_type: str = "unifrac",
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
        encoder_num_layers: int = 2,
        encoder_num_heads: int = 4,
        encoder_intermediate_size: int = None,
        encoder_dropout: float = 0.1,
        encoder_activation: str = "gelu",
        base_output_dim: Optional[int] = None,
        count_num_layers: int = 2,
        count_num_heads: int = 4,
        count_intermediate_size: int = None,
        count_dropout: float = 0.1,
        count_activation: str = "gelu",
        target_num_layers: int = 2,
        target_num_heads: int = 4,
        target_intermediate_size: int = None,
        target_dropout: float = 0.1,
        target_activation: str = "gelu",
        out_dim: int = 1,
        is_classifier: bool = False,
        freeze_base: bool = False,
        predict_nucleotides: bool = False,
    ):
        """Initialize SequenceRegressor.

        Args:
            base_model: Optional SequenceEncoder instance (if None, creates one)
            encoder_type: Type of encoder for base model ('unifrac', 'taxonomy', 'faith_pd', 'combined')
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
            encoder_num_layers: Number of transformer layers for encoder transformer
            encoder_num_heads: Number of attention heads for encoder transformer
            encoder_intermediate_size: FFN intermediate size for encoder transformer
            encoder_dropout: Dropout rate for encoder transformer
            encoder_activation: Activation function for encoder transformer ('gelu' or 'relu')
            base_output_dim: Output dimension for base prediction (None = use embedding_dim)
            count_num_layers: Number of transformer layers for count encoder
            count_num_heads: Number of attention heads for count encoder
            count_intermediate_size: FFN intermediate size for count encoder
            count_dropout: Dropout rate for count encoder
            count_activation: Activation function for count encoder ('gelu' or 'relu')
            target_num_layers: Number of transformer layers for target encoder
            target_num_heads: Number of attention heads for target encoder
            target_intermediate_size: FFN intermediate size for target encoder
            target_dropout: Dropout rate for target encoder
            target_activation: Activation function for target encoder ('gelu' or 'relu')
            out_dim: Output dimension for target prediction
            is_classifier: Whether to use classification (log-softmax) or regression
            freeze_base: Whether to freeze base model parameters
            predict_nucleotides: Whether base model should predict nucleotides
        """
        super().__init__()
        pass

    def forward(
        self,
        tokens: torch.Tensor,
        return_nucleotides: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            tokens: Input tokens [batch_size, num_asvs, seq_len]
            return_nucleotides: Whether to return nucleotide predictions

        Returns:
            Dictionary with keys:
                - 'target_prediction': [batch_size, out_dim]
                - 'count_prediction': [batch_size, num_asvs, 1]
                - 'base_embeddings': [batch_size, num_asvs, embedding_dim]
                - 'base_prediction': [batch_size, base_output_dim] (if training)
                - 'nuc_predictions': [batch_size, num_asvs, seq_len, vocab_size] (if return_nucleotides=True)
        """
        pass
