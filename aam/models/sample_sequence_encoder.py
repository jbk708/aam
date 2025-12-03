"""Sample sequence encoder for processing ASV embeddings at the sample level."""

import torch
import torch.nn as nn
from typing import Optional

from aam.models.asv_encoder import ASVEncoder
from aam.models.position_embedding import PositionEmbedding
from aam.models.transformer import TransformerEncoder


class SampleSequenceEncoder(nn.Module):
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
        asv_chunk_size: Optional[int] = None,
    ):
        """Initialize SampleSequenceEncoder.

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
            asv_chunk_size: Process ASVs in chunks of this size to reduce memory (None = process all)
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.token_limit = token_limit
        self.predict_nucleotides = predict_nucleotides
        
        self.asv_encoder = ASVEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_bp=max_bp,
            num_layers=asv_num_layers,
            num_heads=asv_num_heads,
            intermediate_size=asv_intermediate_size,
            dropout=asv_dropout,
            activation=asv_activation,
            predict_nucleotides=predict_nucleotides,
            asv_chunk_size=asv_chunk_size,
        )
        
        self.sample_position_embedding = PositionEmbedding(
            max_length=token_limit + 5,
            hidden_dim=embedding_dim,
        )
        
        if sample_intermediate_size is None:
            sample_intermediate_size = 4 * embedding_dim
        
        self.sample_transformer = TransformerEncoder(
            num_layers=sample_num_layers,
            num_heads=sample_num_heads,
            hidden_dim=embedding_dim,
            intermediate_size=sample_intermediate_size,
            dropout=sample_dropout,
            activation=sample_activation,
        )

    def forward(
        self, tokens: torch.Tensor, return_nucleotides: bool = False
    ):
        """Forward pass.

        Args:
            tokens: Input tokens [batch_size, num_asvs, seq_len]
            return_nucleotides: Whether to return nucleotide predictions

        Returns:
            If return_nucleotides=False: Sample embeddings [batch_size, num_asvs, embedding_dim]
            If return_nucleotides=True: Tuple of (sample_embeddings, nucleotide_predictions)
                where nucleotide_predictions is [batch_size, num_asvs, seq_len, vocab_size]
        """
        asv_mask = (tokens.sum(dim=-1) > 0).long()
        
        if self.predict_nucleotides and return_nucleotides:
            asv_embeddings, nucleotide_predictions = self.asv_encoder(
                tokens, return_nucleotides=True
            )
        else:
            asv_embeddings = self.asv_encoder(tokens, return_nucleotides=False)
            nucleotide_predictions = None
        
        asv_mask_expanded = asv_mask.unsqueeze(-1).float()
        asv_embeddings = torch.where(
            torch.isnan(asv_embeddings),
            torch.zeros_like(asv_embeddings),
            asv_embeddings
        )
        asv_embeddings = asv_embeddings * asv_mask_expanded
        
        asv_embeddings = self.sample_position_embedding(asv_embeddings)
        sample_embeddings = self.sample_transformer(asv_embeddings, mask=asv_mask)
        
        sample_embeddings = torch.where(
            torch.isnan(sample_embeddings),
            torch.zeros_like(sample_embeddings),
            sample_embeddings
        )
        sample_embeddings = sample_embeddings * asv_mask_expanded
        
        if return_nucleotides and nucleotide_predictions is not None:
            return sample_embeddings, nucleotide_predictions
        else:
            return sample_embeddings
