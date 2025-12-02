"""Sequence encoder with UniFrac prediction head."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from aam.models.sample_sequence_encoder import SampleSequenceEncoder
from aam.models.attention_pooling import AttentionPooling
from aam.models.transformer import TransformerEncoder


class SequenceEncoder(nn.Module):
    """Encoder that adds prediction head for UniFrac distance prediction."""

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
        encoder_num_layers: int = 2,
        encoder_num_heads: int = 4,
        encoder_intermediate_size: int = None,
        encoder_dropout: float = 0.1,
        encoder_activation: str = "gelu",
        base_output_dim: Optional[int] = None,
        encoder_type: str = "unifrac",
        predict_nucleotides: bool = False,
    ):
        """Initialize SequenceEncoder.

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
            encoder_num_layers: Number of transformer layers for encoder transformer
            encoder_num_heads: Number of attention heads for encoder transformer
            encoder_intermediate_size: FFN intermediate size for encoder transformer
            encoder_dropout: Dropout rate for encoder transformer
            encoder_activation: Activation function for encoder transformer ('gelu' or 'relu')
            base_output_dim: Output dimension for base prediction (None = use embedding_dim)
            encoder_type: Type of encoder ('unifrac', 'taxonomy', 'faith_pd', 'combined')
            predict_nucleotides: Whether to include nucleotide prediction head
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.base_output_dim = base_output_dim if base_output_dim is not None else embedding_dim
        self.encoder_type = encoder_type
        self.predict_nucleotides = predict_nucleotides
        
        self.sample_encoder = SampleSequenceEncoder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_bp=max_bp,
            token_limit=token_limit,
            asv_num_layers=asv_num_layers,
            asv_num_heads=asv_num_heads,
            asv_intermediate_size=asv_intermediate_size,
            asv_dropout=asv_dropout,
            asv_activation=asv_activation,
            sample_num_layers=sample_num_layers,
            sample_num_heads=sample_num_heads,
            sample_intermediate_size=sample_intermediate_size,
            sample_dropout=sample_dropout,
            sample_activation=sample_activation,
            predict_nucleotides=predict_nucleotides,
        )
        
        if encoder_intermediate_size is None:
            encoder_intermediate_size = 4 * embedding_dim
        
        self.encoder_transformer = TransformerEncoder(
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            hidden_dim=embedding_dim,
            intermediate_size=encoder_intermediate_size,
            dropout=encoder_dropout,
            activation=encoder_activation,
        )
        
        self.attention_pooling = AttentionPooling(hidden_dim=embedding_dim)
        
        if encoder_type == "combined":
            self.uni_ff = nn.Linear(embedding_dim, 2)
            self.faith_ff = nn.Linear(embedding_dim, 1)
            self.tax_ff = nn.Linear(embedding_dim, 7)
        else:
            self.output_head = nn.Linear(embedding_dim, self.base_output_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        return_nucleotides: bool = False,
    ) -> Union[Dict[str, torch.Tensor], Tuple]:
        """Forward pass.

        Args:
            tokens: Input tokens [batch_size, num_asvs, seq_len]
            return_nucleotides: Whether to return nucleotide predictions

        Returns:
            Dictionary with keys:
                - 'base_prediction': [batch_size, base_output_dim] or [batch_size, embedding_dim]
                - 'sample_embeddings': [batch_size, num_asvs, embedding_dim]
                - 'nuc_predictions': [batch_size, num_asvs, seq_len, vocab_size] (if return_nucleotides=True)
            Or tuple if encoder_type='combined': (unifrac_pred, faith_pred, tax_pred)
        """
        asv_mask = (tokens.sum(dim=-1) > 0).long()
        
        if self.predict_nucleotides and return_nucleotides:
            sample_embeddings, nuc_predictions = self.sample_encoder(
                tokens, return_nucleotides=True
            )
        else:
            sample_embeddings = self.sample_encoder(tokens, return_nucleotides=False)
            nuc_predictions = None
        
        encoder_embeddings = self.encoder_transformer(sample_embeddings, mask=asv_mask)
        pooled_embeddings = self.attention_pooling(encoder_embeddings, mask=asv_mask)
        
        if self.encoder_type == "combined":
            unifrac_pred = self.uni_ff(pooled_embeddings)
            faith_pred = self.faith_ff(pooled_embeddings)
            tax_pred = self.tax_ff(pooled_embeddings)
            
            result = {
                "unifrac_pred": unifrac_pred,
                "faith_pred": faith_pred,
                "tax_pred": tax_pred,
                "sample_embeddings": sample_embeddings,
            }
            
            if return_nucleotides and nuc_predictions is not None:
                result["nuc_predictions"] = nuc_predictions
            
            return result
        else:
            base_prediction = self.output_head(pooled_embeddings)
            
            result = {
                "base_prediction": base_prediction,
                "sample_embeddings": sample_embeddings,
            }
            
            if return_nucleotides and nuc_predictions is not None:
                result["nuc_predictions"] = nuc_predictions
            
            return result
