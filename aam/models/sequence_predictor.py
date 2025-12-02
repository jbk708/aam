"""Main prediction model that composes SequenceEncoder as base model."""

import torch
import torch.nn as nn
from typing import Dict, Optional

from aam.models.sequence_encoder import SequenceEncoder
from aam.models.attention_pooling import AttentionPooling
from aam.models.transformer import TransformerEncoder


class SequencePredictor(nn.Module):
    """Main model for predicting sample-level targets and ASV counts.

    Supports both regression and classification tasks. Composes SequenceEncoder
    as base model to enable transfer learning and multi-task learning.
    """

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
        """Initialize SequencePredictor.

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
        
        if base_model is None:
            self.base_model = SequenceEncoder(
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
                encoder_num_layers=encoder_num_layers,
                encoder_num_heads=encoder_num_heads,
                encoder_intermediate_size=encoder_intermediate_size,
                encoder_dropout=encoder_dropout,
                encoder_activation=encoder_activation,
                base_output_dim=base_output_dim,
                encoder_type=encoder_type,
                predict_nucleotides=predict_nucleotides,
            )
            self.embedding_dim = embedding_dim
        else:
            self.base_model = base_model
            self.embedding_dim = base_model.embedding_dim
        
        self.out_dim = out_dim
        self.is_classifier = is_classifier
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        if count_intermediate_size is None:
            count_intermediate_size = 4 * self.embedding_dim
        
        self.count_encoder = TransformerEncoder(
            num_layers=count_num_layers,
            num_heads=count_num_heads,
            hidden_dim=self.embedding_dim,
            intermediate_size=count_intermediate_size,
            dropout=count_dropout,
            activation=count_activation,
        )
        
        self.count_head = nn.Linear(self.embedding_dim, 1)
        
        if target_intermediate_size is None:
            target_intermediate_size = 4 * self.embedding_dim
        
        self.target_encoder = TransformerEncoder(
            num_layers=target_num_layers,
            num_heads=target_num_heads,
            hidden_dim=self.embedding_dim,
            intermediate_size=target_intermediate_size,
            dropout=target_dropout,
            activation=target_activation,
        )
        
        self.target_pooling = AttentionPooling(hidden_dim=self.embedding_dim)
        self.target_head = nn.Linear(self.embedding_dim, out_dim)

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
                - 'base_prediction': [batch_size, base_output_dim] (if return_nucleotides=True)
                - 'nuc_predictions': [batch_size, num_asvs, seq_len, vocab_size] (if return_nucleotides=True)
        """
        asv_mask = (tokens.sum(dim=-1) > 0).long()
        
        base_outputs = self.base_model(tokens, return_nucleotides=return_nucleotides)
        
        base_embeddings = base_outputs["sample_embeddings"]
        base_prediction = base_outputs.get("base_prediction")
        nuc_predictions = base_outputs.get("nuc_predictions")
        
        count_embeddings = self.count_encoder(base_embeddings, mask=asv_mask)
        count_prediction = self.count_head(count_embeddings)
        
        target_embeddings = self.target_encoder(base_embeddings, mask=asv_mask)
        pooled_target = self.target_pooling(target_embeddings, mask=asv_mask)
        target_prediction = self.target_head(pooled_target)
        
        if self.is_classifier:
            target_prediction = nn.functional.log_softmax(target_prediction, dim=-1)
        
        result = {
            "target_prediction": target_prediction,
            "count_prediction": count_prediction,
            "base_embeddings": base_embeddings,
        }
        
        if return_nucleotides and base_prediction is not None:
            result["base_prediction"] = base_prediction
        
        if return_nucleotides and nuc_predictions is not None:
            result["nuc_predictions"] = nuc_predictions
        
        if return_nucleotides and "unifrac_pred" in base_outputs:
            result["unifrac_pred"] = base_outputs["unifrac_pred"]
            result["faith_pred"] = base_outputs["faith_pred"]
            result["tax_pred"] = base_outputs["tax_pred"]
        
        return result
