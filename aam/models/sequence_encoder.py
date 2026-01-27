"""Sequence encoder with UniFrac prediction head."""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Union

from aam.models.sample_sequence_encoder import SampleSequenceEncoder
from aam.models.attention_pooling import AttentionPooling
from aam.models.transformer import AttnImplementation, TransformerEncoder


class SequenceEncoder(nn.Module):
    """Encoder that adds prediction head for UniFrac distance prediction."""

    def __init__(
        self,
        vocab_size: int = 7,
        embedding_dim: int = 128,
        max_bp: int = 150,
        token_limit: int = 1024,
        asv_num_layers: int = 2,
        asv_num_heads: int = 4,
        asv_intermediate_size: Optional[int] = None,
        asv_dropout: float = 0.1,
        asv_activation: str = "gelu",
        sample_num_layers: int = 2,
        sample_num_heads: int = 4,
        sample_intermediate_size: Optional[int] = None,
        sample_dropout: float = 0.1,
        sample_activation: str = "gelu",
        encoder_num_layers: int = 2,
        encoder_num_heads: int = 4,
        encoder_intermediate_size: Optional[int] = None,
        encoder_dropout: float = 0.1,
        encoder_activation: str = "gelu",
        base_output_dim: Optional[int] = None,
        encoder_type: str = "unifrac",
        predict_nucleotides: bool = False,
        asv_chunk_size: Optional[int] = None,
        gradient_checkpointing: bool = False,
        attn_implementation: Optional[AttnImplementation] = "sdpa",
        mask_ratio: float = 0.0,
        mask_strategy: str = "random",
    ):
        """Initialize SequenceEncoder.

        Args:
            vocab_size: Vocabulary size (default: 7 for pad, A, C, G, T, START, MASK)
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
            asv_chunk_size: Process ASVs in chunks of this size to reduce memory (None = process all)
            gradient_checkpointing: Whether to use gradient checkpointing to save memory
            attn_implementation: Which SDPA backend to use ('sdpa', 'flash', 'mem_efficient', 'math')
            mask_ratio: Fraction of nucleotide positions to mask for MAE training (0.0 = no masking)
            mask_strategy: Masking strategy ('random' or 'span')
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
            asv_chunk_size=asv_chunk_size,
            gradient_checkpointing=gradient_checkpointing,
            attn_implementation=attn_implementation,
            mask_ratio=mask_ratio,
            mask_strategy=mask_strategy,
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
            gradient_checkpointing=gradient_checkpointing,
            attn_implementation=attn_implementation,
        )

        self.attention_pooling = AttentionPooling(hidden_dim=embedding_dim)

        if encoder_type == "combined":
            self.uni_ff = nn.Linear(embedding_dim, 2)
            # Initialize UniFrac head for sigmoid activation
            # Use Xavier/Glorot initialization for proper weight scaling
            # Bias initialized to shift sigmoid output toward expected range (0.7 for UniFrac)
            nn.init.xavier_uniform_(self.uni_ff.weight, gain=1.0)
            # sigmoid(0.85) ≈ 0.7 (typical UniFrac distance), so initialize bias to ~0.85
            nn.init.constant_(self.uni_ff.bias, 0.85)
            self.faith_ff = nn.Linear(embedding_dim, 1)
            self.tax_ff = nn.Linear(embedding_dim, 7)
        else:
            # For UniFrac, we return embeddings directly and compute distances from them
            # For other encoder types, we still use output_head
            if encoder_type != "unifrac":
                self.output_head = nn.Linear(embedding_dim, self.base_output_dim)
            else:
                # UniFrac: no output_head, return embeddings directly
                self.output_head = None

    def forward(
        self,
        tokens: torch.Tensor,
        return_nucleotides: bool = False,
        return_sample_embeddings: bool = False,
    ) -> Union[Dict[str, Optional[torch.Tensor]], Tuple]:
        """Forward pass.

        Args:
            tokens: Input tokens [batch_size, num_asvs, seq_len]
            return_nucleotides: Whether to return nucleotide predictions
            return_sample_embeddings: Whether to return sample_embeddings in output dict.
                Default False to save memory during training (loss doesn't use them).

        Returns:
            Dictionary with keys:
                - 'base_prediction': [batch_size, base_output_dim] or [batch_size, embedding_dim]
                - 'sample_embeddings': [batch_size, num_asvs, embedding_dim] (only if return_sample_embeddings=True)
                - 'nuc_predictions': [batch_size, num_asvs, seq_len, vocab_size] (if return_nucleotides=True)
                - 'mask_indices': [batch_size, num_asvs, seq_len] boolean (if masking, else None)
            Or tuple if encoder_type='combined': (unifrac_pred, faith_pred, tax_pred)
        """
        asv_mask = (tokens.sum(dim=-1) > 0).long()

        if self.predict_nucleotides and return_nucleotides:
            sample_embeddings, nuc_predictions, mask_indices = self.sample_encoder(tokens, return_nucleotides=True)
        else:
            sample_embeddings = self.sample_encoder(tokens, return_nucleotides=False)
            nuc_predictions = None
            mask_indices = None

        encoder_embeddings = self.encoder_transformer(sample_embeddings, mask=asv_mask)
        pooled_embeddings = self.attention_pooling(encoder_embeddings, mask=asv_mask)

        if self.encoder_type == "combined":
            unifrac_pred = self.uni_ff(pooled_embeddings)
            # Use sigmoid activation for UniFrac predictions (bounded regression)
            # Sigmoid provides natural [0, 1] constraint without hard clipping boundaries
            # Bias is initialized to shift output toward expected range (0.7 for UniFrac)
            unifrac_pred = torch.sigmoid(unifrac_pred)
            faith_pred = self.faith_ff(pooled_embeddings)
            tax_pred = self.tax_ff(pooled_embeddings)

            result: Dict[str, Optional[torch.Tensor]] = {
                "unifrac_pred": unifrac_pred,
                "faith_pred": faith_pred,
                "tax_pred": tax_pred,
            }

            if return_sample_embeddings:
                result["sample_embeddings"] = sample_embeddings

            if return_nucleotides and nuc_predictions is not None:
                result["nuc_predictions"] = nuc_predictions
                result["mask_indices"] = mask_indices

            return result
        else:
            # For UniFrac, return embeddings directly (distances computed in loss function)
            # For other encoder types, use output_head
            if self.encoder_type == "unifrac":
                # Return embeddings directly - distances will be computed from embeddings
                result = {
                    "embeddings": pooled_embeddings,
                }
            else:
                # Non-UniFrac encoders: use output_head
                assert self.output_head is not None  # Set for non-unifrac encoder types
                base_prediction = self.output_head(pooled_embeddings)
                result = {
                    "base_prediction": base_prediction,
                }

            if return_sample_embeddings:
                result["sample_embeddings"] = sample_embeddings

            if return_nucleotides and nuc_predictions is not None:
                result["nuc_predictions"] = nuc_predictions
                result["mask_indices"] = mask_indices

            return result
