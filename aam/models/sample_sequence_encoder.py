"""Sample sequence encoder for processing ASV embeddings at the sample level."""

import torch
import torch.nn as nn
from typing import Literal, Optional, Tuple, Union

from aam.models.asv_encoder import ASVEncoder
from aam.models.position_embedding import PositionEmbedding
from aam.models.transformer import AttnImplementation, TransformerEncoder

CountEmbeddingMethod = Literal["add", "concat", "film"]


class SampleSequenceEncoder(nn.Module):
    """Encoder that processes ASV embeddings at the sample level."""

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
        predict_nucleotides: bool = False,
        asv_chunk_size: Optional[int] = None,
        gradient_checkpointing: bool = False,
        attn_implementation: Optional[AttnImplementation] = "sdpa",
        mask_ratio: float = 0.0,
        mask_strategy: str = "random",
        count_embedding: bool = False,
        count_embedding_method: CountEmbeddingMethod = "add",
    ):
        """Initialize SampleSequenceEncoder.

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
            predict_nucleotides: Whether to include nucleotide prediction head
            asv_chunk_size: Process ASVs in chunks of this size to reduce memory (None = process all)
            gradient_checkpointing: Whether to use gradient checkpointing to save memory
            attn_implementation: Which SDPA backend to use ('sdpa', 'flash', 'mem_efficient', 'math')
            mask_ratio: Fraction of nucleotide positions to mask for MAE training (0.0 = no masking)
            mask_strategy: Masking strategy ('random' or 'span')
            count_embedding: Whether to incorporate ASV count magnitudes as input features
            count_embedding_method: How to combine count embeddings with sequence embeddings:
                - 'add': asv_emb = seq_emb + count_emb
                - 'concat': asv_emb = proj(cat(seq_emb, count_emb))
                - 'film': asv_emb = seq_emb * scale + shift (FiLM-style modulation)
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.token_limit = token_limit
        self.predict_nucleotides = predict_nucleotides
        self.count_embedding = count_embedding
        self.count_embedding_method = count_embedding_method

        if count_embedding_method not in ("add", "concat", "film"):
            raise ValueError(f"Invalid count_embedding_method: {count_embedding_method}. Must be 'add', 'concat', or 'film'")

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
            gradient_checkpointing=gradient_checkpointing,
            attn_implementation=attn_implementation,
            mask_ratio=mask_ratio,
            mask_strategy=mask_strategy,
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
            gradient_checkpointing=gradient_checkpointing,
            attn_implementation=attn_implementation,
        )

        # Count embedding layers (only created if count_embedding=True)
        if count_embedding:
            if count_embedding_method == "add":
                # Simple linear projection from scalar log-count to embedding_dim
                self.count_embed = nn.Linear(1, embedding_dim)
            elif count_embedding_method == "concat":
                # Project count to embedding_dim, then project concatenated [seq_emb, count_emb] back
                self.count_embed = nn.Linear(1, embedding_dim)
                self.count_proj = nn.Linear(embedding_dim * 2, embedding_dim)
            elif count_embedding_method == "film":
                # FiLM: predict scale and shift from count
                self.count_film = nn.Linear(1, embedding_dim * 2)

    def _apply_count_embedding(self, asv_embeddings: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """Apply count embedding to ASV embeddings.

        Args:
            asv_embeddings: ASV embeddings [batch_size, num_asvs, embedding_dim]
            counts: ASV counts [batch_size, num_asvs, 1] or [batch_size, num_asvs]

        Returns:
            Modified ASV embeddings [batch_size, num_asvs, embedding_dim]
        """
        # Ensure counts have shape [batch_size, num_asvs, 1]
        if counts.dim() == 2:
            counts = counts.unsqueeze(-1)

        # Log-transform counts: log(count + 1) to handle zeros and compress range
        log_counts = torch.log(counts + 1)

        if self.count_embedding_method == "add":
            count_emb = self.count_embed(log_counts)  # [batch, num_asvs, embedding_dim]
            return asv_embeddings + count_emb
        elif self.count_embedding_method == "concat":
            count_emb = self.count_embed(log_counts)  # [batch, num_asvs, embedding_dim]
            combined = torch.cat([asv_embeddings, count_emb], dim=-1)  # [batch, num_asvs, 2*embedding_dim]
            return self.count_proj(combined)  # [batch, num_asvs, embedding_dim]
        elif self.count_embedding_method == "film":
            film_params = self.count_film(log_counts)  # [batch, num_asvs, 2*embedding_dim]
            scale, shift = film_params.chunk(2, dim=-1)  # Each [batch, num_asvs, embedding_dim]
            # FiLM modulation: scale centered around 1, shift around 0
            return asv_embeddings * (1 + scale) + shift
        else:
            return asv_embeddings

    def forward(
        self,
        tokens: torch.Tensor,
        counts: Optional[torch.Tensor] = None,
        return_nucleotides: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """Forward pass.

        Args:
            tokens: Input tokens [batch_size, num_asvs, seq_len]
            counts: Optional ASV counts [batch_size, num_asvs, 1] or [batch_size, num_asvs].
                Required if count_embedding=True.
            return_nucleotides: Whether to return nucleotide predictions

        Returns:
            If return_nucleotides=False: Sample embeddings [batch_size, num_asvs, embedding_dim].
            If return_nucleotides=True: Tuple of (sample_embeddings, nucleotide_predictions, mask_indices)
            where nucleotide_predictions is [batch_size, num_asvs, seq_len, vocab_size]
            and mask_indices is [batch_size, num_asvs, seq_len] boolean tensor (None if not masking).
        """
        if self.count_embedding and counts is None:
            raise ValueError("counts must be provided when count_embedding=True")

        asv_mask = (tokens.sum(dim=-1) > 0).long()

        if self.predict_nucleotides and return_nucleotides:
            asv_embeddings, nucleotide_predictions, mask_indices = self.asv_encoder(tokens, return_nucleotides=True)
        else:
            asv_embeddings = self.asv_encoder(tokens, return_nucleotides=False)
            nucleotide_predictions = None
            mask_indices = None

        asv_mask_expanded = asv_mask.unsqueeze(-1).float()
        asv_embeddings = torch.where(torch.isnan(asv_embeddings), torch.zeros_like(asv_embeddings), asv_embeddings)
        asv_embeddings = asv_embeddings * asv_mask_expanded

        # Apply count embedding if enabled
        if self.count_embedding and counts is not None:
            asv_embeddings = self._apply_count_embedding(asv_embeddings, counts)
            asv_embeddings = asv_embeddings * asv_mask_expanded  # Re-apply mask after count embedding

        asv_embeddings = self.sample_position_embedding(asv_embeddings)
        sample_embeddings = self.sample_transformer(asv_embeddings, mask=asv_mask)

        sample_embeddings = torch.where(torch.isnan(sample_embeddings), torch.zeros_like(sample_embeddings), sample_embeddings)
        sample_embeddings = sample_embeddings * asv_mask_expanded

        if return_nucleotides and nucleotide_predictions is not None:
            return sample_embeddings, nucleotide_predictions, mask_indices
        else:
            return sample_embeddings
