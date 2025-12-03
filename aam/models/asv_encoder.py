"""ASV-level sequence encoder for processing nucleotide sequences."""

import torch
import torch.nn as nn
from typing import Optional

from aam.models.attention_pooling import AttentionPooling
from aam.models.position_embedding import PositionEmbedding
from aam.models.transformer import TransformerEncoder


class ASVEncoder(nn.Module):
    """Encoder that processes nucleotide sequences at the ASV level."""

    def __init__(
        self,
        vocab_size: int = 5,
        embedding_dim: int = 128,
        max_bp: int = 150,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        predict_nucleotides: bool = False,
        asv_chunk_size: Optional[int] = None,
    ):
        """Initialize ASVEncoder.

        Args:
            vocab_size: Vocabulary size (default: 5 for pad, A, C, G, T)
            embedding_dim: Embedding dimension
            max_bp: Maximum sequence length (base pairs)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            intermediate_size: FFN intermediate size (defaults to 4 * embedding_dim)
            dropout: Dropout rate
            activation: Activation function ('gelu' or 'relu')
            predict_nucleotides: Whether to include nucleotide prediction head
            asv_chunk_size: Process ASVs in chunks of this size (None = process all at once)
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_bp = max_bp
        self.predict_nucleotides = predict_nucleotides
        self.asv_chunk_size = asv_chunk_size
        
        if intermediate_size is None:
            intermediate_size = 4 * embedding_dim
        
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionEmbedding(max_length=max_bp + 1, hidden_dim=embedding_dim)
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=embedding_dim,
            intermediate_size=intermediate_size,
            dropout=dropout,
            activation=activation,
        )
        self.attention_pooling = AttentionPooling(hidden_dim=embedding_dim)
        
        if predict_nucleotides:
            self.nucleotide_head = nn.Linear(embedding_dim, vocab_size)

    def forward(
        self, tokens: torch.Tensor, return_nucleotides: bool = False
    ):
        """Forward pass.

        Args:
            tokens: Input tokens [batch_size, num_asvs, seq_len]
            return_nucleotides: Whether to return nucleotide predictions

        Returns:
            If return_nucleotides=False: ASV embeddings [batch_size, num_asvs, embedding_dim]
            If return_nucleotides=True: Tuple of (embeddings, nucleotide_predictions)
                where nucleotide_predictions is [batch_size, num_asvs, seq_len, vocab_size]
        """
        batch_size, num_asvs, seq_len = tokens.shape
        
        tokens = tokens.long()
        
        if self.asv_chunk_size is not None and num_asvs > self.asv_chunk_size:
            asv_embeddings_list = []
            nucleotide_predictions_list = []
            
            for i in range(0, num_asvs, self.asv_chunk_size):
                end_idx = min(i + self.asv_chunk_size, num_asvs)
                chunk_tokens = tokens[:, i:end_idx, :]
                chunk_num_asvs = chunk_tokens.size(1)
                
                tokens_flat = chunk_tokens.reshape(batch_size * chunk_num_asvs, seq_len)
                mask = (tokens_flat > 0).long()
                
                embeddings = self.token_embedding(tokens_flat)
                embeddings = self.position_embedding(embeddings)
                embeddings = self.transformer(embeddings, mask=mask)
                
                if self.predict_nucleotides and return_nucleotides:
                    nucleotide_logits = self.nucleotide_head(embeddings)
                    chunk_nuc_preds = nucleotide_logits.reshape(batch_size, chunk_num_asvs, seq_len, self.vocab_size)
                    nucleotide_predictions_list.append(chunk_nuc_preds)
                    del nucleotide_logits
                
                pooled_embeddings = self.attention_pooling(embeddings, mask=mask)
                chunk_asv_embeddings = pooled_embeddings.reshape(batch_size, chunk_num_asvs, self.embedding_dim)
                asv_embeddings_list.append(chunk_asv_embeddings)
                
                del embeddings, pooled_embeddings, tokens_flat, mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            asv_embeddings = torch.cat(asv_embeddings_list, dim=1)
            
            if return_nucleotides and nucleotide_predictions_list:
                nucleotide_predictions = torch.cat(nucleotide_predictions_list, dim=1)
            else:
                nucleotide_predictions = None
        else:
            tokens_flat = tokens.reshape(batch_size * num_asvs, seq_len)
            mask = (tokens_flat > 0).long()
            
            embeddings = self.token_embedding(tokens_flat)
            embeddings = self.position_embedding(embeddings)
            embeddings = self.transformer(embeddings, mask=mask)
            
            nucleotide_predictions = None
            if self.predict_nucleotides and return_nucleotides:
                nucleotide_logits = self.nucleotide_head(embeddings)
                nucleotide_predictions = nucleotide_logits.reshape(batch_size, num_asvs, seq_len, self.vocab_size)
            
            pooled_embeddings = self.attention_pooling(embeddings, mask=mask)
            asv_embeddings = pooled_embeddings.reshape(batch_size, num_asvs, self.embedding_dim)
        
        if return_nucleotides and nucleotide_predictions is not None:
            return asv_embeddings, nucleotide_predictions
        else:
            return asv_embeddings
