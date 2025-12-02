"""ASV-level sequence encoder for processing nucleotide sequences."""

import torch
import torch.nn as nn

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
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_bp = max_bp
        self.predict_nucleotides = predict_nucleotides
        
        if intermediate_size is None:
            intermediate_size = 4 * embedding_dim
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Position embedding (max_bp + 1 to account for 0-indexed positions)
        self.position_embedding = PositionEmbedding(max_length=max_bp + 1, hidden_dim=embedding_dim)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=embedding_dim,
            intermediate_size=intermediate_size,
            dropout=dropout,
            activation=activation,
        )
        
        # Attention pooling
        self.attention_pooling = AttentionPooling(hidden_dim=embedding_dim)
        
        # Nucleotide prediction head (optional)
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
        
        # Ensure tokens are long integers for embedding layer
        tokens = tokens.long()
        
        # Reshape: [B, S, L] -> [B*S, L]
        tokens_flat = tokens.view(batch_size * num_asvs, seq_len)
        
        # Create mask: 1 for valid tokens, 0 for padding
        mask = (tokens_flat > 0).long()
        
        # Embed tokens: [B*S, L] -> [B*S, L, D]
        embeddings = self.token_embedding(tokens_flat)
        
        # Add position embeddings: [B*S, L, D] -> [B*S, L, D]
        embeddings = self.position_embedding(embeddings)
        
        # Apply transformer: [B*S, L, D] -> [B*S, L, D]
        embeddings = self.transformer(embeddings, mask=mask)
        
        # Optionally predict nucleotides
        nucleotide_predictions = None
        if self.predict_nucleotides and return_nucleotides:
            # [B*S, L, D] -> [B*S, L, vocab_size]
            nucleotide_logits = self.nucleotide_head(embeddings)
            # Reshape back: [B*S, L, vocab_size] -> [B, S, L, vocab_size]
            nucleotide_predictions = nucleotide_logits.view(batch_size, num_asvs, seq_len, self.vocab_size)
        
        # Pool: [B*S, L, D] -> [B*S, D]
        pooled_embeddings = self.attention_pooling(embeddings, mask=mask)
        
        # Reshape back: [B*S, D] -> [B, S, D]
        asv_embeddings = pooled_embeddings.view(batch_size, num_asvs, self.embedding_dim)
        
        # Return based on whether nucleotide predictions are requested
        if return_nucleotides and nucleotide_predictions is not None:
            return asv_embeddings, nucleotide_predictions
        else:
            return asv_embeddings
