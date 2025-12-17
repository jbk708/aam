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
        vocab_size: int = 6,
        embedding_dim: int = 128,
        max_bp: int = 150,
        num_layers: int = 2,
        num_heads: int = 4,
        intermediate_size: int = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        predict_nucleotides: bool = False,
        asv_chunk_size: Optional[int] = None,
        gradient_checkpointing: bool = False,
        attn_implementation: Optional[str] = "sdpa",
    ):
        """Initialize ASVEncoder.

        Args:
            vocab_size: Vocabulary size (default: 6 for pad, A, C, G, T, START)
            embedding_dim: Embedding dimension
            max_bp: Maximum sequence length (base pairs)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            intermediate_size: FFN intermediate size (defaults to 4 * embedding_dim)
            dropout: Dropout rate
            activation: Activation function ('gelu' or 'relu')
            predict_nucleotides: Whether to include nucleotide prediction head
            asv_chunk_size: Process ASVs in chunks of this size (None = process all at once)
            gradient_checkpointing: Whether to use gradient checkpointing to save memory
            attn_implementation: Which SDPA backend to use ('sdpa', 'flash', 'mem_efficient', 'math')
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
            gradient_checkpointing=gradient_checkpointing,
            attn_implementation=attn_implementation,
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
                
                # Identify all-padding sequences (will cause NaN in transformer)
                mask_sum = mask.sum(dim=-1)  # [batch_size * chunk_num_asvs]
                all_padding = (mask_sum == 0)  # [batch_size * chunk_num_asvs]
                
                embeddings = self.token_embedding(tokens_flat)
                embeddings = self.position_embedding(embeddings)
                
                # Handle all-padding sequences: skip transformer and set embeddings to zero
                # This prevents NaN from being produced by the transformer
                if all_padding.any():
                    # For sequences with valid positions, run transformer normally
                    # For all-padding sequences, set embeddings to zero (skip transformer)
                    valid_mask = ~all_padding  # [batch_size * chunk_num_asvs]
                    
                    if valid_mask.any():
                        # Process valid sequences through transformer
                        valid_indices = torch.where(valid_mask)[0]
                        valid_embeddings = embeddings[valid_indices]
                        valid_mask_tensor = mask[valid_indices]
                        
                        valid_transformed = self.transformer(valid_embeddings, mask=valid_mask_tensor)
                        
                        # Combine: valid sequences get transformer output, all-padding get zeros
                        all_embeddings = torch.zeros_like(embeddings)
                        all_embeddings[valid_indices] = valid_transformed
                        embeddings = all_embeddings
                    else:
                        # All sequences are all-padding, set all to zero
                        embeddings = torch.zeros_like(embeddings)
                else:
                    # No all-padding sequences, normal transformer processing
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
            
            # Identify all-padding sequences (will cause NaN in transformer)
            mask_sum = mask.sum(dim=-1)  # [batch_size * num_asvs]
            all_padding = (mask_sum == 0)  # [batch_size * num_asvs]
            
            embeddings = self.token_embedding(tokens_flat)
            embeddings = self.position_embedding(embeddings)
            
            # Handle all-padding sequences: skip transformer and set embeddings to zero
            # This prevents NaN from being produced by the transformer
            if all_padding.any():
                # For sequences with valid positions, run transformer normally
                # For all-padding sequences, set embeddings to zero (skip transformer)
                valid_mask = ~all_padding  # [batch_size * num_asvs]
                
                if valid_mask.any():
                    # Process valid sequences through transformer
                    valid_indices = torch.where(valid_mask)[0]
                    valid_embeddings = embeddings[valid_indices]
                    valid_mask_tensor = mask[valid_indices]
                    
                    valid_transformed = self.transformer(valid_embeddings, mask=valid_mask_tensor)
                    
                    # Combine: valid sequences get transformer output, all-padding get zeros
                    all_embeddings = torch.zeros_like(embeddings)
                    all_embeddings[valid_indices] = valid_transformed
                    embeddings = all_embeddings
                else:
                    # All sequences are all-padding, set all to zero
                    embeddings = torch.zeros_like(embeddings)
            else:
                # No all-padding sequences, normal transformer processing
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
