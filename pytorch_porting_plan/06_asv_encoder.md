# ASVEncoder

## Objective
Implement encoder that processes nucleotide sequences at the ASV (sequence) level.

## Architecture

**Data Flow**:
```
Input: [B, S, L] tokens
  ↓
Reshape: [B*S, L]
  ↓
Embedding: [B*S, L, D]
  ↓
Position Embedding: [B*S, L, D]
  ↓
Transformer: [B*S, L, D]
  ↓
Attention Pooling: [B*S, D]
  ↓
Reshape: [B, S, D]
```

Where:
- B = batch size
- S = number of ASVs per sample
- L = max_bp (sequence length)
- D = embedding dimension

## Components

**1. Embedding Layer**
- Maps nucleotide tokens to embeddings
- Vocabulary size: 5 (pad, A, C, G, T) or larger if special tokens
- Embedding dimension: `embedding_dim`

**2. Position Embeddings**
- Adds positional information for nucleotide positions
- Max length: `max_bp + 1`
- Learned embeddings

**3. Transformer Encoder**
- Processes nucleotide sequences
- Multiple layers of self-attention
- Learns nucleotide-level patterns

**4. Attention Pooling**
- Pools sequence to single embedding per ASV
- Learns which nucleotides are important

**5. Nucleotide Prediction Head** (Optional)
- Auxiliary task for self-supervised learning
- Predicts nucleotide at each position
- Only used during training

## Implementation Requirements

**Class Structure**:
- Inherit from `nn.Module`
- Initialize all layers in `__init__()`
- Forward pass handles reshaping and processing

**Key Operations**:
1. Reshape input: Flatten batch and ASV dimensions
2. Create mask: Identify valid tokens (non-padding)
3. Embed tokens: Convert to embeddings
4. Add positions: Add position embeddings
5. Transform: Apply transformer encoder
6. Predict nucleotides: Optional auxiliary task
7. Pool: Attention pooling to single embedding
8. Reshape output: Back to batch structure

**Training vs Inference**:
- Training: Return embeddings and nucleotide predictions
- Inference: Return only embeddings

## Implementation Checklist

- [x] Create `ASVEncoder` class
- [x] Initialize embedding layer
- [x] Initialize position embedding
- [x] Initialize transformer encoder
- [x] Initialize attention pooling
- [x] Initialize nucleotide prediction head (optional)
- [x] Implement forward pass with reshaping
- [x] Handle mask creation
- [x] Handle training vs inference modes
- [x] Test with dummy data
- [x] Verify output shapes
- [x] Test with different sequence lengths

## Key Considerations

### Reshaping
- Flatten batch and ASV dimensions for parallel processing
- Process all ASVs in parallel: `[B*S, L]`
- Reshape back after processing: `[B, S, D]`
- Use `view()` or `reshape()` - ensure contiguous if needed

### Masking
- Create mask from tokens: `mask = (tokens > 0)`
- Use for transformer and attention pooling
- Convert to correct format for each layer

### Nucleotide Prediction
- Auxiliary task for self-supervised learning
- Predicts nucleotide at each position
- Only compute during training
- Can be skipped for simpler prototype

### Memory Efficiency
- Processing `B*S` sequences in parallel uses memory
- Consider batch size limits
- Use gradient checkpointing if needed

## Testing Requirements

- Test with different batch sizes and ASV counts
- Test with variable-length sequences (with padding)
- Test training mode (with nucleotide predictions)
- Test inference mode (embeddings only)
- Verify output shapes: `[B, S, D]` for embeddings

## Notes

- ASVEncoder processes sequences at finest granularity
- Output embeddings capture sequence-level information
- Nucleotide prediction adds self-supervised learning signal
