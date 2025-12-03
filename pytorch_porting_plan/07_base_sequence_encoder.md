# SampleSequenceEncoder

## Objective
Implement encoder that processes ASV embeddings at the sample level.

## Architecture

**Data Flow**:
```
Input: [B, S, L] tokens
  ↓
ASVEncoder: [B, S, D] ASV embeddings
  ↓
Position Embedding: [B, S, D]
  ↓
Transformer: [B, S, D]
  ↓
Output: [B, S, D] base embeddings
```

## Components

**1. ASVEncoder**
- Processes nucleotide sequences (from `06_asv_encoder.md`)
- Outputs ASV-level embeddings
- Handles nucleotide-level processing

**2. Sample-Level Position Embeddings**
- Adds positional information for ASV ordering
- Max length: `token_limit + 5` (for special tokens)
- Encodes ASV position within sample

**3. Sample-Level Transformer**
- Processes ASV embeddings at sample level
- Learns relationships between ASVs
- Multiple layers of self-attention

## Implementation Requirements

**Class Structure**:
- Inherit from `nn.Module`
- Composes ASVEncoder and sample-level components
- Forward pass coordinates processing

**Key Operations**:
1. Process nucleotides: Call ASVEncoder
2. Create ASV mask: Identify valid ASVs
3. Add positions: Sample-level position embeddings
4. Transform: Sample-level transformer
5. Return: Base embeddings (and nucleotide predictions if training)

**Training vs Inference**:
- Training: Return base embeddings and nucleotide predictions
- Inference: Return only base embeddings

## Implementation Checklist

- [x] Create `SampleSequenceEncoder` class
- [x] Initialize ASVEncoder
- [x] Initialize sample-level position embedding
- [x] Initialize sample-level transformer
- [x] Implement forward pass
- [x] Handle mask creation from tokens
- [x] Handle training vs inference modes
- [x] Pass through nucleotide predictions
- [x] Test with dummy data
- [x] Verify output shapes

## Key Considerations

### ASV Masking
- Create mask by summing tokens: `mask = (tokens.sum(dim=-1) > 0)`
- Identifies ASVs with at least one valid token
- Used for sample-level attention

### Position Embeddings
- Sample-level positions encode ASV ordering
- Important for learning ASV relationships
- May need special handling for first/last tokens

### Integration
- Uses ASVEncoder for nucleotide processing
- Outputs base embeddings for downstream tasks
- Handles both training and inference modes

## Testing Requirements

- Test with different numbers of ASVs per sample
- Test with partial ASV masking
- Test training mode (with nucleotide predictions)
- Test inference mode (embeddings only)
- Verify output shapes: `[B, S, D]` for base embeddings

## Notes

- SampleSequenceEncoder combines nucleotide and sample-level processing
- Output embeddings ready for encoder or regression heads
- Critical component connecting ASV and sample levels
