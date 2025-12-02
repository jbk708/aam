# Core Layers

## Objective
Implement fundamental neural network layers: AttentionPooling and PositionEmbedding.

## Components

### 1. AttentionPooling

**Purpose**: Pool sequence-level embeddings to single vector using learned attention weights.

**Architecture**:
- Query projection: `Linear(hidden_dim → 1)`
- Attention computation: Softmax over sequence dimension
- Weighted sum: Sum embeddings weighted by attention
- Normalization: LayerNorm on pooled output

**Input**: `[batch_size, seq_len, hidden_dim]` embeddings
**Output**: `[batch_size, hidden_dim]` pooled embedding

**Key Features**:
- Learns which positions are important
- Handles variable-length sequences via masking
- Scales attention scores by `sqrt(hidden_dim)` for stability

**Implementation Requirements**:
- Inherit from `nn.Module`
- Initialize: `nn.Linear(hidden_dim, 1, bias=False)` for query, `nn.LayerNorm(hidden_dim)` for norm
- Forward: Compute attention scores, apply mask, softmax, weighted sum, normalize

### 2. PositionEmbedding

**Purpose**: Add learned position information to embeddings.

**Architecture**:
- Embedding layer: `Embedding(max_length, hidden_dim)`
- Position indices: `[0, 1, 2, ..., seq_len-1]`
- Addition: `embeddings + position_embeddings`

**Input**: `[batch_size, seq_len, hidden_dim]` embeddings
**Output**: `[batch_size, seq_len, hidden_dim]` embeddings with positions

**Key Features**:
- Learned (not fixed sinusoidal)
- Adds positional information without concatenation
- Handles variable sequence lengths

**Implementation Requirements**:
- Inherit from `nn.Module`
- Initialize: `nn.Embedding(max_length, hidden_dim)`
- Forward: Create position indices, get embeddings, add to input

### 3. Utility Functions

**Masking Functions**:
- `float_mask(tensor)`: Convert to float mask (1 for nonzero, 0 for zero)
- `create_mask_from_tokens(tokens)`: Create mask from token tensor
- `apply_mask(embeddings, mask)`: Apply mask to embeddings

## Implementation Checklist

- [x] Create `AttentionPooling` class
- [x] Implement query projection
- [x] Implement attention computation with masking
- [x] Implement weighted sum
- [x] Add LayerNorm
- [x] Create `PositionEmbedding` class
- [x] Implement position index creation
- [x] Implement embedding lookup and addition
- [x] Create utility masking functions
- [x] Test with dummy data
- [x] Verify output shapes
- [x] Test with masking

## Key Considerations

### AttentionPooling
- Mask format: `1` for valid, `0` for padding
- Use `masked_fill` to set masked positions to `-inf` before softmax
- Scale scores by `sqrt(hidden_dim)` for numerical stability
- Ensure attention weights sum to 1 over valid positions

### PositionEmbedding
- Position indices should be on same device as input
- Use actual sequence length, not max_length, for efficiency
- Initialize embeddings (zeros, normal, or uniform)
- Handle variable sequence lengths correctly

### Device Handling
- Ensure all tensors on same device
- Position indices created on input device
- Use `.to(device)` where needed

## Testing Requirements

- Test AttentionPooling with different sequence lengths
- Test with and without masking
- Verify attention weights sum to 1
- Test PositionEmbedding with different sequence lengths
- Verify embeddings are added (not concatenated)
- Test utility functions with various inputs

## Notes

- AttentionPooling is used throughout the model
- PositionEmbedding adds crucial positional information
- Proper masking is essential for correct attention computation
