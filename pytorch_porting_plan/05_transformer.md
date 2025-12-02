# Transformer Encoder

## Objective
Implement transformer encoder layer for processing sequences with self-attention.

## Architecture

**Components**:
1. Multi-head self-attention
2. Feed-forward network
3. Layer normalization
4. Residual connections
5. Dropout

**Stack Multiple Layers**:
- Configurable number of layers
- Each layer processes sequences independently
- Output normalization after all layers

## Implementation Options

### Option 1: Use PyTorch Built-in (Recommended)

**Advantages**:
- Well-tested and optimized
- Supports batch-first format
- Handles masking automatically

**Implementation**:
- Use `nn.TransformerEncoderLayer` and `nn.TransformerEncoder`
- Set `batch_first=True`
- Handle masking with `src_key_padding_mask`

### Option 2: Custom Implementation

**Use When**:
- Need exact control over architecture
- Want to match specific paper implementation
- Need custom attention mechanisms

**Components to Implement**:
- Multi-head attention module
- Feed-forward network module
- Transformer layer (attention + FFN + residuals)
- Stack of transformer layers

## Implementation Requirements

**Class Structure**:
- Inherit from `nn.Module`
- Initialize with:
  - `num_layers`: Number of transformer layers
  - `num_heads`: Number of attention heads
  - `hidden_dim`: Embedding dimension
  - `intermediate_size`: FFN intermediate size
  - `dropout`: Dropout rate
  - `activation`: Activation function ('gelu' or 'relu')

**Forward Pass**:
- Input: `[batch_size, seq_len, hidden_dim]` embeddings
- Mask: `[batch_size, seq_len]` (1 for valid, 0 for padding)
- Output: `[batch_size, seq_len, hidden_dim]` processed embeddings

**Key Features**:
- Pre-norm architecture (more stable)
- Multi-head attention
- GELU activation
- Dropout for regularization

## Implementation Checklist

- [ ] Choose implementation approach (built-in or custom)
- [ ] If built-in: Create wrapper around `nn.TransformerEncoder`
- [ ] If custom: Implement multi-head attention
- [ ] If custom: Implement feed-forward network
- [ ] If custom: Implement transformer layer
- [ ] Implement stacking of multiple layers
- [ ] Handle mask conversion (1 for valid → False for padding)
- [ ] Add output normalization
- [ ] Test with dummy data
- [ ] Verify output shapes
- [ ] Test with masking
- [ ] Test with different numbers of layers

## Key Considerations

### Mask Format
- Input mask: `1` for valid tokens, `0` for padding
- PyTorch `src_key_padding_mask`: `True` means ignore (padding), `False` means attend (valid)
- Conversion: `mask_pytorch = (mask_input == 0)` (invert)

### Batch First
- Use `batch_first=True` for `(batch, seq, hidden)` format
- Consistent with rest of model

### Normalization
- Pre-norm: Normalize before attention/FFN (more stable)
- Post-norm: Normalize after residual (traditional)
- Use LayerNorm with `eps=1e-6`

### Activation
- GELU is standard for transformers
- Can use `nn.GELU()` or `F.gelu()`

## Testing Requirements

- Test with different batch sizes and sequence lengths
- Test with masking (partial and full padding)
- Test with different numbers of layers
- Verify gradients flow correctly
- Check for gradient explosion/vanishing

## Notes

- Transformer is core component used throughout model
- Proper masking is critical for correct attention
- Pre-norm architecture is more stable for training
