# Fix NaN in Nucleotide Predictions During Pretraining with Token Limit

**Status:** ✅ Completed  
**Ticket:** PYT-8.9

## Overview
Fixed NaN values appearing in nucleotide predictions (`nuc_predictions`) during pretraining when using `--token-limit` with gradient accumulation. The issue was caused by all-padding sequences (sequences consisting entirely of padding tokens) causing NaN in transformer attention mechanisms.

## Problem

### Error Details
```
ERROR: NaN in nuc_predictions before loss computation
nuc_predictions shape=torch.Size([6, 512, 151, 6])
nuc_predictions min=nan, max=nan
ValueError: NaN values found in nuc_pred with shape torch.Size([6, 512, 151, 6])
```

### Root Cause Analysis

**Primary Issue**: All-padding sequences (sequences with all zero tokens) cause NaN in PyTorch's TransformerEncoder when all positions are masked. This occurs because `softmax(all -inf)` produces NaN.

**Investigation Process**:
1. ✅ Verified batching and UniFrac distance extraction logic was correct (no issues there)
2. ✅ Confirmed START_TOKEN is preserved after truncation (not the issue)
3. ✅ Traced NaN propagation: Transformer → Attention Pooling → Nucleotide Head → Final Predictions
4. ✅ Identified that transformer produces NaN for all-padding sequences even before attention pooling
5. ✅ Found that some samples could have all ASVs truncated away or have zero counts after truncation

## Solution

### Fix 1: AttentionPooling - Handle All-Padding Sequences
**File**: `aam/models/attention_pooling.py`

**Changes**:
- Detect all-padding sequences (where `mask.sum() == 0`)
- Set scores to `0.0` for all-padding sequences before softmax (prevents `softmax(all -inf)`)
- Use uniform attention weights (`1.0 / seq_len`) for all-padding sequences after normalization
- Prevents NaN in attention pooling layer

**Code**:
```python
mask_sum = mask.sum(dim=-1, keepdim=True)  # [batch_size, 1]
all_padding = (mask_sum == 0)  # [batch_size, 1]

if all_padding.any():
    # Set scores to 0 for all-padding sequences (prevents NaN from softmax(all -inf))
    scores = scores.masked_fill(all_padding, 0.0)
    # For sequences with valid positions, mask padding with -inf
    scores = scores.masked_fill(~all_padding & (mask == 0), float("-inf"))

attention_weights = torch.softmax(scores, dim=-1)

if mask is not None:
    attention_weights = attention_weights * mask
    attention_weights = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-8)
    
    # For all-padding sequences, use uniform attention weights
    if all_padding.any():
        attention_weights = attention_weights.masked_fill(all_padding, 1.0 / seq_len)
```

### Fix 2: ASVEncoder - Mask NaN from Transformer Output
**File**: `aam/models/asv_encoder.py`

**Changes**:
- Detect all-padding sequences after transformer output (where NaN originates)
- Explicitly set embeddings to zero for all-padding sequences using `torch.where`
- Prevents NaN propagation from transformer to downstream layers
- Applied to both chunked and non-chunked processing paths

**Code**:
```python
mask_sum = mask.sum(dim=-1)  # [batch_size * num_asvs]
all_padding = (mask_sum == 0)  # [batch_size * num_asvs]

embeddings = self.transformer(embeddings, mask=mask)

# Mask out NaN values for all-padding sequences
# Transformer may produce NaN for all-padding sequences
if all_padding.any():
    # Set embeddings to zero for all-padding sequences
    all_padding_expanded = all_padding.unsqueeze(-1).unsqueeze(-1)  # [batch_size * num_asvs, 1, 1]
    embeddings = torch.where(
        all_padding_expanded,
        torch.zeros_like(embeddings),
        embeddings
    )
```

### Fix 3: Data Validation - Ensure Sample Integrity
**File**: `aam/data/dataset.py`

**Changes**:
- Added validation in `collate_fn` to ensure samples have at least one ASV with `count > 0` after truncation
- Added validation in `__getitem__` to ensure samples yield at least one ASV
- Prevents all-padding samples from entering the model

**Code**:
```python
# Verify sample has at least one ASV with count > 0
# This prevents all-padding samples that cause NaN in attention pooling
if num_asvs == 0 or (counts.sum() == 0).all():
    error_msg = (
        f"Sample {sample['sample_id']} has no ASVs with count > 0 "
        f"(num_asvs={num_asvs}, counts_sum={counts.sum().item()})"
    )
    raise ValueError(error_msg)
```

### Fix 4: Loss Function Safety - Safe Tensor Statistics
**File**: `aam/training/losses.py`

**Changes**:
- Added `_format_tensor_stats()` helper function to safely format tensor statistics
- Handles integer tensors (like `tokens`) without attempting to compute `mean()` or `std()`
- Prevents `RuntimeError` when printing error details for integer tensors

**Code**:
```python
def _format_tensor_stats(tensor: torch.Tensor) -> str:
    min_val = tensor.min().item()
    max_val = tensor.max().item()
    
    # Handle integer tensors (can't compute mean)
    if tensor.dtype in (torch.int64, torch.int32, torch.int16, torch.int8, torch.long):
        return f"min={min_val}, max={max_val} (integer tensor)"
    else:
        mean_val = tensor.mean().item()
        return f"min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}"
```

## Testing

### Debug Scripts Created
- `debug/investigate_nucleotide_nan.py` - Traces NaN step-by-step through model
- `debug/investigate_all_padding.py` - Analyzes padding patterns in batches
- `debug/investigate_batching_logic.py` - Verifies batching correctness
- `debug/debug_nan_token_limit.py` - Comprehensive debugging script

### Verification
- ✅ No NaN in transformer output after fix
- ✅ No NaN in attention pooling output
- ✅ No NaN in nucleotide predictions
- ✅ Training stability confirmed with various token_limit values (64, 256, 512, 1024)
- ✅ Works correctly with gradient accumulation
- ✅ Works correctly with different batch sizes

## Key Lessons Learned

1. **All-padding sequences cause NaN in transformers**: When all positions are masked, `softmax(all -inf)` produces NaN. This must be handled explicitly.

2. **NaN propagates through the model**: NaN originating in the transformer propagates to all downstream layers. Fixing at the source (transformer output) is more effective than fixing downstream.

3. **Data validation is critical**: Ensuring samples have at least one valid ASV prevents invalid batches from entering the model.

4. **Integer tensor handling**: When logging/debugging, integer tensors (like tokens) cannot have `mean()` or `std()` computed. Use conditional logic to handle different tensor types.

5. **Comprehensive debugging**: Creating detailed debugging scripts helped identify the exact point where NaN first appears, making the fix more targeted and effective.

## Files Modified

- `aam/models/attention_pooling.py` - Handle all-padding sequences in attention mechanism
- `aam/models/asv_encoder.py` - Mask NaN from transformer output for all-padding sequences
- `aam/data/dataset.py` - Add validation to ensure sample integrity after truncation
- `aam/training/losses.py` - Safe tensor statistics formatting for error messages
- `debug/` - Created comprehensive debugging scripts and documentation

## Dependencies

- PYT-8.8: Add Start Token to Prevent All-Padding Sequence NaN Issues (completed)

## Related Issues

This ticket addresses NaN issues that were partially addressed in PYT-8.8 (START_TOKEN addition). However, PYT-8.9 specifically addresses NaN that occurs when using `token_limit` truncation, which can create all-padding sequences even with START_TOKEN present.
