# Investigation: Batching Logic and UniFrac Distance Extraction

## Overview

Investigation into potential issues with batching logic and UniFrac distance extraction, particularly when batches are shuffled and `token_limit` truncation occurs.

## Code Flow Analysis

### 1. `collate_fn` in `aam/data/dataset.py`

**Order of Operations:**
1. Iterate through batch samples in order (line 42: `for sample in batch:`)
2. Truncate tokens/counts if `num_asvs > token_limit` (lines 47-50)
3. Pad tokens/counts to `token_limit` (lines 52-55)
4. Append padded tokens/counts to lists (lines 57-58)
5. Append `sample_id` to list (line 59)
6. Stack tokens/counts into tensors (lines 65-66)
7. Extract UniFrac distances using `sample_ids` list (line 75)

**Key Observation:** The order of `sample_ids` matches the order of tokens/counts because:
- All three are appended in the same loop iteration
- `torch.stack()` preserves the order of the list

### 2. `extract_batch_distances` in `aam/data/unifrac.py`

**Order of Operations:**
1. Filter distance matrix to batch samples (line 142: `distances.filter(sample_ids)`)
2. Create mapping from filtered IDs to indices (line 146)
3. Create reorder indices based on batch `sample_ids` order (line 147)
4. Reorder distance matrix using `np.ix_` (line 148)

**Key Observation:** The reordering logic appears correct:
- It filters first to get only batch samples
- Then reorders to match the exact order of `sample_ids` in the batch
- Uses `np.ix_` to reorder both rows and columns simultaneously

### 3. Potential Issues

#### Issue 1: Order Consistency After Truncation
**Status:** ✅ **VERIFIED CORRECT**
- Truncation (lines 47-50) only modifies tokens/counts, not `sample_id`
- `sample_id` is appended after truncation (line 59)
- Order is preserved

#### Issue 2: Shuffled Batch Order
**Status:** ✅ **VERIFIED CORRECT**
- When DataLoader shuffles, batch samples are in random order
- `collate_fn` preserves this order in `sample_ids` list
- `extract_batch_distances` reorders distance matrix to match batch order
- Tests confirm this works correctly (see `test_collate_fn_extracts_batch_distances_shuffled_order`)

#### Issue 3: Distance Matrix Shape vs. Model Output Shape
**Potential Issue:** ⚠️ **NEEDS VERIFICATION**

When using `token_limit` with pretraining:
- Distance matrix shape: `[batch_size, batch_size]` for unweighted UniFrac
- Model `base_output_dim` is set to `batch_size` (line 525 in `cli.py`)
- Model outputs `base_prediction` with shape `[batch_size, batch_size]`

**Question:** Is there a mismatch when:
1. Batch is shuffled
2. Distance matrix is reordered to match batch order
3. Model processes tokens in batch order
4. Loss computation expects distances to match predictions?

**Hypothesis:** The order should match because:
- Tokens are in batch order (from `collate_fn`)
- Distances are reordered to match batch order (from `extract_batch_distances`)
- Model processes tokens in batch order
- Loss compares `base_prediction[batch_i, batch_j]` with `base_target[batch_i, batch_j]`

#### Issue 4: Token Limit Truncation and Distance Extraction
**Status:** ✅ **VERIFIED CORRECT**
- Truncation happens at ASV level (within each sample)
- Distance extraction happens at sample level (between samples)
- These are independent operations
- Truncation doesn't affect sample-level distances

## Test Results

### Test 1: Order Consistency
- ✅ `sample_ids` order matches tokens/counts order
- ✅ Distance matrix is correctly reordered

### Test 2: Shuffled Batches
- ✅ Shuffled batch order is preserved
- ✅ Distance matrix is correctly reordered to match shuffled order
- ✅ Distances match expected values

### Test 3: DataLoader Shuffling
- ✅ DataLoader shuffling works correctly
- ✅ Distance matrix extraction works with shuffled batches
- ✅ Order consistency maintained

## Conclusion

**The batching logic appears correct:**
1. ✅ Order of `sample_ids` matches order of tokens/counts
2. ✅ Distance matrix is correctly reordered to match batch order
3. ✅ Shuffling is handled correctly
4. ✅ Truncation doesn't affect distance extraction

## Remaining Questions

1. **Is there a mismatch between distance matrix order and model prediction order?**
   - Need to verify that `base_prediction` order matches `base_target` order
   - Check loss computation to ensure indices align correctly

2. **Could there be an issue with how distances are used in loss computation?**
   - Check `compute_base_loss` in `aam/training/losses.py`
   - Verify that pairwise distances are correctly matched to predictions

3. **Could NaN be coming from distance matrix values themselves?**
   - Check if distance matrix contains NaN/Inf values
   - Verify distance computation is stable

## Next Steps

1. Add detailed logging to verify order consistency at runtime
2. Check loss computation to ensure distance matrix indices match prediction indices
3. Verify distance matrix values are valid (no NaN/Inf)
4. Test with actual pretraining scenario to see if issue reproduces

## Related Files

- `aam/data/dataset.py` - `collate_fn` function
- `aam/data/unifrac.py` - `extract_batch_distances` method
- `aam/training/losses.py` - `compute_base_loss` method
- `aam/cli.py` - Pretraining setup with `base_output_dim`
