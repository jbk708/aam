# Fix Sigmoid Saturation in Distance Normalization

**Status:** 🔴 **HIGH PRIORITY** - In Progress  
**Priority:** HIGH  
**Created:** 2025-12-15  
**Ticket:** PYT-11.5

## Problem Statement

Validation output plots show all predictions clustering at 0.55 and not changing during training. This indicates sigmoid saturation in the distance normalization, preventing the model from learning meaningful distance relationships.

### Symptoms
- All predicted UniFrac distances are approximately 0.55
- Predictions do not change during training
- Model appears to be stuck in a local minimum
- Loss decreases but predictions remain constant

### Root Cause Analysis

The current normalization approach in `compute_pairwise_distances()` and `compute_stripe_distances()`:

1. Normalizes distances by max distance: `normalized = distances / (max_dist * scale)`
2. Applies sigmoid: `distances = torch.sigmoid(normalized)`

**Problem:** After normalization by max distance, values are typically in [0, 1] range. Applying sigmoid to this range:
- `sigmoid(0) ≈ 0.5`
- `sigmoid(1) ≈ 0.73`
- Most values cluster around 0.5-0.73, explaining the 0.55 observation

This causes:
- **Gradient saturation**: Sigmoid gradients are very small in the [0.5, 0.73] range
- **Loss of information**: All distance relationships compressed into narrow [0.5, 0.73] range
- **Training stagnation**: Model cannot learn because gradients are too small

## Proposed Solution

### Option 1: Remove Sigmoid, Use Direct Normalization (Recommended)
- Normalize by max distance only: `distances = distances / max_dist`
- Keep values in [0, 1] range without sigmoid
- Preserves distance relationships and gradient flow
- Matches original TensorFlow approach (no sigmoid)

### Option 2: Use Different Normalization Range
- Normalize to wider range before sigmoid: `normalized = distances / (max_dist * scale * 2)`
- This would map distances to roughly [-1, 1] before sigmoid
- Sigmoid would then map to [0.27, 0.73] range (wider than current)

### Option 3: Use Tanh Instead of Sigmoid
- Use `tanh` which maps to [-1, 1] range
- Then normalize to [0, 1]: `(tanh(normalized) + 1) / 2`
- Provides better gradient flow than sigmoid

## Implementation Plan

### Phase 1: Investigation
1. ✅ Confirm sigmoid saturation issue (validation plots show 0.55)
2. Add diagnostic logging to track:
   - Raw distance values before normalization
   - Normalized values before sigmoid
   - Final values after sigmoid
   - Gradient magnitudes
3. Plot distribution of values at each stage

### Phase 2: Fix Implementation
1. **Update `compute_pairwise_distances()`:**
   - Remove sigmoid application
   - Use direct normalization: `distances = distances / max_dist`
   - Ensure diagonal remains 0.0
   - Keep values in [0, 1] range

2. **Update `compute_stripe_distances()`:**
   - Remove sigmoid application
   - Use direct normalization: `distances = distances / max_dist`
   - Keep values in [0, 1] range

3. **Update tests:**
   - Verify distances are in [0, 1] range
   - Verify gradient flow is maintained
   - Verify no saturation occurs

### Phase 3: Validation
1. Run training with fixed normalization
2. Verify predictions vary across [0, 1] range
3. Verify loss decreases and predictions improve
4. Compare training dynamics before/after fix

## Files to Modify

- `aam/training/losses.py`:
  - `compute_pairwise_distances()` - Remove sigmoid, use direct normalization
  - `compute_stripe_distances()` - Remove sigmoid, use direct normalization

- `tests/test_losses.py`:
  - Update tests to verify new normalization behavior
  - Add tests for gradient flow without sigmoid

## Expected Outcomes

### Before Fix
- Predictions: All ~0.55
- Gradient magnitudes: Very small (< 1e-6)
- Training: Stagnant, loss plateaus
- Validation plots: Flat line at 0.55

### After Fix
- Predictions: Distributed across [0, 1] range
- Gradient magnitudes: Healthy (> 1e-4)
- Training: Loss decreases, predictions improve
- Validation plots: Predictions vary and correlate with true values

## Testing Strategy

1. **Unit Tests:**
   - Verify distances are in [0, 1] range
   - Verify no sigmoid saturation (values not all ~0.5)
   - Verify gradient flow is maintained
   - Verify diagonal is 0.0 for pairwise distances

2. **Integration Tests:**
   - Run short training run (10 epochs)
   - Verify predictions vary across range
   - Verify loss decreases
   - Verify no NaN/Inf values

3. **Validation:**
   - Compare training curves before/after
   - Verify R² and correlation metrics improve
   - Verify predictions match true distances better

## Risk Assessment

### Low Risk
- Removing sigmoid is straightforward
- Direct normalization is simpler and more interpretable
- Matches original TensorFlow approach

### Medium Risk
- May need to adjust learning rate if gradient magnitudes change significantly
- May need to adjust loss scaling if loss values change

### Mitigation
- Keep `normalize` parameter for backward compatibility
- Add deprecation warning if sigmoid is used
- Monitor gradient magnitudes during training

## Dependencies

- None (self-contained fix)

## Success Criteria

- ✅ Predictions vary across [0, 1] range (not all 0.55)
- ✅ Gradients flow properly (magnitudes > 1e-4)
- ✅ Training loss decreases and predictions improve
- ✅ Validation plots show varied predictions
- ✅ All tests pass
- ✅ No regression in training stability

## Notes

- This issue was discovered during PYT-11.4 validation
- The sigmoid was added to ensure [0, 1] range, but direct normalization achieves the same without saturation
- Original TensorFlow implementation used direct normalization without sigmoid
- The 0.55 value is approximately `sigmoid(0)` which occurs when normalized distances are near 0

## Related Issues

- PYT-11.4: Pre-computed UniFrac matrix ingestion (where issue was discovered)
- PYT-8.16b: Original UniFrac distance computation implementation

---

**Next Steps:**
1. Implement Option 1 (remove sigmoid, use direct normalization)
2. Update tests
3. Run validation training
4. Verify fix resolves saturation issue
