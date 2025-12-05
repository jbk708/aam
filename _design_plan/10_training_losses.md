# Training Losses and Metrics

**Status:** ✅ Completed (Reverted to stable version)

## Overview
Multi-task loss functions and metrics. Implemented in `aam/training/losses.py` and `aam/training/metrics.py`.

## Loss Functions
- **Target Loss**: MSE (regression) or NLL (classification) with optional class weights
- **Count Loss**: Masked MSE for ASV counts
- **Base Loss**: MSE for UniFrac distances (weighted by `penalty`)
- **Nucleotide Loss**: Masked CrossEntropy (weighted by `nuc_penalty`)
- **Total Loss**: Weighted sum of all losses

## Metrics
- **Regression**: MAE, MSE, R-squared
- **Classification**: Accuracy, Confusion Matrix, Precision/Recall/F1

## Implementation
- **Losses**: `MultiTaskLoss` in `aam/training/losses.py`
- **Metrics**: `compute_regression_metrics()`, `compute_classification_metrics()`, `compute_count_metrics()` in `aam/training/metrics.py`
- **Testing**: Comprehensive unit tests (25 loss tests + 11 metrics tests passing)

## Known Issues and Lessons Learned

### Shape Mismatch Handling (PYT-8.6)
**Status:** ✅ Completed

**Issue:**
Previous attempt to handle shape mismatches in `compute_base_loss` (commit 1a68364) caused NaN propagation issues that corrupted model weights. The implementation used slicing operations (`base_pred[:, :base_true.shape[1]]`) which, while preserving gradients, caused numerical instability when combined with other losses.

**Root Cause:**
- Slicing operations in loss computation can interact unexpectedly with other losses (nucleotide loss) flowing through the same computation graph
- Complex conditional logic in loss functions can affect autograd behavior
- NaN in model outputs (from corrupted embeddings) propagates through loss computation, creating NaN gradients that corrupt weights

**Solution:**
- **Primary Fix:** Use `drop_last=True` in DataLoader to ensure all batches are exactly `batch_size`, eliminating shape mismatches at the source
- **Verification:** Confirmed `drop_last=True` is set in all DataLoaders (train, pretrain, validation)
- **NaN Checks:** Verified NaN checks are in place before loss computation (trainer.py and losses.py)
- **Tests:** Added 7 comprehensive tests for shape mismatch scenarios

**Lessons Learned:**
1. **Use `drop_last=True` in DataLoader** - Ensures all batches are exactly `batch_size`, eliminating shape mismatches at the source ✅ IMPLEMENTED
2. **Keep loss computation simple** - Avoid complex conditionals and slicing operations in loss functions ✅ VERIFIED
3. **Add early NaN detection** - Check for NaN in model outputs BEFORE computing loss, not after ✅ VERIFIED
4. **Test with remainder batches** - Always test edge cases (last batch with fewer samples) ✅ COMPLETED
5. **Monitor embedding weights** - NaN in embeddings propagates through entire forward pass. Check early. ✅ VERIFIED

**Current State:**
- ✅ `drop_last=True` verified in all DataLoaders (prevents shape mismatches)
- ✅ Simple MSE loss without shape handling (keeps computation simple)
- ✅ Shape validation in `compute_base_loss` raises clear errors if mismatch occurs
- ✅ Comprehensive tests added (7 new tests covering all scenarios)
- ✅ All tests passing (25 passed, 1 skipped)

**Implementation Details:**
- `drop_last=True` ensures consistent batch sizes, preventing shape mismatches
- Shape validation in `compute_base_loss` provides clear error messages if issues occur
- NaN checks prevent corrupted gradients from propagating
- Tests verify correct behavior with consistent batch sizes and error handling for mismatches
