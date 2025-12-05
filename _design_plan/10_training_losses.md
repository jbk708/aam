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
- **Testing**: Comprehensive unit tests (18 loss tests + 11 metrics tests passing)

## Known Issues and Lessons Learned

### Shape Mismatch Handling (PYT-8.6)
**Status:** Reverted - Needs re-implementation

**Issue:**
Previous attempt to handle shape mismatches in `compute_base_loss` (commit 1a68364) caused NaN propagation issues that corrupted model weights. The implementation used slicing operations (`base_pred[:, :base_true.shape[1]]`) which, while preserving gradients, caused numerical instability when combined with other losses.

**Root Cause:**
- Slicing operations in loss computation can interact unexpectedly with other losses (nucleotide loss) flowing through the same computation graph
- Complex conditional logic in loss functions can affect autograd behavior
- NaN in model outputs (from corrupted embeddings) propagates through loss computation, creating NaN gradients that corrupt weights

**Lessons Learned:**
1. **Use `drop_last=True` in DataLoader** - Ensures all batches are exactly `batch_size`, eliminating shape mismatches at the source
2. **Keep loss computation simple** - Avoid complex conditionals and slicing operations in loss functions
3. **Add early NaN detection** - Check for NaN in model outputs BEFORE computing loss, not after
4. **Test with remainder batches** - Always test edge cases (last batch with fewer samples)
5. **Monitor embedding weights** - NaN in embeddings propagates through entire forward pass. Check early.

**Current State:**
- Reverted to commit 68597fc (pre-shape-mismatch-fix)
- Current implementation: Simple MSE loss without shape handling
- Works correctly with consistent batch sizes (using `drop_last=True`)
- See PYT-8.6 ticket for proper re-implementation approach

**Future Implementation:**
- Use `drop_last=True` as primary solution (prevents shape mismatches)
- If shape handling still needed, use `torch.index_select` or padding instead of slicing
- Add NaN checks before loss computation
- Keep loss functions simple and straightforward
