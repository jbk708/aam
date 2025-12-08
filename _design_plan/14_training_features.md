# Training Features and Enhancements

**Status:** ✅ All Completed

## Overview
This document summarizes training-related features and enhancements implemented in Phase 8.

## Completed Features

### PYT-8.1: TensorBoard Train/Val Overlay ✅
**Status:** Completed

Train and validation metrics automatically overlay in TensorBoard graphs for easy comparison. TensorBoard supports automatic overlay when both train and val metrics are selected in the UI.

**Implementation:**
- Train metrics logged with `train/{metric}` prefix
- Validation metrics logged with `val/{metric}` prefix
- Consistent metric naming enables automatic overlay
- No code changes required - TensorBoard UI feature

### PYT-8.2: Single Best Model File Saving ✅
**Status:** Completed

Modified checkpoint saving to keep only the single best model file, replacing previous best model instead of saving multiple epoch-specific files.

**Implementation:**
- Saves best model as `best_model.pt` (single file, no epoch number)
- Replaces previous best model file when new best is found
- Includes optimizer and scheduler state in checkpoint
- Saves when validation loss improves (or train loss if no validation)

**Files Modified:**
- `aam/training/trainer.py` - Modified checkpoint saving logic

### PYT-8.3: Early Stopping Default to 10 Epochs ✅
**Status:** Completed

Changed default early stopping patience from 50 epochs to 10 epochs for faster iteration.

**Implementation:**
- Updated `trainer.py` default `early_stopping_patience` to 10
- Updated CLI defaults to be consistent (both `train` and `pretrain` use 10)
- Backward compatible - users can override with `--patience` flag

**Files Modified:**
- `aam/training/trainer.py` - Changed default to 10
- `aam/cli.py` - Updated pretrain command default to 10

### PYT-8.4: Validation Prediction Plots ✅
**Status:** Completed

Creates validation prediction plots showing predicted vs actual values with linear fit, R² metric, and 1:1 reference line. Saves plots to both TensorBoard and disk files.

**Features:**
- Scatter plot with predicted vs actual values
- Linear fit line with R² displayed
- 1:1 reference line for perfect prediction
- Confusion matrix for classification tasks
- Plots saved to disk as PNG files
- Plots logged to TensorBoard
- Only created when validation improves

**Files Modified:**
- `aam/training/trainer.py` - Added plot creation methods
- Added matplotlib dependency

### PYT-8.9: Fix NaN in Nucleotide Predictions ✅
**Status:** Completed

Fixed NaN values appearing in nucleotide predictions during pretraining when using `--token-limit` with gradient accumulation. The issue was caused by all-padding sequences causing NaN in transformer attention mechanisms.

**Root Cause:**
- All-padding sequences (all zero tokens) cause NaN in TransformerEncoder
- `softmax(all -inf)` produces NaN
- NaN propagates through model to downstream layers

**Solution:**
1. **AttentionPooling**: Handle all-padding sequences by setting scores to 0.0 before softmax
2. **ASVEncoder**: Mask NaN from transformer output for all-padding sequences
3. **Dataset**: Validate samples have at least one ASV with count > 0
4. **Loss Function**: Safe tensor statistics formatting for error messages

**Files Modified:**
- `aam/models/attention_pooling.py` - Handle all-padding sequences
- `aam/models/asv_encoder.py` - Mask NaN from transformer
- `aam/data/dataset.py` - Add sample validation
- `aam/training/losses.py` - Safe tensor statistics

## Related Features

### PYT-8.5: Shuffled Batches for UniFrac ✅
Support for shuffled batches in UniFrac distance extraction.

### PYT-8.6: Base Loss Shape Mismatch Fix ✅
Fixed shape mismatch issues for variable batch sizes in pretrain mode.

### PYT-8.7: Model NaN Issue and Gradient Clipping ✅
Fixed model NaN issues and added gradient clipping support.

### PYT-8.8: Start Token Addition ✅
Added START_TOKEN to prevent all-padding sequence NaN issues.

### PYT-8.10: Progress Bar Updates ✅
Updated training progress bar and renamed base_loss to unifrac_loss.

### PYT-8.11: Learning Rate Schedulers ✅
Explored and implemented multiple learning rate schedulers (warmup_cosine, cosine, plateau, onecycle).

## Summary

All training features from Phase 8 have been completed and are in production use. These enhancements improve training stability, usability, and debugging capabilities.
