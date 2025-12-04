# PyTorch Porting Tickets

**Priority**: MEDIUM - Feature Enhancements  
**Status**: Not Started

This document contains tickets for implementing feature enhancements for the PyTorch port of AAM.

---

## Phase 8: Feature Enhancements

### PYT-8.1: Implement TensorBoard Train/Val Overlay Verification
**Priority:** LOW | **Effort:** Low | **Status:** ✅ Completed

**Description:**
Verify that TensorBoard train/val metrics automatically overlay correctly for easy comparison. This is primarily a verification and documentation task as TensorBoard already supports automatic overlay.

**Files Modified:**
- Verified `aam/training/trainer.py` metric naming consistency
- TensorBoard logging already creates proper tags for overlay

**Acceptance Criteria:**
- [x] Verify current TensorBoard logging creates proper tags for overlay
- [x] Test overlay functionality in TensorBoard UI
- [x] Document overlay usage in README or training guide
- [x] Ensure consistent metric names between train/val (verify existing implementation)
- [x] Add note in documentation about how to use TensorBoard overlay feature

**Implementation Notes:**
- TensorBoard automatically overlays metrics with same base name but different prefixes
- Users select both `train/{metric}` and `val/{metric}` in TensorBoard UI to see overlay
- No code changes required - this is primarily documentation/verification
- Verified that all metrics (losses, regression metrics, classification metrics, count metrics) can be overlaid
- Current implementation already uses consistent naming: `train/{metric}` and `val/{metric}`

**Dependencies:** PYT-4.4

**Estimated Time:** 1-2 hours
**Actual Time:** ~1 hour

---

### PYT-8.2: Implement Single Best Model File Saving
**Priority:** MEDIUM | **Effort:** Low | **Status:** ✅ Completed

**Description:**
Modify checkpoint saving to keep only the single best model file, replacing previous best model instead of saving multiple epoch-specific files.

**Files Modified:**
- `aam/training/trainer.py` - Modified checkpoint saving logic to save single `best_model.pt`
- `tests/test_trainer.py` - Added tests for single best model saving

**Acceptance Criteria:**
- [x] Modify `train()` method to save single `best_model.pt` file (no epoch number)
- [x] Remove epoch number from best model filename
- [x] Ensure old best model is replaced (not accumulated)
- [x] Test checkpoint saving/loading with best model
- [x] Update documentation to reflect single best model file
- [x] Verify resume from checkpoint still works
- [x] Unit tests pass (add tests for single best model file)

**Implementation Notes:**
- Changed checkpoint filename from `best_model_epoch_{epoch}.pt` to `best_model.pt`
- Added logic to remove old best model file using `Path.unlink()` before saving new one
- Keep optimizer and scheduler state in checkpoint (unchanged)
- Save when validation loss improves (or when training without validation, use train loss)
- Added support for saving best model when training without validation loader
- `load_checkpoint()` can load `best_model.pt` (no changes needed)
- Final model save remains separate for comparison purposes (unchanged)
- Added three new tests: `test_single_best_model_file`, `test_best_model_replacement`, `test_load_best_model_checkpoint`
- Updated `test_resume_training` to use `best_model.pt` instead of generic checkpoint files
- All 41 tests in `test_trainer.py` pass

**Dependencies:** PYT-4.2

**Estimated Time:** 2-3 hours
**Actual Time:** ~2 hours

---

### PYT-8.3: Change Early Stopping Default to 10 Epochs
**Priority:** MEDIUM | **Effort:** Low | **Status:** ✅ Completed

**Description:**
Change the default early stopping patience from 50 epochs to 10 epochs for faster iteration and consistency between CLI commands.

**Files Modified:**
- `aam/training/trainer.py` - Updated default `early_stopping_patience` to 10
- `aam/cli.py` - Updated pretrain command `--patience` default to 10
- `tests/test_trainer.py` - Added test to verify default value
- `tests/test_cli.py` - Added tests to verify CLI command defaults

**Acceptance Criteria:**
- [x] Change `trainer.py` default `early_stopping_patience` from 50 to 10
- [x] Update `cli.py` pretrain command `--patience` default from 50 to 10
- [x] Verify train command default is 10 (already was)
- [x] Test early stopping with new default (triggers after 10 epochs without improvement)
- [x] Test that `--patience` flag still works to override default
- [x] Verify both train and pretrain commands use same default
- [x] Update documentation if needed
- [x] Unit tests pass

**Implementation Notes:**
- Changed trainer default from 50 to 10 to match CLI train command
- Changed CLI pretrain default from 50 to 10 for consistency
- Users can still override with `--patience` flag (backward compatible)
- Added `test_train_default_early_stopping_patience()` to verify trainer default using `inspect.signature()`
- Added `test_train_command_default_patience()` and `test_pretrain_command_default_patience()` to verify CLI defaults
- All defaults now consistently set to 10 epochs
- 10 epochs is more reasonable default for faster iteration

**Dependencies:** PYT-4.2

**Estimated Time:** 1 hour
**Actual Time:** ~1 hour

---

### PYT-8.4: Implement Validation Prediction Plots
**Priority:** MEDIUM | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Create validation prediction plots showing predicted vs actual values with linear fit, R² metric, and 1:1 reference line. Save plots to both TensorBoard and disk files. Support both regression and classification tasks.

**Files Modified:**
- `aam/training/trainer.py` - Added plot creation methods and integration
- `pyproject.toml` - Added matplotlib dependency
- `environment.yml` - Added matplotlib dependency
- `tests/test_trainer.py` - Added 10 comprehensive tests for plot generation

**Acceptance Criteria:**
- [x] Add matplotlib dependency to `pyproject.toml` and `environment.yml`
- [x] Create `_create_prediction_plot()` method for regression tasks
- [x] Create `_create_confusion_matrix_plot()` method for classification tasks
- [x] Integrate plot creation into `train()` method (when validation improves)
- [x] Save plots to disk as PNG files in `{checkpoint_dir}/plots/` directory
- [x] Log plots to TensorBoard using `add_figure()`
- [x] Add `save_plots` parameter to `train()` method (default: True)
- [x] Create plots directory automatically
- [x] Test with regression tasks (scatter plot with linear fit, R², 1:1 line)
- [x] Test with classification tasks (confusion matrix with metrics)
- [x] Test plot saving and TensorBoard logging
- [x] Test that plots are only created when validation improves
- [x] Test with `save_plots=False` to disable plotting
- [x] Unit tests pass (51 tests total, all passing)

**Implementation Notes:**
- Implemented `_create_prediction_plot()` for regression: scatter plot with predicted vs actual, linear fit using `numpy.polyfit`, R² value in title, 1:1 reference line (dashed gray)
- Implemented `_create_confusion_matrix_plot()` for classification: confusion matrix heatmap using `sklearn.metrics.confusion_matrix`, displays accuracy, precision, recall, F1 in text box
- Modified `validate_epoch()` to optionally return predictions and targets via `return_predictions` parameter
- Integrated plot creation into `train()` method: plots created when validation improves (new best model)
- Plots saved to `{checkpoint_dir}/plots/pred_vs_actual_best.png` (replaces previous best plot)
- TensorBoard logging: plots logged to `validation/prediction_plot` tag
- Figure size: 8x6 inches, DPI: 100
- Uses `plt.close()` after saving/logging to free memory
- All 10 new tests pass, no regressions in existing tests

**Dependencies:** PYT-4.2

**Estimated Time:** 4-6 hours
**Actual Time:** ~4 hours

---

### PYT-8.5: Support Shuffled Batches for UniFrac Distance Extraction
**Priority:** MEDIUM | **Effort:** Medium | **Status:** Not Started

**Description:**
Fix batch-level UniFrac distance extraction to properly handle shuffled batches. Currently, the `collate_fn` assumes sample order matches dataset order when extracting batch-level pairwise distances `[batch_size, batch_size]` from dataset-level distances. When batches are shuffled (which is the default for training), this assumption breaks and incorrect distances are extracted.

**Files to Modify:**
- `aam/data/dataset.py` - Update `collate_fn` to properly map batch sample_ids to dataset indices
- `tests/test_dataset.py` - Add tests for shuffled batch distance extraction
- `tests/test_integration.py` - Add integration tests for shuffled batches

**Acceptance Criteria:**
- [ ] Update `collate_fn` to properly extract batch-level distances for shuffled batches
- [ ] Create mapping from batch sample_ids to their positions in dataset's distance matrix
- [ ] Extract correct `[batch_size, batch_size]` pairwise distance matrix for batch samples
- [ ] Handle both unweighted UniFrac (pairwise matrix) and Faith PD (per-sample values)
- [ ] Test with shuffled batches (default DataLoader behavior)
- [ ] Test with non-shuffled batches (validation sets)
- [ ] Test with different batch sizes
- [ ] Verify extracted distances match expected pairwise distances
- [ ] Ensure backward compatibility with existing code
- [ ] Unit tests pass (add comprehensive tests for shuffled batch scenarios)
- [ ] Integration tests verify correct loss computation with shuffled batches

**Implementation Notes:**
- Current issue: `collate_fn` assumes first `batch_size` columns correspond to batch samples
- Solution: Map batch `sample_ids` to their indices in `dataset.sample_ids`, then extract corresponding columns from distance matrix
- For unweighted UniFrac: Extract `[batch_size, batch_size]` submatrix where `distances[i, j]` is distance from batch sample `i` to batch sample `j`
- For Faith PD: Already per-sample, so stacking should work correctly
- Consider passing dataset reference to collate_fn or storing sample_id-to-index mapping in dataset
- Alternative: Use `UniFracComputer.extract_batch_distances()` in collate_fn if we can pass DistanceMatrix
- Need to ensure distance matrix symmetry is preserved (dist[i,j] == dist[j,i])
- Verify diagonal is zero (distance from sample to itself)

**Dependencies:** None (can be implemented independently)

**Estimated Time:** 3-4 hours

---

## Summary

**Total Estimated Time:** 11-16 hours

**Implementation Order:**
1. ✅ PYT-8.3: Change Early Stopping Default to 10 Epochs (1 hour) - Completed
2. ✅ PYT-8.2: Implement Single Best Model File Saving (2-3 hours) - Completed
3. ✅ PYT-8.1: Implement TensorBoard Train/Val Overlay Verification (1-2 hours) - Completed
4. ✅ PYT-8.4: Implement Validation Prediction Plots (4-6 hours) - Completed
5. ⏳ PYT-8.5: Support Shuffled Batches for UniFrac Distance Extraction (3-4 hours) - Not Started

**Notes:**
- All tickets are independent and can be implemented in any order
- PYT-8.3 completed - early stopping defaults now consistent at 10 epochs
- PYT-8.2 completed - single best model file saving implemented
- PYT-8.1 completed - TensorBoard overlay verification documented
- PYT-8.4 completed - validation prediction plots with matplotlib dependency added
- PYT-8.5 not started - needed to fix UniFrac loss computation with shuffled batches
- Follow the workflow in `.agents/workflow.md` for implementation
