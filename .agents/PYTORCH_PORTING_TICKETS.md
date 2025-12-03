# PyTorch Porting Tickets

**Priority**: MEDIUM - Feature Enhancements  
**Status**: Not Started

This document contains tickets for implementing feature enhancements for the PyTorch port of AAM.

---

## Phase 8: Feature Enhancements

### PYT-8.1: Implement TensorBoard Train/Val Overlay Verification
**Priority:** LOW | **Effort:** Low | **Status:** Not Started

**Description:**
Verify that TensorBoard train/val metrics automatically overlay correctly for easy comparison. This is primarily a verification and documentation task as TensorBoard already supports automatic overlay.

**Files to be Modified:**
- `aam/training/trainer.py` - Verify metric naming consistency
- Documentation - Add usage guide for overlay feature

**Acceptance Criteria:**
- [ ] Verify current TensorBoard logging creates proper tags for overlay
- [ ] Test overlay functionality in TensorBoard UI
- [ ] Document overlay usage in README or training guide
- [ ] Ensure consistent metric names between train/val (verify existing implementation)
- [ ] Add note in documentation about how to use TensorBoard overlay feature

**Implementation Notes:**
- TensorBoard automatically overlays metrics with same base name but different prefixes
- Users select both `train/{metric}` and `val/{metric}` in TensorBoard UI to see overlay
- No code changes required - this is primarily documentation/verification
- Verify that all metrics (losses, regression metrics, classification metrics, count metrics) can be overlaid

**Dependencies:** PYT-4.4

**Estimated Time:** 1-2 hours

---

### PYT-8.2: Implement Single Best Model File Saving
**Priority:** MEDIUM | **Effort:** Low | **Status:** Not Started

**Description:**
Modify checkpoint saving to keep only the single best model file, replacing previous best model instead of saving multiple epoch-specific files.

**Files to be Modified:**
- `aam/training/trainer.py` - Modify checkpoint saving logic
- `tests/test_trainer.py` - Add tests for single best model saving

**Acceptance Criteria:**
- [ ] Modify `train()` method to save single `best_model.pt` file (no epoch number)
- [ ] Remove epoch number from best model filename
- [ ] Ensure old best model is replaced (not accumulated)
- [ ] Test checkpoint saving/loading with best model
- [ ] Update documentation to reflect single best model file
- [ ] Verify resume from checkpoint still works
- [ ] Unit tests pass (add tests for single best model file)

**Implementation Notes:**
- Save best model as `best_model.pt` (single file, no epoch number)
- Replace previous best model file when new best is found
- Use `Path.unlink()` to remove old file before saving, or just overwrite
- Keep optimizer and scheduler state in checkpoint
- Save when validation loss improves (or when training without validation, use train loss)
- Ensure `load_checkpoint()` can load `best_model.pt`
- Final model save can remain separate for comparison purposes

**Dependencies:** PYT-4.2

**Estimated Time:** 2-3 hours

---

### PYT-8.3: Change Early Stopping Default to 10 Epochs
**Priority:** MEDIUM | **Effort:** Low | **Status:** Not Started

**Description:**
Change the default early stopping patience from 50 epochs to 10 epochs for faster iteration and consistency between CLI commands.

**Files to be Modified:**
- `aam/training/trainer.py` - Update default `early_stopping_patience` to 10
- `aam/cli.py` - Update pretrain command `--patience` default to 10
- `tests/test_trainer.py` - Verify new default works correctly

**Acceptance Criteria:**
- [ ] Change `trainer.py` default `early_stopping_patience` from 50 to 10
- [ ] Update `cli.py` pretrain command `--patience` default from 50 to 10
- [ ] Verify train command default is 10 (should already be)
- [ ] Test early stopping with new default (triggers after 10 epochs without improvement)
- [ ] Test that `--patience` flag still works to override default
- [ ] Verify both train and pretrain commands use same default
- [ ] Update documentation if needed
- [ ] Unit tests pass

**Implementation Notes:**
- Current state: CLI train command has `--patience` default=10, CLI pretrain command has `--patience` default=50, trainer has `early_stopping_patience: int = 50`
- Change trainer default to 10 to match CLI train command
- Change CLI pretrain default to 10 for consistency
- Users can still override with `--patience` flag (backward compatible)
- 10 epochs is more reasonable default for faster iteration

**Dependencies:** PYT-4.2

**Estimated Time:** 1 hour

---

### PYT-8.4: Implement Validation Prediction Plots
**Priority:** MEDIUM | **Effort:** Medium | **Status:** Not Started

**Description:**
Create validation prediction plots showing predicted vs actual values with linear fit, R² metric, and 1:1 reference line. Save plots to both TensorBoard and disk files. Support both regression and classification tasks.

**Files to be Created/Modified:**
- `aam/training/trainer.py` - Add plot creation methods and integration
- `pyproject.toml` - Add matplotlib dependency
- `environment.yml` - Add matplotlib dependency
- `tests/test_trainer.py` - Add tests for plot generation

**Acceptance Criteria:**
- [ ] Add matplotlib dependency to `pyproject.toml` and `environment.yml`
- [ ] Create `_create_prediction_plot()` method for regression tasks
- [ ] Create `_create_confusion_matrix_plot()` method for classification tasks
- [ ] Integrate plot creation into `train()` method (when validation improves)
- [ ] Save plots to disk as PNG files in `{output_dir}/plots/` directory
- [ ] Log plots to TensorBoard using `add_figure()`
- [ ] Add `save_plots` parameter to `train()` method (default: True)
- [ ] Create plots directory automatically
- [ ] Test with regression tasks (scatter plot with linear fit, R², 1:1 line)
- [ ] Test with classification tasks (confusion matrix with metrics)
- [ ] Test plot saving and TensorBoard logging
- [ ] Test that plots are only created when validation improves
- [ ] Test with `save_plots=False` to disable plotting
- [ ] Unit tests pass

**Implementation Notes:**
- Use `matplotlib` for plotting (standard and lightweight)
- Regression plot: Scatter plot with predicted vs actual, linear fit line, R² value, 1:1 reference line
- Classification plot: Confusion matrix heatmap with accuracy, precision, recall, F1 scores
- Plot only when validation improves (reduces overhead)
- Save plots as `pred_vs_actual_epoch_{epoch}.png` or `pred_vs_actual_best.png`
- Use `plt.close()` after saving to free memory
- Figure size: 8x6 inches, DPI: 100-150
- Use `numpy.polyfit` or `scipy.stats.linregress` for linear fit
- Use `sklearn.metrics.confusion_matrix` for classification
- Consider using matplotlib with seaborn style: `plt.style.use('seaborn-v0_8')`

**Dependencies:** PYT-4.2

**Estimated Time:** 4-6 hours

---

## Summary

**Total Estimated Time:** 8-12 hours

**Implementation Order:**
1. PYT-8.3: Change Early Stopping Default to 10 Epochs (1 hour) - Simplest, no dependencies
2. PYT-8.2: Implement Single Best Model File Saving (2-3 hours) - Straightforward modification
3. PYT-8.1: Implement TensorBoard Train/Val Overlay Verification (1-2 hours) - Documentation/verification
4. PYT-8.4: Implement Validation Prediction Plots (4-6 hours) - Most complex, requires new dependencies

**Notes:**
- All tickets are independent and can be implemented in any order
- PYT-8.3 is recommended first as it's the simplest and improves consistency
- PYT-8.4 requires adding matplotlib dependency
- Follow the workflow in `.agents/workflow.md` for implementation
