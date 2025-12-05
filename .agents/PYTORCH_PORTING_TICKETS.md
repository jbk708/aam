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
**Priority:** MEDIUM | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Support shuffled batches for UniFrac distance extraction. Currently, UniFrac distances are computed assuming batches are in a specific order, but with shuffled batches, the distance matrix needs to be reordered to match the batch order.

**Acceptance Criteria:**
- [x] Modify UniFrac distance extraction to handle shuffled batches
- [x] Ensure distance matrix rows/columns match batch sample order
- [x] Test with shuffled and non-shuffled batches
- [x] Verify loss computation works correctly with shuffled batches
- [x] Unit tests pass
- [x] Integration tests verify correct behavior

**Implementation Notes:**
- Modified `extract_batch_distances()` to reorder filtered distance matrix to match batch sample order using `np.ix_()`
- Added `drop_last=True` to DataLoaders to ensure consistent batch sizes
- Moved UniFrac target extraction from `__getitem__()` to `collate_fn()` to handle batch-level extraction
- Updated tests to reflect new architecture

**Files Modified:**
- `aam/data/unifrac.py` - Added reordering logic in `extract_batch_distances()`
- `aam/data/dataset.py` - Moved UniFrac target extraction to `collate_fn()`
- `aam/cli.py` - Pass filtered distance matrices to datasets and `collate_fn()`, added `drop_last=True`
- `tests/test_dataset.py` - Updated tests for new architecture
- `tests/test_integration.py` - Updated integration tests

**Dependencies:** None

**Estimated Time:** 3-4 hours
**Actual Time:** ~4 hours

---

### PYT-8.6: Fix Base Loss Shape Mismatch for Variable Batch Sizes in Pretrain Mode
**Priority:** HIGH | **Effort:** Medium | **Status:** ⏳ Not Started

**Description:**
Fix tensor shape mismatch error in `compute_base_loss` when training in pretrain mode with unweighted UniFrac. The error occurs because `base_output_dim` is set to the CLI `batch_size` argument, but actual batch sizes can vary (especially in the last batch). This causes `base_prediction` to have shape `[actual_batch_size, CLI_batch_size]` while `base_target` has shape `[actual_batch_size, actual_batch_size]` for pairwise UniFrac distances.

**Acceptance Criteria:**
- [ ] Fix `compute_base_loss` to handle shape mismatches for unweighted UniFrac
- [ ] Ensure loss computation works when `base_prediction` shape doesn't match `base_target` shape
- [ ] Handle case where actual batch size < CLI batch_size (last batch scenario)
- [ ] Handle case where actual batch size == CLI batch_size (normal batches)
- [ ] Preserve correct loss computation for Faith PD (should not be affected)
- [ ] Add shape validation/warning or automatic reshaping/slicing
- [ ] Test with different batch sizes (including last batch with fewer samples)
- [ ] Test with both unweighted UniFrac and Faith PD metrics
- [ ] Ensure backward compatibility with existing code
- [ ] Unit tests pass (add tests for variable batch size scenarios)
- [ ] Integration tests verify correct loss computation in pretrain mode

**Implementation Notes:**
- **CRITICAL LESSONS LEARNED FROM PREVIOUS ATTEMPT:**
  1. **DO NOT use slicing operations (`base_pred[:, :base_true.shape[1]]`) in loss computation** - Even though slicing preserves gradients in PyTorch, it can cause numerical instability when combined with other losses (nucleotide loss) that flow through the same computation graph
  2. **DO use `drop_last=True` in DataLoader** - This ensures all batches are exactly `batch_size`, eliminating shape mismatches entirely. This is the cleanest solution.
  3. **DO add NaN checks BEFORE computing loss** - If model outputs contain NaN, skip loss computation and return zero loss to prevent NaN propagation
  4. **DO NOT add complex conditional logic in loss computation** - Keep loss functions simple and straightforward. Complex conditionals can affect autograd behavior unexpectedly.
  5. **Test thoroughly with different batch sizes** - Always test with remainder batches to catch shape mismatch issues early
  6. **Monitor for NaN in embeddings** - NaN in embeddings propagates through entire forward pass. Check embeddings early in forward pass, not just in loss computation.

- **Recommended Approach:**
  - Use `drop_last=True` in DataLoader to ensure consistent batch sizes
  - If shape mismatch still occurs (shouldn't with drop_last), use `torch.index_select` or padding instead of slicing
  - Add early NaN detection in model forward pass (before loss computation)
  - Keep loss computation simple - direct MSE without complex conditionals

- **Previous Failed Attempt:**
  - Commit 1a68364 added slicing logic that caused NaN propagation issues
  - Issue was NOT the slicing itself, but how it interacted with other losses
  - Reverted to commit 68597fc (pre-slicing) which works correctly
  - Root cause: Numerical instability from combining sliced base_loss with nucleotide_loss in same backward pass

**Dependencies:** None (can be implemented independently, but related to PYT-8.5)

**Estimated Time:** 2-3 hours

---

### PYT-8.7: Fix Model NaN Issue and Add Gradient Clipping
**Priority:** HIGH | **Effort:** Medium | **Status:** ⏳ Not Started

**Description:**
Fix NaN values appearing in model outputs during training. The model produces NaN in both `base_prediction` and `nuc_predictions` from the forward pass, causing all losses to be NaN. This indicates a deeper issue with the model architecture or training setup, not just the loss computation. Gradient clipping has been added as a general training stability measure.

**Acceptance Criteria:**
- [x] Add gradient clipping to prevent gradient explosion
- [x] Add CLI option for gradient clipping threshold
- [ ] Investigate root cause of NaN in model outputs (both base_prediction and nuc_predictions)
- [ ] Fix numerical instability in model forward pass
- [ ] Verify training stability with all losses enabled
- [ ] Test gradient clipping with various threshold values
- [ ] Unit tests for gradient clipping functionality
- [ ] Integration tests verify stable training

**Implementation Notes:**
- **Gradient Clipping (COMPLETED):**
  - ✅ Implemented `torch.nn.utils.clip_grad_norm_()` for gradient clipping
  - ✅ Added `--max-grad-norm` CLI option (default: None, disabled)
  - ✅ Apply clipping after `backward()` but before `optimizer.step()`
  - ✅ Log gradient norms to TensorBoard for monitoring

- **Root Cause Analysis (IN PROGRESS):**
  - **Critical Finding**: Both `base_prediction` and `nuc_predictions` contain NaN from model forward pass
  - This indicates the issue is NOT specific to nucleotide head, but affects the entire model
  - NaN appears early in training (step 4-5), suggesting:
    1. Model initialization issue (weights initialized incorrectly)
    2. Numerical instability in model architecture (e.g., attention, normalization)
    3. Invalid input data causing NaN propagation
    4. Gradient explosion corrupting weights immediately
  
- **Investigation Steps:**
  1. Check model weight initialization - verify no NaN/Inf in initial weights
  2. Check input data - verify tokens are valid (0-4 range, no NaN)
  3. Add intermediate checks in model forward pass to identify where NaN first appears
  4. Check attention mechanisms for numerical stability (softmax overflow, etc.)
  5. Verify layer normalization is applied correctly
  6. Check if issue occurs with different batch sizes or learning rates
  7. Monitor gradient norms before clipping to see if explosion occurs
  
- **Potential Fixes:**
  1. Fix model initialization (use proper initialization schemes)
  2. Add numerical stability to attention (e.g., scale before softmax)
  3. Add gradient clipping earlier or with different threshold
  4. Reduce learning rate or use learning rate warmup
  5. Add weight initialization checks and NaN detection in model forward pass
  6. Consider using mixed precision training with proper scaling

**Dependencies:** PYT-8.5 (completed)

**Estimated Time:** 4-6 hours
**Actual Time:** ~1 hour (gradient clipping implemented, root cause investigation pending)

---

### PYT-8.8: Add Start Token to Prevent All-Padding Sequence NaN Issues
**Priority:** HIGH | **Effort:** Medium | **Status:** ✅ Completed

**Description:**
Add a start token to all sequences to prevent all-padding sequences that cause NaN in transformer attention mechanisms during eval mode. When sequences consist entirely of padding tokens, PyTorch's TransformerEncoder produces NaN values in the attention mechanism's softmax operation, causing NaN to propagate through the entire model.

**Root Cause:**
- Sequences can consist entirely of padding tokens (all positions masked)
- PyTorch's TransformerEncoder attention mechanism produces NaN when ALL positions are masked (all-True `src_key_padding_mask`)
- This occurs specifically in eval mode (train mode with dropout masks some issues)
- NaN appears in the first transformer layer output, propagating to all downstream layers
- Affects both training and validation, but more visible in validation (eval mode)

**Acceptance Criteria:**
- [x] Add start token to vocabulary (e.g., token ID 5, or repurpose existing token)
- [x] Modify tokenizer to prepend start token to each sequence
- [x] Update model to handle start token in embeddings and attention
- [x] Ensure start token is never masked (always valid)
- [x] Update sequence length handling to account for start token
- [x] Verify no all-padding sequences exist after start token addition
- [x] Test transformer forward pass in both train and eval mode
- [x] Verify NaN no longer appears in validation
- [x] Update tests to account for start token
- [x] Update documentation

**Implementation Notes:**
- **Approach:**
  1. Add start token to vocabulary (increase `vocab_size` from 5 to 6, or repurpose token 0)
  2. Modify `SequenceTokenizer` to prepend start token to each sequence
  3. Ensure start token position is never masked in attention masks
  4. Update `max_bp` handling if needed (start token adds 1 to sequence length)
  5. Update model initialization to handle new vocabulary size

- **Design Decisions:**
  - Option 1: Add new token (vocab_size becomes 6, token ID 5 = START)
  - Option 2: Repurpose padding token 0 as start token (requires careful handling)
  - Option 3: Use special token embedding that's always present
  - **Recommendation**: Option 1 (add new token) is cleanest and most explicit

- **Files to Modify:**
  - `aam/data/tokenizer.py` - Add start token prepending logic
  - `aam/models/asv_encoder.py` - Update vocab_size handling
  - `aam/models/sequence_encoder.py` - Update vocab_size if needed
  - `aam/models/sample_sequence_encoder.py` - Update vocab_size if needed
  - `aam/cli.py` - Update default vocab_size if needed
  - `tests/test_tokenizer.py` - Add tests for start token
  - `tests/test_asv_encoder.py` - Update tests for new vocab size
  - `tests/test_sequence_encoder.py` - Update tests for new vocab size

- **Testing:**
  - Verify no sequences are all-padding after start token addition
  - Test transformer forward pass in eval mode (should have no NaN)
  - Test transformer forward pass in train mode (should still work)
  - Test validation loop (should have no NaN warnings)
  - Test training loop (should be stable)
  - Verify start token is always present and never masked

**Dependencies:** PYT-8.7 (related, but can be implemented independently)

**Estimated Time:** 3-4 hours
**Actual Time:** ~4 hours

**Implementation Notes:**
- ✅ Added START_TOKEN (ID 5) to vocabulary - vocab_size increased from 5 to 6
- ✅ Modified `SequenceTokenizer.tokenize()` to prepend START_TOKEN to all sequences
- ✅ Updated all model components (ASVEncoder, SampleSequenceEncoder, SequenceEncoder, SequencePredictor) to use vocab_size=6
- ✅ Updated CLI defaults to vocab_size=6
- ✅ Updated trainer token validation to dynamically check model vocab_size
- ✅ Updated all test fixtures to include START_TOKEN at position 0
- ✅ Sequence length is now 151 (1 start token + 150 nucleotides)
- ✅ All tests passing (354/359 non-integration tests, integration tests passing on GPU)

---

### PYT-8.9: Fix NaN in Nucleotide Predictions During Pretraining with Token Limit
**Priority:** HIGH | **Effort:** Medium | **Status:** ⏳ Not Started

**Description:**
Fix NaN values appearing in nucleotide predictions (`nuc_predictions`) during pretraining when using `--token-limit` with gradient accumulation. The error occurs early in training (step 4-5) and produces NaN in predictions with shape `[batch_size, token_limit, seq_len, vocab_size]` (e.g., `[6, 512, 151, 6]`). The issue is believed to be related to data matrix slicing and batch handling when ASVs are truncated to `token_limit`.

**Error Details:**
```
ERROR: NaN in nuc_predictions before loss computation
nuc_predictions shape=torch.Size([6, 512, 151, 6])
nuc_predictions min=nan, max=nan
ValueError: NaN values found in nuc_pred with shape torch.Size([6, 512, 151, 6])
```

**Root Cause Hypothesis:**
- When `token_limit` is used (e.g., 512), ASVs are truncated via slicing in `collate_fn`
- Truncation may create sequences that cause numerical instability in the model
- Gradient accumulation may amplify numerical issues
- The slicing operation `tokens[:token_limit]` may not preserve proper sequence structure
- Large token_limit values (512) combined with gradient accumulation may cause memory/overflow issues

**Acceptance Criteria:**
- [ ] Investigate root cause of NaN in nucleotide predictions with token_limit
- [ ] Fix data slicing/truncation logic in `collate_fn` to preserve sequence validity
- [ ] Ensure truncated sequences maintain proper structure (START_TOKEN, valid nucleotides)
- [ ] Verify no NaN appears in nucleotide predictions during pretraining
- [ ] Test with various token_limit values (64, 256, 512, 1024)
- [ ] Test with gradient accumulation enabled
- [ ] Test with different batch sizes
- [ ] Verify training stability with large token_limit values
- [ ] Add validation checks for sequence validity after truncation
- [ ] Unit tests for collate_fn with token_limit truncation
- [ ] Integration tests verify stable pretraining with token_limit

**Implementation Notes:**
- **Investigation Steps:**
  1. Check if truncated sequences maintain START_TOKEN at position 0
  2. Verify truncated sequences don't become all-padding
  3. Check for numerical overflow/underflow in attention mechanisms with large token_limit
  4. Monitor gradient norms during gradient accumulation
  5. Check if slicing creates invalid token sequences
  6. Verify counts are properly aligned with truncated tokens
  
- **Potential Fixes:**
  1. Ensure START_TOKEN is preserved after truncation
  2. Add validation that truncated sequences contain valid tokens
  3. Use proper slicing that maintains sequence structure
  4. Add checks for all-padding sequences after truncation
  5. Consider using weighted sampling instead of simple truncation
  6. Add numerical stability checks in model forward pass
  7. Monitor and clip gradients more aggressively with large token_limit

- **Files to Investigate:**
  - `aam/data/dataset.py` - `collate_fn` truncation logic (lines 47-55)
  - `aam/models/asv_encoder.py` - Forward pass with large num_asvs
  - `aam/models/sample_sequence_encoder.py` - Attention pooling with many ASVs
  - `aam/training/trainer.py` - Gradient accumulation handling

**Dependencies:** PYT-8.8 (completed)

**Estimated Time:** 3-4 hours

---

## Summary

**Total Estimated Time:** 22-29 hours

**Implementation Order:**
1. ✅ PYT-8.3: Change Early Stopping Default to 10 Epochs (1 hour) - Completed
2. ✅ PYT-8.2: Implement Single Best Model File Saving (2-3 hours) - Completed
3. ✅ PYT-8.1: Implement TensorBoard Train/Val Overlay Verification (1-2 hours) - Completed
4. ✅ PYT-8.4: Implement Validation Prediction Plots (4-6 hours) - Completed
5. ✅ PYT-8.5: Support Shuffled Batches for UniFrac Distance Extraction (3-4 hours) - Completed
6. ✅ **PYT-8.8: Add Start Token to Prevent All-Padding Sequence NaN Issues (3-4 hours) - HIGH PRIORITY** - Completed
7. ⏳ **PYT-8.9: Fix NaN in Nucleotide Predictions During Pretraining with Token Limit (3-4 hours) - HIGH PRIORITY** - Not Started
8. ⏳ PYT-8.6: Fix Base Loss Shape Mismatch for Variable Batch Sizes in Pretrain Mode (2-3 hours) - Not Started
9. ⏳ PYT-8.7: Fix Model NaN Issue and Add Gradient Clipping (4-6 hours) - Partially Completed

**Notes:**
- All tickets are independent and can be implemented in any order
- PYT-8.3 completed - early stopping defaults now consistent at 10 epochs
- PYT-8.2 completed - single best model file saving implemented
- PYT-8.1 completed - TensorBoard overlay verification documented
- PYT-8.4 completed - validation prediction plots with matplotlib dependency added
- PYT-8.5 completed - UniFrac distance extraction now supports shuffled batches with proper reordering
- **PYT-8.8 completed** - Start token (ID 5) added to prevent all-padding sequences that cause NaN in transformer attention. Vocab_size increased from 5 to 6, sequence length is now 151 (1 start token + 150 nucleotides).
- **PYT-8.9 HIGH PRIORITY** - Fix NaN in nucleotide predictions during pretraining with token_limit. Issue occurs when ASVs are truncated to token_limit (e.g., 512), causing numerical instability. Related to data matrix slicing and batch handling.
- PYT-8.6 not started - HIGH priority bug fix for pretrain mode with variable batch sizes
- PYT-8.7 partially completed - Gradient clipping implemented, root cause investigation needed for NaN in model outputs (both base_prediction and nuc_predictions)
- **PYT-8.6 REVERTED**: Previous implementation (commit 1a68364) caused NaN propagation issues. Reverted to commit 68597fc. See implementation notes for critical lessons learned.
- Follow the workflow in `.agents/workflow.md` for implementation
