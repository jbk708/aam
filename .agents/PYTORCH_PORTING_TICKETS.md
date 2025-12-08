# PyTorch Porting Tickets

**Priority**: MEDIUM - Feature Enhancements  
**Status**: In Progress

This document contains tickets for implementing feature enhancements for the PyTorch port of AAM.

---

## Phase 9: UniFrac Underfitting Fixes

### PYT-8.16b: Refactor UniFrac Distance Prediction to Match TensorFlow Approach
**Priority:** HIGH | **Effort:** High | **Status:** ✅ Completed

**Description:**
Refactor UniFrac distance prediction to match TensorFlow implementation by computing pairwise distances from embeddings (Euclidean distance) instead of predicting distances directly. This architectural change will eliminate sigmoid saturation issues, mode collapse, and boundary clustering problems identified in the investigation.

**Problem:**
- **Investigation completed (PYT-8.16a)**: Found that hard clipping caused 53.59% boundary clustering (40% at 0.0, 13.59% at 1.0)
- **Sigmoid fix attempted**: Replaced clipping with sigmoid, but caused mode collapse to 0.5 predictions
- **Root cause identified**: PyTorch uses fundamentally different architecture than TensorFlow:
  - **TensorFlow**: Embeddings → Euclidean distance computation (natural, unbounded)
  - **PyTorch**: Embeddings → Linear head → Direct predictions → Sigmoid constraint (artificial, bounded)
- **Architectural mismatch**: Direct prediction approach requires sigmoid/clipping, causing training issues
- **Solution**: Match TensorFlow approach by computing distances from embeddings instead of predicting directly

**Acceptance Criteria:**

**Investigation Phase (PYT-8.16a) - ✅ Completed:**
- [x] Create analysis script to investigate prediction distribution during inference
- [x] Count and analyze predictions at 0.0 and 1.0 boundaries
- [x] Compare prediction distribution to actual UniFrac distance distribution
- [x] Investigate clipping behavior in model forward pass and loss computation
- [x] Check if predictions are being forced to boundaries by clipping operations
- [x] Analyze model outputs before clipping (raw predictions)
- [x] Identify root cause: Hard clipping forced 53.59% to boundaries
- [x] Document findings in analysis report
- [x] Create visualizations of prediction vs actual distributions
- [x] Compare TensorFlow vs PyTorch implementations

**Implementation Phase (PYT-8.16b) - ✅ Completed:**
- [x] Remove direct distance prediction head (`output_head` for UniFrac)
- [x] Modify `SequenceEncoder` to return embeddings directly (no sigmoid/clipping)
- [x] Implement pairwise distance computation from embeddings (Euclidean distance)
- [x] Update loss function to compute distances from embeddings
- [x] Update CLI to handle new architecture (remove `base_output_dim` for UniFrac)
- [x] Update dataset/collate to work with embedding-based approach
- [x] Update trainer to compute distances from embeddings
- [x] Remove sigmoid activation and clipping from forward pass
- [x] Update tests to reflect new architecture
- [x] Verify training works correctly with new approach
- [x] Test that predictions are continuous (no boundary clustering)
- [x] Verify no mode collapse (predictions not all 0.5)
- [x] Document architectural changes
- [x] Fix NaN issue in attention pooling by handling all-padding sequences before transformer

**Implementation Notes:**

**Investigation Findings (PYT-8.16a - ✅ Completed):**
- Root cause: Hard clipping (`torch.clamp`) forced 53.59% of predictions to boundaries (40% at 0.0, 13.59% at 1.0)
- Raw predictions were continuous (0% at boundaries), but clipping created artificial clustering
- Sigmoid fix attempted but caused mode collapse to 0.5 predictions
- **Key discovery**: TensorFlow uses different architecture - computes distances from embeddings, not direct predictions
- See `debug/BOUNDARY_PREDICTION_ANALYSIS.md` and `debug/TENSORFLOW_VS_PYTORCH_COMPARISON.md` for details

**Implementation Results (PYT-8.16b - ✅ Completed):**
- ✅ Removed `output_head` for UniFrac encoder type in `SequenceEncoder`
- ✅ Modified `SequenceEncoder` to return embeddings directly for UniFrac (no sigmoid/clipping)
- ✅ Implemented `compute_pairwise_distances()` function in `losses.py` to compute Euclidean distances from embeddings
- ✅ Updated `compute_base_loss()` to compute distances from embeddings when `encoder_type == "unifrac"`
- ✅ Updated CLI to handle new architecture (removed `base_output_dim` requirement for UniFrac)
- ✅ Updated all tests to reflect new architecture (all passing)
- ✅ Fixed NaN issue in attention pooling by handling all-padding sequences before transformer in `ASVEncoder`
- ✅ All integration tests passing (13/13)
- ✅ All attention pooling tests passing (17/17)
- ✅ Created debug script `debug/investigate_attention_pooling_nan.py` for future debugging
- **Key Fix**: Handled all-padding sequences (mask sum = 0) in `ASVEncoder` by skipping transformer and setting embeddings to zero, preventing NaN propagation

**Architectural Change Required:**
1. **Remove direct prediction head** for UniFrac:
   - Remove `output_head` when `encoder_type == "unifrac"`
   - Return embeddings directly from `SequenceEncoder.forward()`
   
2. **Implement pairwise distance computation**:
   - Add function to compute Euclidean distances from embeddings: `sqrt(||a - b||^2)`
   - Compute pairwise distance matrix: `[batch_size, batch_size]`
   - Similar to TensorFlow's `_pairwise_distances()` function
   
3. **Update loss computation**:
   - Modify `compute_base_loss()` to compute distances from embeddings when `encoder_type == "unifrac"`
   - Remove sigmoid/clipping (distances naturally ≥ 0)
   - Keep diagonal masking (already implemented)
   
4. **Update CLI and dataset**:
   - Remove `base_output_dim = batch_size` for UniFrac (not needed)
   - Update model initialization to not create `output_head` for UniFrac
   - Ensure embeddings are returned and distances computed in loss

**Files to Modify:**
- `aam/models/sequence_encoder.py` - Remove output_head for UniFrac, return embeddings directly
- `aam/training/losses.py` - Add pairwise distance computation, update `compute_base_loss()`
- `aam/cli.py` - Update model initialization (remove base_output_dim for UniFrac)
- `aam/data/dataset.py` - Verify compatibility with embedding-based approach
- `tests/test_sequence_encoder.py` - Update tests for new architecture
- `tests/test_losses.py` - Add tests for pairwise distance computation
- `tests/test_trainer.py` - Update tests if needed

**Benefits:**
- Matches TensorFlow implementation exactly
- Eliminates sigmoid saturation issues
- No mode collapse (no sigmoid needed)
- No boundary clustering (no clipping needed)
- Natural distance computation (Euclidean from embeddings)
- Better gradient flow (no sigmoid/clipping operations)

**Dependencies:** PYT-8.12, PYT-8.14 (completed), PYT-8.16a (investigation completed)

**Estimated Time:** 6-8 hours (architectural refactoring)

---

## Summary

**Total Estimated Time Remaining:** 0 hours (All Phase 9 tickets completed)

**Implementation Order:**

### Phase 8: Feature Enhancements (✅ All Completed)
1. ✅ PYT-8.1: Implement TensorBoard Train/Val Overlay Verification (1-2 hours) - Completed
2. ✅ PYT-8.2: Implement Single Best Model File Saving (2-3 hours) - Completed
3. ✅ PYT-8.3: Change Early Stopping Default to 10 Epochs (1 hour) - Completed
4. ✅ PYT-8.4: Implement Validation Prediction Plots (4-6 hours) - Completed
5. ✅ PYT-8.5: Support Shuffled Batches for UniFrac Distance Extraction (3-4 hours) - Completed
6. ✅ PYT-8.6: Fix Base Loss Shape Mismatch for Variable Batch Sizes in Pretrain Mode (2-3 hours) - Completed
7. ✅ PYT-8.7: Fix Model NaN Issue and Add Gradient Clipping (4-6 hours) - Completed
8. ✅ PYT-8.8: Add Start Token to Prevent All-Padding Sequence NaN Issues (3-4 hours) - Completed
9. ✅ PYT-8.9: Fix NaN in Nucleotide Predictions During Pretraining with Token Limit (3-4 hours) - Completed
10. ✅ PYT-8.10: Update Training Progress Bar and Rename base_loss to unifrac_loss (2-3 hours) - Completed
11. ✅ PYT-8.11: Explore Learning Rate Optimizers and Schedulers (4-6 hours) - Completed

### Phase 9: UniFrac Underfitting Fixes
12. ✅ PYT-8.12: Mask Diagonal in UniFrac Loss Computation (1-2 hours) - Completed
13. ✅ PYT-8.13: Investigate Zero-Distance Samples in UniFrac Data (2-3 hours) - Completed
14. ✅ PYT-8.14: Implement Bounded Regression Loss for UniFrac Distances (3-4 hours) - Completed
15. ❌ PYT-8.15: Implement Weighted Loss for Zero-Distance UniFrac Pairs (1-2 hours) - Cancelled (zero distances too rare per PYT-8.13 findings)
16. ✅ PYT-8.16a: Investigate Prediction Clustering at 0.0 and 1.0 During Inference (4-6 hours) - Completed
17. ✅ **PYT-8.16b: Refactor UniFrac Distance Prediction to Match TensorFlow Approach (6-8 hours) - Completed**

**Recommended Implementation Order for UniFrac Fixes:**
1. ✅ **PYT-8.12** (diagonal masking) - Foundation fix, completed
2. ✅ **PYT-8.13** (zero-distance investigation) - Completed, found zero distances are extremely rare
3. ✅ **PYT-8.14** (bounded loss) - Completed, added clipping to [0, 1]
4. ❌ **PYT-8.15** (weighted loss) - Cancelled per PYT-8.13 findings
5. ✅ **PYT-8.16a** (investigate boundary prediction clustering) - Completed, identified root cause and architectural mismatch
6. ✅ **PYT-8.16b** (refactor to TensorFlow approach) - Completed, architectural change to match TensorFlow implementation

**Notes:**
- All Phase 8 tickets completed
- Phase 9 tickets address critical underfitting issue (R² = 0.0455) identified in model performance analysis
- See `_design_plan/19_unifrac_underfitting_analysis.md` for detailed analysis and rationale
- See `debug/BOUNDARY_PREDICTION_ANALYSIS.md` and `debug/TENSORFLOW_VS_PYTORCH_COMPARISON.md` for investigation findings
- Follow the workflow in `.agents/workflow.md` for implementation
