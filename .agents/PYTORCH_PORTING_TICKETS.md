# Outstanding Tickets

**Last Updated:** 2025-12-18
**Status:** Phases 8-11 Complete (see `ARCHIVED_TICKETS.md`), PYT-19.1 Complete, PYT-20.1 Complete (MAE for Nucleotide Prediction)

---

## Phase 21: Transfer Learning & Fine-Tuning Fixes (NEW - HIGH PRIORITY)

### PYT-21.2: Fix Pretrained Encoder Loading and Freeze-Base Verification
**Priority:** HIGH | **Effort:** 4-5 hours | **Status:** Complete

**Problem:**
Multiple issues with pretrained encoder loading and freeze-base functionality:
1. `load_pretrained_encoder()` uses `strict=False` with **no logging** of matched/mismatched keys
2. No verification that weights were actually loaded (silently fails on key mismatch)
3. No startup logging of frozen vs trainable parameter counts
4. `nuc_penalty` defaults to 1.0 even with `--freeze-base`, meaning nucleotide loss still contributes to training when the encoder is frozen
5. **Root cause found:** `torch.compile()` adds `_orig_mod.` prefix to all state dict keys, causing 100% key mismatch when loading into non-compiled model

**Evidence:**
When loading a pretrained encoder that achieved NA=97.98% during pretraining, fine-tuning starts with NA=19.49% - indicating the weights weren't loaded correctly.
```
# Expected (if loaded correctly): NA should start high
# Actual: Epoch 1/1000: TL=55945.4088, NL=2.1022, NA=19.49%
```

**Solution:**
1. Strip `_orig_mod.` prefix from checkpoint keys when loading compiled model checkpoints
2. Add detailed logging to `load_pretrained_encoder()`:
   - Log number of matched/loaded keys
   - Warn on any unmatched keys (even with `strict=False`)
   - Log total parameters loaded
   - Detect and report shape mismatches with helpful error messages
3. Add parameter count logging at training start (frozen vs trainable)
4. Auto-set `nuc_penalty=0` when `freeze_base=True`

**Acceptance Criteria:**
- [x] `load_pretrained_encoder()` logs: "Loaded X/Y keys (Z parameters)"
- [x] Warn on mismatched keys: "WARNING: N keys not found in checkpoint: [key1, key2, ...]"
- [x] Warn on unexpected keys: "WARNING: N unexpected keys in checkpoint: [key1, key2, ...]"
- [x] Log at training start: "Frozen: X params, Trainable: Y params (Z%)"
- [x] Auto-set `nuc_penalty=0` when `freeze_base=True`
- [x] Strip `_orig_mod.` prefix for torch.compile() checkpoints
- [x] Raise helpful error on shape mismatch (not silent fail)
- [x] Add test verifying pretrained weights are actually loaded
- [x] Add test for compiled model checkpoint loading

**Expected Impact:**
- Clear visibility into what weights are loaded during transfer learning
- Early detection of configuration mismatches
- Proper fine-tuning behavior with frozen base
- Checkpoints from `--compile-model` runs now load correctly

**Files:** `aam/training/trainer.py`, `aam/cli.py`, `tests/test_trainer.py`

---

### PYT-21.1: Target Loss Improvements and Normalize-Targets Default
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** Not Started

**Problem:**
1. `target_loss` has an implicit weight of 1.0, while `unifrac_loss` and `nuc_loss` have configurable penalties. This inconsistency makes loss tuning difficult.
2. `--normalize-targets` should be the default for regression tasks (better training dynamics).

Current formula:
```python
total_loss = target_loss + count_loss + unifrac_loss * penalty + nuc_loss * nuc_penalty
```

**Solution:**
1. Add `target_penalty` parameter to `MultiTaskLoss` (default: 1.0 for backward compatibility)
2. Add `--target-penalty` CLI flag to `train` command
3. Make `--normalize-targets` the default (add `--no-normalize-targets` to disable)
4. Log individual loss contributions to TensorBoard

**Acceptance Criteria:**
- [ ] Add `target_penalty` parameter to `MultiTaskLoss.__init__()`
- [ ] Update total_loss formula: `target_loss * target_penalty + count_loss + ...`
- [ ] Add `--target-penalty` flag to CLI (default: 1.0)
- [ ] Change `--normalize-targets` to default=True
- [ ] Add `--no-normalize-targets` flag to opt out
- [ ] Add TensorBoard scalars for weighted vs unweighted loss components
- [ ] Update tests for new defaults
- [ ] Document loss weighting strategy in help text

**Expected Impact:**
- Clearer loss contribution visibility for debugging
- Better default training behavior with normalized targets
- Ability to tune target loss weight relative to auxiliary tasks

**Files:** `aam/training/losses.py`, `aam/cli.py`, `aam/training/trainer.py`, `tests/test_losses.py`

---

### PYT-21.3: Regressor Head Optimization
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

**Problem:**
The current target prediction head uses:
- Sigmoid activation bounded to [0,1] (assumes normalized targets)
- Simple linear projection after attention pooling
- No learned temperature or scale adjustment
- No residual connections in target_encoder

This may limit performance for:
- Targets with different scales or distributions
- Multi-target regression where targets have varying ranges
- Tasks where unbounded predictions are appropriate

**Solution:**
Enhance the regressor with configurable improvements:
1. Add `--unbounded-targets` flag to disable sigmoid (for non-normalized targets)
2. Add optional learnable output scaling: `pred = pred * scale + bias` after sigmoid
3. Add optional LayerNorm before final projection
4. Add residual connection option in target_encoder
5. Improve initialization for regression head (Xavier/Kaiming)

**Acceptance Criteria:**
- [ ] Add `--unbounded-targets` flag (disables sigmoid, uses identity activation)
- [ ] Add `--learnable-output-scale` flag with trainable scale/bias parameters
- [ ] Add `--target-layer-norm` flag for LayerNorm before projection
- [ ] Add proper weight initialization for regression head (`nn.init.xavier_uniform_`)
- [ ] Log output statistics during validation (pred min/max/mean/std)
- [ ] Add tests for each configuration option
- [ ] Document when to use each option in CLI help

**Expected Impact:**
- Better out-of-the-box performance for various target distributions
- Flexibility for different regression tasks
- More stable training dynamics

**Files:** `aam/models/sequence_predictor.py`, `aam/cli.py`, `tests/test_sequence_predictor.py`

---

### PYT-21.4: Skip Auxiliary Loss Computation During Fine-Tuning
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Not Started

**Problem:**
When fine-tuning with `--freeze-base`, auxiliary losses (nucleotide, count) still:
1. Compute predictions (wasting memory/compute)
2. Get logged in progress bar (confusing monitoring)

Note: PYT-21.2 already auto-disables `nuc_penalty` when `--freeze-base` is set, so the loss doesn't contribute to training. This ticket addresses the remaining compute/logging overhead.

**Solution:**
1. Add `--skip-auxiliary-predictions` flag (auto-enabled with `--freeze-base`)
2. Skip nucleotide prediction computation when disabled
3. Update progress bar to only show relevant losses
4. Add `--keep-auxiliary-predictions` to force computation if needed for monitoring

**Acceptance Criteria:**
- [ ] Add `--skip-auxiliary-predictions` flag
- [ ] Auto-enable when `--freeze-base` is set
- [ ] Skip `nuc_predictions` computation in model forward when disabled
- [ ] Update progress bar format to exclude disabled losses
- [ ] Document the fine-tuning workflow in help text
- [ ] Add tests for flag behavior

**Expected Impact:**
- Faster fine-tuning (no unnecessary nucleotide prediction computation)
- Cleaner progress bar during fine-tuning
- Reduced memory usage

**Files:** `aam/cli.py`, `aam/models/sequence_predictor.py`, `aam/training/trainer.py`

---

## Phase 20: Self-Supervised Learning Improvements (NEW - HIGH PRIORITY)

### PYT-20.1: Masked Autoencoder for Nucleotide Prediction
**Priority:** HIGH | **Effort:** 4-6 hours | **Status:** Complete

Replace current autoencoding nucleotide loss with masked autoencoding (MAE) to provide meaningful gradient signal throughout training.

**Problem:**
Current nucleotide loss saturates within 1-2 epochs (NL: 0.7319 → 0.0002) because autoencoding (predict all tokens from full context) is trivial. After saturation, the auxiliary task provides zero useful gradient signal - only UniFrac loss drives learning.

**Solution:**
Implement masked prediction similar to BERT's MLM objective:
1. Randomly mask 15-30% of nucleotide positions (excluding padding/START)
2. Replace masked positions with a MASK token
3. Only compute loss on masked positions
4. Forces model to learn true contextual sequence patterns

**Implementation Details:**
```python
# Masking strategy options:
# 1. Random masking (15-30% of valid positions)
# 2. Span masking (contiguous 3-5bp chunks) - harder, forces longer-range deps
# 3. Hybrid approach

# Vocab change: Add MASK token (id=6) to vocab_size=7
# Or reuse padding token for masking (simpler, no vocab change)
```

**Acceptance Criteria:**
- [x] Add `mask_ratio` parameter to ASVEncoder (default: 0.15)
- [x] Add `mask_strategy` parameter ('random', 'span', default: 'random')
- [x] Implement masking logic in ASVEncoder.forward() during training
- [x] Only compute nucleotide loss on masked positions
- [x] Add CLI flags: `--nuc-mask-ratio`, `--nuc-mask-strategy`
- [x] Verify nucleotide loss no longer saturates (should remain > 0.1 throughout training)
- [x] Update tests for new masking behavior
- [x] Add TensorBoard visualization for nucleotide metrics (train/val overlay)
- [x] Add nucleotide accuracy metric

**Expected Impact:**
- Continuous gradient signal from nucleotide task throughout training
- Better sequence representations (must learn actual context, not position mapping)
- Improved regularization effect
- Potentially better final UniFrac prediction performance

**Files:** `asv_encoder.py`, `sample_sequence_encoder.py`, `sequence_encoder.py`, `losses.py`, `cli.py`, `trainer.py`, `tests/test_asv_encoder.py`

**References:**
- BERT: 15% masking with 80/10/10 mask/random/keep strategy
- DNA-BERT: Similar approach for genomic sequences
- Observation: Current NL saturates epoch 1→2: 0.7319 → 0.0002

---

## Maintenance: Test Fixes

### PYT-19.1: Fix Failing Unit Tests
**Priority:** HIGH | **Effort:** 1-2 hours | **Status:** Complete

Fix 4 pre-existing failing unit tests discovered during PYT-18.1 implementation.

**Failing Tests:**
1. `test_cli.py::TestCLIIntegration::test_train_command_full_flow`
2. `test_cli.py::TestPretrainedEncoderLoading::test_train_command_loads_pretrained_encoder`
3. `test_cli.py::TestPretrainedEncoderLoading::test_train_command_pretrained_encoder_with_freeze_base`
4. `test_losses.py::TestPairwiseDistances::test_compute_pairwise_distances_no_saturation`

**Root Causes:**
- Tests 1-3: Mock `metadata_df.columns` returns a list instead of pandas Index, causing `'list' object has no attribute 'str'` error when `metadata_df.columns.str.strip()` is called
- Test 4: Flaky assertion `std_val > 0.03` fails intermittently (got 0.024)

**Acceptance Criteria:**
- [x] Fix mock to return proper pandas Index for metadata columns
- [x] Relax or fix flaky saturation test threshold
- [x] All 582 tests pass

**Files:** `tests/test_cli.py`, `tests/test_losses.py`

---

## Phase 18: Memory Optimization (NEW - HIGH PRIORITY)

See `_design_plan/23_memory_optimization_plan.md` for detailed analysis.

### PYT-18.1: Enable Memory-Efficient Defaults
**Priority:** HIGH | **Effort:** 2-3 hours | **Status:** Complete

Update defaults: `asv_chunk_size=256`, `gradient_checkpointing=True`, `attn_implementation=mem_efficient`

**Acceptance Criteria:**
- [x] Update CLI defaults in `cli.py`
- [x] Add `--no-gradient-checkpointing` flag to opt out
- [x] Add `--asv-chunk-size` to train command
- [x] Document memory impact in help text
- [x] Verify no accuracy regression (tests pass)
- [x] Add model summary logging at training start

**Files:** `cli.py`, `sequence_predictor.py`, `model_summary.py`

---

### PYT-18.2: Streaming Validation Metrics
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Complete

Replace O(dataset) memory accumulation with streaming metrics.

**Acceptance Criteria:**
- [x] Compute metrics incrementally (running mean/variance)
- [x] Streaming metrics as default behavior (no flag needed)
- [x] Reduce validation memory from O(dataset) to O(batch)
- [x] Maintain metric accuracy
- [x] Reservoir sampling for plot data (1000 samples max)

**Files:** `trainer.py`, `metrics.py`

---

### PYT-18.3: Skip Nucleotide Predictions During Inference
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** Not Started

Skip nucleotide head `[batch, 1024, 150, 6]` tensors when not needed.

**Acceptance Criteria:**
- [ ] Add `predict_nucleotides` flag to ASVEncoder
- [ ] Auto-disable during eval/fine-tuning
- [ ] Reduce memory by ~3.7 MB/sample during inference

**Files:** `asv_encoder.py`, `sample_sequence_encoder.py`

---

### PYT-18.4: Configurable FFN Intermediate Size
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** Not Started

Add `--ffn-ratio` flag (default 4, can reduce to 2 for memory savings).

**Acceptance Criteria:**
- [ ] Add `ffn_ratio` parameter to transformers
- [ ] Propagate through model hierarchy
- [ ] Add CLI flag
- [ ] Document memory vs accuracy trade-off

**Files:** `transformer.py`, model files, `cli.py`

---

### PYT-18.5: Lazy Sample Embedding Computation
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

Only compute/return sample_embeddings when needed for loss.

### PYT-18.6: Memory-Aware Dynamic Batching
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

Add `--max-memory-gb` flag for dynamic batch adjustment.

---

## Phase 10: Performance (1 remaining)

### PYT-10.6: Multi-GPU Training (DDP)
**Priority:** MEDIUM | **Effort:** 8-12 hours | **Status:** Not Started

Implement DistributedDataParallel for linear scaling across GPUs.

**Acceptance Criteria:**
- [ ] DDP setup and initialization
- [ ] Data splitting across GPUs
- [ ] Metrics sync across processes
- [ ] CLI options for distributed training
- [ ] Test on 2+ GPUs
- [ ] Verify same results as single GPU

**Files:** `trainer.py`, `cli.py`, new distributed script

---

## Phase 12: Additional Performance (3 tickets)

### PYT-12.1: FSDP
**Priority:** LOW | **Effort:** 12-16 hours

Fully Sharded Data Parallel for memory-efficient distributed training.

### PYT-12.2: Batch Size Optimization
**Priority:** LOW | **Effort:** 4-6 hours

Dynamic batch sizing and automatic batch size finder.

### PYT-12.3: Caching Mechanisms
**Priority:** LOW | **Effort:** 3-4 hours | **Status:** Complete

Cache tokenized sequences at dataset initialization to avoid redundant tokenization.

**Acceptance Criteria:**
- [x] Add `cache_sequences` parameter to ASVDataset (default: True)
- [x] Build sequence→tensor cache at init when enabled
- [x] Use cached tensors in __getitem__ when available
- [x] Add `--no-sequence-cache` CLI flag to train/pretrain/predict
- [x] Tests for caching behavior

**Files:** `aam/data/dataset.py`, `aam/cli.py`, `tests/test_dataset.py`

---

## Phase 13-17: Future Enhancements (13 tickets)

- **Phase 13:** Attention Visualization, Feature Importance, Encoder Types
- **Phase 14:** Streaming Data, Augmentation
- **Phase 15:** Experiment Tracking, Hyperparameter Optimization
- **Phase 16:** Benchmarking, Error Analysis
- **Phase 17:** Docs, Tutorials, ONNX, Docker

---

## Summary

| Phase | Tickets | Est. Hours | Priority |
|-------|---------|------------|----------|
| 21 (Fine-Tuning) | 3 (1 complete) | 8-12 | **HIGH** |
| 20 (SSL) | 0 (1 complete) | - | **DONE** |
| 19 (Tests) | 0 (1 complete) | - | **DONE** |
| 18 (Memory) | 5 | 16-23 | **HIGH** |
| 10 | 1 | 8-12 | MEDIUM |
| 12 | 2 (1 complete) | 16-22 | LOW |
| 13-17 | 13 | 53-73 | LOW |
| **Total** | **24** | **101-142** |
