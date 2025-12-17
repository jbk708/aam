# Outstanding Tickets

**Last Updated:** 2025-12-17
**Status:** Phases 8-11 Complete (see `ARCHIVED_TICKETS.md`), PYT-19.1 Complete, PYT-20.1 Complete (MAE for Nucleotide Prediction)

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
**Priority:** LOW | **Effort:** 3-4 hours

Cache tokenized sequences and expensive computations.

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
| 20 (SSL) | 0 (1 complete) | - | **DONE** |
| 19 (Tests) | 0 (1 complete) | - | **DONE** |
| 18 (Memory) | 5 | 16-23 | **HIGH** |
| 10 | 1 | 8-12 | MEDIUM |
| 12 | 3 | 19-26 | LOW |
| 13-17 | 13 | 53-73 | LOW |
| **Total** | **22** | **96-134** |
