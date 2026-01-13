# Outstanding PyTorch Tickets

**Last Updated:** 2026-01-08
**Status:** Phases 8-11, 18-21 mostly complete (see `ARCHIVED_TICKETS.md`)

---

## Phase 10: Performance (1 remaining)

### PYT-10.6: Multi-GPU Training (DDP) Validation
**Priority:** LOW | **Effort:** 8-12 hours | **Status:** COMPLETE

DDP infrastructure exists in `aam/training/distributed.py` (COS-4.1). Validated on ROCm MI300A.

**Completed:**
- Fixed `DistributedSampler.set_epoch()` not being called (broke shuffling)
- Added `train_sampler` parameter to `Trainer` class
- Validated DDP runs on 4-GPU MI300A node

**Finding:** DDP is not suitable for pretraining with pairwise UniFrac loss. Each GPU only computes local pairwise distances, missing cross-GPU comparisons. See PYT-10.7 for solution.

### PYT-10.7: DataParallel for Pretraining
**Priority:** MEDIUM | **Effort:** 2-4 hours | **Status:** COMPLETE

DataParallel preserves full pairwise comparisons for UniFrac loss by gathering outputs to GPU 0 before loss computation.

**Problem:**
DDP computes pairwise UniFrac loss locally per GPU, causing predictions to converge to mean (~0.5) instead of learning full distance distribution. Single-GPU training works correctly.

**Completed:**
- Added `--data-parallel` flag to `pretrain.py` (mutually exclusive with `--distributed`)
- Wrapped model with `nn.DataParallel` when flag is set
- Documented DP vs DDP trade-offs in README.md (Multi-GPU Training section)
- Added tests for flag, mutual exclusion, CUDA requirement, and wrapping

**Files Modified:**
- `aam/cli/pretrain.py` - Added flag and DP wrapping
- `README.md` - Added Multi-GPU Training section with DP vs DDP guidance
- `tests/test_cli.py` - Added 4 tests for DataParallel functionality

**Usage:**
```bash
# Single process uses all visible GPUs
python -m aam.cli pretrain --data-parallel --batch-size 32 ...
```

**Acceptance Criteria:**
- [x] `--data-parallel` flag works with multi-GPU pretraining
- [ ] UniFrac predictions show full variance (not clustered at 0.5) - *requires multi-GPU hardware validation*
- [ ] Training metrics match single-GPU behavior - *requires multi-GPU hardware validation*
- [x] Cannot use `--data-parallel` and `--distributed` together
- [x] GPU 0 memory overhead documented

**Note:** DataParallel is single-node only. For multi-node pretraining, would need PYT-10.8 (cross-GPU gathering for DDP).

---

## Phase 12: Distributed Training (2 remaining)

### PYT-12.1: FSDP (Fully Sharded Data Parallel)
**Priority:** LOW | **Effort:** 12-16 hours | **Status:** Not Started

*Note: Consolidated with COS-4.2. Implement once, validate on both CUDA and ROCm.*

**Scope:**
- Implement FSDP for memory-efficient distributed training
- Support model sharding across GPUs
- Validate on both CUDA and ROCm (MI300A)
- May be unnecessary given MI300A's 128GB unified memory

**Files:** `aam/training/distributed.py`

### PYT-12.2: Batch Size Optimization
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

Dynamic batch sizing and automatic batch size finder.

---

## Phase 18: Memory Optimization (2 remaining)

### PYT-18.5: Lazy Sample Embedding Computation
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

Only compute/return sample_embeddings when needed for loss.

### PYT-18.6: Memory-Aware Dynamic Batching
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

Add `--max-memory-gb` flag for dynamic batch adjustment.

---

## Phase 19: Regression Optimization (2 tickets)

### PYT-19.1: Non-Negative Regression Output Constraints
**Priority:** HIGH | **Effort:** 4-6 hours | **Status:** COMPLETE

Model predicts negative values for targets that should be non-negative (e.g., concentrations, counts, distances). Need output constraints to enforce valid prediction ranges.

**Completed:**
- Added `--output-activation` flag: `none` (default), `relu`, `softplus`, `exp`
- Validates mutual exclusion with `--bounded-targets` and `--classifier`
- Documented in README under "Regression Options"

**Files Modified:**
- `aam/models/sequence_predictor.py` - Added `output_activation` parameter and `_apply_output_activation()` method
- `aam/cli/train.py` - Added `--output-activation` flag
- `tests/test_sequence_predictor.py` - Added TestOutputActivation test class with 15 tests
- `README.md` - Added "Regression Options" section

**Acceptance Criteria:**
- [x] `--output-activation softplus` produces non-negative predictions
- [x] Validation metrics comparable or better than unconstrained
- [x] Document in README under "Regression Options"

### PYT-19.2: Categorical Data Integration Improvements
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

Improve integration of categorical metadata features for conditioning target predictions.

**Scope:**
- TBD based on user requirements

---

## Phase 13-17: Future Enhancements (Backlog)

Low priority future work:
- **Phase 13:** Attention Visualization, Feature Importance, Encoder Types
- **Phase 14:** Streaming Data, Augmentation
- **Phase 15:** Experiment Tracking, Hyperparameter Optimization
- **Phase 16:** Benchmarking, Error Analysis
- **Phase 17:** Docs, Tutorials, ONNX, Docker

---

## Summary

| Phase | Remaining | Est. Hours |
|-------|-----------|------------|
| 10 (Performance) | 0 | 0 |
| 12 (Distributed) | 2 | 16-22 |
| 18 (Memory) | 2 | 8-12 |
| 19 (Regression) | 1 | 4-6 |
| 13-17 (Future) | ~13 | 50+ |
| **Total** | **5 + backlog** | **28-40 + 50+** |
