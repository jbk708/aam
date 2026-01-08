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
**Priority:** MEDIUM | **Effort:** 2-4 hours | **Status:** Not Started

DataParallel preserves full pairwise comparisons for UniFrac loss by gathering outputs to GPU 0 before loss computation.

**Problem:**
DDP computes pairwise UniFrac loss locally per GPU, causing predictions to converge to mean (~0.5) instead of learning full distance distribution. Single-GPU training works correctly.

**Scope:**
- Add `--data-parallel` flag to `pretrain.py` (mutually exclusive with `--distributed`)
- Wrap model with `nn.DataParallel` when flag is set
- Validate UniFrac predictions show full variance
- Document when to use DP vs DDP

**Files:**
- `aam/cli/pretrain.py` - Add flag and DP wrapping
- `README.md` - Document DP vs DDP guidance

**Usage:**
```bash
# Single process uses all visible GPUs
python -m aam.cli pretrain --data-parallel --batch-size 32 ...
```

**Acceptance Criteria:**
- [ ] `--data-parallel` flag works with multi-GPU pretraining
- [ ] UniFrac predictions show full variance (not clustered at 0.5)
- [ ] Training metrics match single-GPU behavior
- [ ] Cannot use `--data-parallel` and `--distributed` together
- [ ] GPU 0 memory overhead documented

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
| 10 (Performance) | 1 | 2-4 |
| 12 (Distributed) | 2 | 16-22 |
| 18 (Memory) | 2 | 8-12 |
| 13-17 (Future) | ~13 | 50+ |
| **Total** | **5 + backlog** | **26-38 + 50+** |
