# Outstanding Tickets

**Last Updated:** 2025-12-17
**Status:** Phases 8-11 Complete (see `ARCHIVED_TICKETS.md`)

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

Fully Sharded Data Parallel for memory-efficient distributed training of large models.

### PYT-12.2: Batch Size Optimization
**Priority:** LOW | **Effort:** 4-6 hours

Dynamic batch sizing and automatic batch size finder.

### PYT-12.3: Caching Mechanisms
**Priority:** LOW | **Effort:** 3-4 hours

Cache tokenized sequences and expensive computations.

---

## Phase 13: Model Improvements (3 tickets)

### PYT-13.1: Attention Visualization
**Priority:** LOW | **Effort:** 4-6 hours

Extract and visualize attention patterns for interpretability.

### PYT-13.2: Feature Importance Analysis
**Priority:** LOW | **Effort:** 4-6 hours

Gradient-based, attention-based, and permutation importance methods.

### PYT-13.3: Additional Encoder Types
**Priority:** LOW | **Effort:** 4-6 hours

Support Bray-Curtis, Jaccard, Aitchison distances.

---

## Phase 14: Data Pipeline (2 tickets)

### PYT-14.1: Streaming Data Loading
**Priority:** LOW | **Effort:** 6-8 hours

Lazy loading for datasets larger than RAM.

### PYT-14.2: Data Augmentation
**Priority:** LOW | **Effort:** 4-6 hours

Sequence shuffling, masking, noise injection.

---

## Phase 15: Training Improvements (2 tickets)

### PYT-15.1: Experiment Tracking
**Priority:** LOW | **Effort:** 4-6 hours

Weights & Biases and MLflow integration.

### PYT-15.2: Hyperparameter Optimization
**Priority:** LOW | **Effort:** 6-8 hours

Optuna or Ray Tune integration.

---

## Phase 16: Evaluation Tools (2 tickets)

### PYT-16.1: Benchmarking Suite
**Priority:** LOW | **Effort:** 4-6 hours

Standardized benchmarks and performance metrics.

### PYT-16.2: Error Analysis Tools
**Priority:** LOW | **Effort:** 4-6 hours

Error distribution and sample-level analysis.

---

## Phase 17: Documentation & Deployment (4 tickets)

### PYT-17.1: API Documentation (Sphinx)
**Priority:** LOW | **Effort:** 4-6 hours

### PYT-17.2: Tutorial Notebooks
**Priority:** LOW | **Effort:** 4-6 hours

### PYT-17.3: ONNX Export
**Priority:** LOW | **Effort:** 3-4 hours

### PYT-17.4: Docker Containerization
**Priority:** LOW | **Effort:** 2-3 hours

---

## Summary

| Phase | Tickets | Est. Hours |
|-------|---------|------------|
| 10 | 1 | 8-12 |
| 12 | 3 | 19-26 |
| 13 | 3 | 12-18 |
| 14 | 2 | 10-14 |
| 15 | 2 | 10-14 |
| 16 | 2 | 8-12 |
| 17 | 4 | 13-19 |
| **Total** | **17** | **80-115** |
