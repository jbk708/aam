# Outstanding PyTorch Tickets

**Last Updated:** 2025-12-19
**Status:** Phases 8-11, 19-21 Complete (see `ARCHIVED_TICKETS.md`)

---

## Phase 21: Transfer Learning (1 remaining)

### PYT-21.3: Regressor Head Optimization
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

Enhance regressor with configurable improvements:
- `--unbounded-targets` flag (disables sigmoid)
- `--learnable-output-scale` with trainable scale/bias
- `--target-layer-norm` for LayerNorm before projection
- Proper weight initialization

**Files:** `aam/models/sequence_predictor.py`, `aam/cli/train.py`

---

### PYT-21.5: Skip Nucleotide Predictions During Fine-Tuning
**Priority:** LOW | **Effort:** 2-3 hours | **Status:** Not Started

When `--freeze-base`, skip nucleotide predictions entirely to save memory/compute.
- Add `--skip-nuc-predictions` flag (auto-enabled with `--freeze-base`)
- Add `--keep-nuc-predictions` to force computation

**Files:** `aam/cli/train.py`, `aam/models/sequence_predictor.py`

---

## Phase 18: Memory Optimization (2 remaining)

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

Note: DDP infrastructure exists in `aam/training/distributed.py` (COS-4.1). This ticket is for testing/validation on multi-GPU CUDA systems.

---

## Phase 12: Additional Performance (2 remaining)

### PYT-12.1: FSDP
**Priority:** LOW | **Effort:** 12-16 hours

Fully Sharded Data Parallel for memory-efficient distributed training.

### PYT-12.2: Batch Size Optimization
**Priority:** LOW | **Effort:** 4-6 hours

Dynamic batch sizing and automatic batch size finder.

---

## Phase 13-17: Future Enhancements

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
| 21 (Fine-Tuning) | 2 | 6-9 |
| 18 (Memory) | 2 | 8-12 |
| 10 (Performance) | 1 | 8-12 |
| 12 (Additional) | 2 | 16-22 |
| 13-17 (Future) | ~13 | 50+ |
