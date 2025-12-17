# Outstanding Tickets

**Last Updated:** 2025-12-17
**Status:** Phases 8-11 Complete (see `ARCHIVED_TICKETS.md`)

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

**Files:** `cli.py`, `sequence_predictor.py`

---

### PYT-18.2: Streaming Validation Metrics
**Priority:** HIGH | **Effort:** 3-4 hours | **Status:** Not Started

Replace O(dataset) memory accumulation with streaming metrics.

**Acceptance Criteria:**
- [ ] Compute metrics incrementally (running mean/variance)
- [ ] Add `--validation-plot-samples` flag to limit plot data
- [ ] Reduce validation memory from O(dataset) to O(batch)
- [ ] Maintain metric accuracy

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
| 18 (Memory) | 6 | 18-26 | **HIGH** |
| 10 | 1 | 8-12 | MEDIUM |
| 12 | 3 | 19-26 | LOW |
| 13-17 | 13 | 53-73 | LOW |
| **Total** | **23** | **98-137** |
