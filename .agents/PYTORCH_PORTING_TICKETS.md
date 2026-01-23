# Outstanding PyTorch Tickets

**Last Updated:** 2026-01-23
**Status:** 6 tickets remaining (~20-32 hours)

**Completed:** PYT-BUG-1 to PYT-BUG-4, PYT-10.6-10.7, PYT-12.1a/b/c, PYT-12.2, PYT-19.1, PYT-19.3 (see `ARCHIVED_TICKETS.md`)

---

## Outstanding Tickets

### PYT-18.5: Lazy Sample Embedding Computation
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

Only compute/return sample_embeddings when needed for loss.

---

### PYT-18.6: Memory-Aware Dynamic Batching
**Priority:** LOW | **Effort:** 4-6 hours | **Status:** Not Started

Add `--max-memory-gb` flag for dynamic batch adjustment.

---

### PYT-19.2: Categorical Cross-Attention
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Superseded by FUS-2

*Note: This ticket is superseded by FUS-2 in `FUSION_CLEANUP_TICKETS.md`.*

---

### PYT-19.3: Per-Category Loss Weighting
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** ✅ Complete

Weight samples by category frequency for imbalanced data.

**Scope:**
- `--categorical-loss-weights auto` (inverse frequency)
- `--categorical-loss-weights <json>` (manual weights)

**Implementation (2026-01-23):**
- Added `CategoryWeighter` class in `aam/data/normalization.py`
- Integrated `sample_weights` in `MultiTaskLoss.compute_target_loss()`
- Added CLI args: `--categorical-loss-weights`, `--categorical-loss-weight-column`
- Full pipeline: dataset → collate_fn → trainer → loss function
- Tests: 28 new tests (19 for CategoryWeighter, 9 for weighted loss)

---

### PYT-19.4: Hierarchical Categorical Encoding
**Priority:** LOW | **Effort:** 6-8 hours | **Status:** Not Started

Child categories inherit parent embeddings (e.g., city → state → country).

---

### PYT-MAINT-1: CLI Flag Cleanup
**Priority:** LOW | **Effort:** 2-4 hours | **Status:** Superseded by CLN-1/CLN-2

*Note: This ticket is superseded by CLN-1 and CLN-2 in `FUSION_CLEANUP_TICKETS.md`.*

---

### PYT-MAINT-2: TensorBoard Logging Improvements
**Priority:** LOW | **Effort:** 2-4 hours | **Status:** Not Started

FiLM-specific metrics, conditional scaling stats, learning rate visualization.

---

## Backlog (Phases 13-17)

Future enhancement phases (~50+ hours):
- **Phase 13:** Attention Visualization, Feature Importance
- **Phase 14:** Streaming Data, Augmentation
- **Phase 15:** Experiment Tracking, Hyperparameter Optimization
- **Phase 16:** Benchmarking, Error Analysis
- **Phase 17:** Docs, Tutorials, ONNX, Docker

---

## Summary

| Ticket | Description | Effort | Priority | Notes |
|--------|-------------|--------|----------|-------|
| **PYT-18.5** | Lazy embeddings | 4-6h | LOW | |
| **PYT-18.6** | Memory-aware batching | 4-6h | LOW | |
| **PYT-19.2** | Cross-attention | 4-6h | MEDIUM | → FUS-2 |
| ~~PYT-19.3~~ | ~~Category loss weights~~ | ~~3-4h~~ | ~~MEDIUM~~ | ✅ Complete |
| **PYT-19.4** | Hierarchical categories | 6-8h | LOW | |
| **PYT-MAINT-1** | CLI cleanup | 2-4h | LOW | → CLN-1/2 |
| **PYT-MAINT-2** | TensorBoard | 2-4h | LOW | |
| **Total (unique)** | | **~16-30h** | | |
