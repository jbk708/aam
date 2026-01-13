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

## Phase 19: Regression & Categorical (4 tickets)

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

### PYT-19.2: Categorical Cross-Attention
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

Use cross-attention between sequence embeddings and categorical embeddings instead of simple concat/add fusion.

**Problem:**
- Current fusion strategies (concat/add) apply categorical information uniformly across all ASV positions
- Cross-attention allows the model to learn which sequence features are relevant for each category

**Scope:**
- Add `--categorical-fusion cross_attention` option
- Sequence embeddings (queries) attend to categorical embeddings (keys/values)
- Multi-head cross-attention with configurable heads
- Residual connection: `output = sequence_emb + cross_attn(sequence_emb, cat_emb)`

**Implementation:**
```python
# In SequencePredictor
self.categorical_cross_attn = nn.MultiheadAttention(
    embed_dim=embedding_dim,
    num_heads=num_heads,
    kdim=categorical_embed_dim,
    vdim=categorical_embed_dim,
)
# Q: [B, S, D] sequence embeddings
# K, V: [B, num_cats, cat_dim] categorical embeddings
```

**Acceptance Criteria:**
- [ ] `--categorical-fusion cross_attention` works with training
- [ ] Attention weights can be extracted for interpretability
- [ ] Performance comparable or better than concat fusion
- [ ] Tests for cross-attention pathway

**Files:**
- `aam/models/sequence_predictor.py` - Add cross-attention fusion
- `aam/models/categorical_embedder.py` - Return per-column embeddings for K/V
- `aam/cli/train.py` - Add cross_attention to fusion choices
- `tests/test_sequence_predictor.py` - Test cross-attention integration

---

### PYT-19.3: Per-Category Loss Weighting
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** Not Started

Weight samples differently based on categorical values to handle imbalanced categories.

**Problem:**
- Some categorical values may be underrepresented in training data
- Model may underfit rare categories without reweighting

**Scope:**
- Add `--categorical-loss-weights` flag accepting JSON or auto-compute from frequencies
- Weight each sample's loss contribution by its category weight
- Support multiple categorical columns with combined weighting

**Implementation Options:**
1. **Inverse frequency weighting**: `weight = 1 / freq(category)`
2. **Effective number weighting**: `weight = (1 - beta^n) / (1 - beta)` where n = count
3. **Manual weights**: User-specified JSON `{"location": {"urban": 1.0, "rural": 2.0}}`

**Acceptance Criteria:**
- [ ] `--categorical-loss-weights auto` computes inverse frequency weights
- [ ] `--categorical-loss-weights <json_file>` loads manual weights
- [ ] Weights applied correctly to loss computation
- [ ] Logging shows effective weights per category

**Files:**
- `aam/training/losses.py` - Add sample weighting to MultiTaskLoss
- `aam/data/categorical.py` - Add weight computation utilities
- `aam/cli/train.py` - Add `--categorical-loss-weights` flag
- `tests/test_losses.py` - Test weighted loss computation

---

### PYT-19.4: Hierarchical Categorical Encoding
**Priority:** LOW | **Effort:** 6-8 hours | **Status:** Not Started

Handle hierarchical categories where child categories inherit from parents (e.g., taxonomy, geography).

**Problem:**
- Categories often have natural hierarchies (country > state > city)
- Flat embeddings don't capture hierarchical relationships
- Child categories should share information with parents

**Scope:**
- Define hierarchy via config: `{"city": "state", "state": "country"}`
- Child embedding = child_embed + parent_embed (or learned combination)
- Support multiple independent hierarchies

**Implementation:**
```python
# Hierarchical embedding computation
class HierarchicalCategoricalEmbedder:
    def __init__(self, hierarchies: Dict[str, str], ...):
        # hierarchies maps child_col -> parent_col
        self.hierarchies = hierarchies

    def forward(self, categorical_ids):
        embeddings = {}
        for col in topological_order(self.hierarchies):
            emb = self.embeddings[col](categorical_ids[col])
            if col in self.hierarchies:
                parent_col = self.hierarchies[col]
                emb = emb + embeddings[parent_col]
            embeddings[col] = emb
        return concat(embeddings.values())
```

**Acceptance Criteria:**
- [ ] `--categorical-hierarchy` accepts hierarchy definition
- [ ] Topological ordering ensures parents computed before children
- [ ] Child embeddings incorporate parent information
- [ ] Works with cross-attention fusion (PYT-19.2)

**Files:**
- `aam/models/categorical_embedder.py` - Add HierarchicalCategoricalEmbedder
- `aam/data/categorical.py` - Add hierarchy schema validation
- `aam/cli/train.py` - Add `--categorical-hierarchy` flag
- `tests/test_categorical_embedder.py` - Test hierarchical encoding

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
| 19 (Regression/Categorical) | 3 | 13-18 |
| 13-17 (Future) | ~13 | 50+ |
| **Total** | **7 + backlog** | **39-52 + 50+** |
