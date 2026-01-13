# Outstanding PyTorch Tickets

**Last Updated:** 2026-01-13
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

## Phase 12: Distributed Training (4 remaining)

### PYT-12.1: FSDP (Fully Sharded Data Parallel)
**Priority:** MEDIUM | **Effort:** 11-16 hours | **Status:** IN PROGRESS
**Branch:** `pyt-12.1-fsdp-implementation`

*Note: Consolidated with COS-4.2. Implement once, validate on both CUDA and ROCm.*

FSDP enables memory-efficient distributed training by sharding model parameters, gradients, and optimizer states across GPUs. Unlike DDP which replicates the full model on each GPU, FSDP only materializes full parameters during forward/backward passes.

**Motivation:**
- Enable training larger models than GPU memory allows
- More memory-efficient than DataParallel (no GPU 0 bottleneck)
- Better scaling for multi-node training than DataParallel
- May be unnecessary for MI300A's 128GB unified memory, but valuable for smaller GPUs

**Sub-tickets:**

---

#### PYT-12.1a: FSDP Infrastructure
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

Add basic FSDP support to `distributed.py` with auto-wrap policy for transformer modules.

**Background:**
FSDP wraps model modules to shard their parameters. The wrapping policy determines which modules get their own FSDP unit. For transformer models, wrapping each transformer layer is standard practice.

**Scope:**
- Add `wrap_model_fsdp()` function in `distributed.py`
- Define transformer wrapping policy using `ModuleWrapPolicy` targeting `TransformerEncoderLayer`
- Add `--fsdp` flag to `train.py` (fine-tuning only initially, avoids UniFrac pairwise issue)
- Use `FULL_SHARD` sharding strategy (most memory efficient)
- Integrate with existing `setup_distributed()` infrastructure
- Handle device placement (FSDP manages this internally)

**Implementation Notes:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

def wrap_model_fsdp(
    model: nn.Module,
    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD,
    mixed_precision: Optional[MixedPrecision] = None,
) -> FSDP:
    # Wrap transformer layers individually
    wrap_policy = ModuleWrapPolicy({TransformerEncoderLayer})
    return FSDP(
        model,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision,
        device_id=torch.cuda.current_device(),
    )
```

**Acceptance Criteria:**
- [ ] `wrap_model_fsdp()` function added to `distributed.py`
- [ ] `--fsdp` flag added to `train.py` (mutually exclusive with `--distributed` and `--data-parallel`)
- [ ] Training loop completes without error on multi-GPU
- [ ] Memory usage per GPU lower than DDP for same model
- [ ] Unit tests for FSDP wrapping

**Files:**
- `aam/training/distributed.py` - Add `wrap_model_fsdp()`, FSDP utilities
- `aam/cli/train.py` - Add `--fsdp` flag
- `tests/test_distributed.py` - Add FSDP wrapping tests

**Dependencies:** None

---

#### PYT-12.1b: FSDP Checkpoint Support
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** Not Started

Handle FSDP's special checkpoint formats for save/load compatibility.

**Background:**
FSDP models have sharded state dicts by default. PyTorch provides `StateDictType` options:
- `FULL_STATE_DICT`: Gathers full state dict on rank 0 (compatible with non-FSDP)
- `SHARDED_STATE_DICT`: Each rank saves its shard (faster for large models)
- `LOCAL_STATE_DICT`: Raw local state (not portable)

For compatibility with existing checkpoints and non-FSDP inference, we'll use `FULL_STATE_DICT` by default with an option for sharded.

**Scope:**
- Add FSDP state dict context managers to `Trainer.save_checkpoint()`
- Add FSDP state dict handling to `Trainer.load_checkpoint()`
- Support loading non-FSDP checkpoints into FSDP models
- Support loading FSDP checkpoints into non-FSDP models
- Add `--fsdp-sharded-checkpoint` flag for large model optimization

**Implementation Notes:**
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig

def save_fsdp_checkpoint(model: FSDP, path: str):
    # Gather full state dict on rank 0
    cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
        state_dict = model.state_dict()
        if is_main_process():
            torch.save(state_dict, path)
```

**Acceptance Criteria:**
- [ ] FSDP model checkpoints save correctly
- [ ] FSDP model can load its own checkpoints
- [ ] Non-FSDP model can load FSDP checkpoint (for inference)
- [ ] FSDP model can load pre-trained non-FSDP checkpoint
- [ ] `--fsdp-sharded-checkpoint` option for large models
- [ ] Tests for all checkpoint roundtrip scenarios

**Files:**
- `aam/training/trainer.py` - Add FSDP checkpoint handling
- `aam/training/distributed.py` - Add checkpoint utility functions
- `tests/test_trainer.py` - Add FSDP checkpoint tests

**Dependencies:** PYT-12.1a

---

#### PYT-12.1c: FSDP for Pretraining + ROCm Validation
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** Not Started

Extend FSDP to pretraining with cross-GPU embedding gathering for UniFrac loss.

**Background:**
The UniFrac loss requires pairwise distance computation across all samples in a batch. With FSDP/DDP, each GPU only sees its local batch, so we need to gather embeddings across GPUs before computing pairwise distances. This is the same issue that made DDP unsuitable for pretraining (PYT-10.6).

**Problem:**
```
GPU 0: samples [0,1,2,3] -> local pairwise distances only
GPU 1: samples [4,5,6,7] -> local pairwise distances only
Missing: cross-GPU pairs (0,4), (0,5), (1,4), etc.
```

**Solution:**
Gather embeddings from all GPUs before UniFrac loss computation:
```python
def gather_embeddings_for_unifrac(embeddings: torch.Tensor) -> torch.Tensor:
    # All-gather embeddings across GPUs
    gathered = [torch.zeros_like(embeddings) for _ in range(world_size)]
    dist.all_gather(gathered, embeddings)
    return torch.cat(gathered, dim=0)  # [batch_size * world_size, embed_dim]
```

**Scope:**
- Add embedding gathering utility to `distributed.py`
- Modify UniFrac loss computation to use gathered embeddings
- Add `--fsdp` flag to `pretrain.py`
- Validate on ROCm MI300A (4-GPU node)
- Document FSDP vs DP vs DDP trade-offs in README
- Performance benchmarking: memory usage, throughput

**ROCm Considerations:**
- FSDP uses NCCL/RCCL for communication (same as DDP)
- ROCm 6.3+ recommended (fixes SDPA issues)
- Test with `--attn-implementation math` fallback if needed

**Acceptance Criteria:**
- [ ] `--fsdp` flag added to `pretrain.py`
- [ ] UniFrac predictions show full variance (not clustered at 0.5)
- [ ] Pretraining metrics match DataParallel behavior
- [ ] Works on CUDA (if available for testing)
- [ ] Works on ROCm MI300A
- [ ] Memory usage documented (vs DataParallel)
- [ ] README updated with FSDP usage guidance

**Files:**
- `aam/training/distributed.py` - Add `gather_embeddings_for_unifrac()`
- `aam/training/losses.py` - Integrate gathering into UniFrac loss path
- `aam/cli/pretrain.py` - Add `--fsdp` flag
- `README.md` - Add FSDP documentation
- `tests/test_distributed.py` - Add gathering tests

**Dependencies:** PYT-12.1a, PYT-12.1b

---

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
| 12 (Distributed) | 4 | 15-22 |
| 18 (Memory) | 2 | 8-12 |
| 19 (Regression/Categorical) | 3 | 13-18 |
| 13-17 (Future) | ~13 | 50+ |
| **Total** | **9 + backlog** | **36-52 + 50+** |

### PYT-12.1 Sub-ticket Breakdown

| Sub-ticket | Description | Effort | Status |
|------------|-------------|--------|--------|
| PYT-12.1a | FSDP Infrastructure | 4-6h | Not Started |
| PYT-12.1b | FSDP Checkpoint Support | 3-4h | Not Started |
| PYT-12.1c | FSDP Pretraining + ROCm | 4-6h | Not Started |
