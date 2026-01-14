# Outstanding PyTorch Tickets

**Last Updated:** 2026-01-13
**Status:** Phases 8-11, 18-21 mostly complete (see `ARCHIVED_TICKETS.md`)

---

## URGENT: Bug Fixes

### PYT-BUG-1: Distributed Validation Metrics Not Synchronized
**Priority:** URGENT | **Effort:** 2-4 hours | **Status:** COMPLETE

Validation metrics (R², MAE) are computed per-GPU and logged independently, showing inconsistent results.

**Problem:**
With DDP/FSDP training using multiple GPUs, each process:
1. Gets a different subset of validation samples (via DistributedSampler)
2. Computes R²/MAE on only its local ~25% of data
3. Logs independently without synchronization

Result: 4 different metric values logged, graph shows only one (often the worst).

**Example Output (nproc=4):**
```
GPU 0: r2=-0.30, mae=84.11  ← This is what TensorBoard shows
GPU 1: r2=0.28, mae=59.49
GPU 2: r2=0.39, mae=66.68
GPU 3: r2=0.58, mae=40.42   ← True best performance
```

**Root Cause:**
`Evaluator.validate_epoch()` uses streaming metrics that don't gather predictions across ranks. Unlike UniFrac loss which has `gather_embeddings_for_unifrac()`, validation metrics have no cross-GPU aggregation.

**Solution:**
Add `torch.distributed.all_gather()` for predictions and targets before computing metrics:
```python
# In Evaluator.validate_epoch(), before computing final metrics:
if dist.is_initialized():
    all_preds = [torch.zeros_like(preds) for _ in range(world_size)]
    all_targets = [torch.zeros_like(targets) for _ in range(world_size)]
    dist.all_gather(all_preds, preds)
    dist.all_gather(all_targets, targets)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
```

**Acceptance Criteria:**
- [ ] All GPUs compute metrics on the FULL validation set
- [ ] Only rank 0 logs metrics (already the case for TensorBoard)
- [ ] Logged R²/MAE matches single-GPU training results
- [ ] Works with both DDP and FSDP
- [ ] No memory explosion from gathering (use streaming if needed)

**Files:**
- `aam/training/evaluation.py` - Add all_gather before metric computation
- `aam/training/metrics.py` - May need distributed-aware streaming metrics
- `tests/test_distributed.py` - Add test for metric synchronization

**Dependencies:** None

---

### PYT-BUG-2: Best Model Selection Uses Loss Instead of Primary Metric
**Priority:** MEDIUM | **Effort:** 2-3 hours | **Status:** COMPLETE

Best model checkpoint is saved based on lowest validation loss, not the primary evaluation metric (R² for regression, accuracy for classification).

**Problem:**
- Current logic: `if val_loss < best_val_loss: save_checkpoint()`
- Loss can decrease while R² gets worse (overfitting to auxiliary losses)
- For regression, R² is typically the better indicator of model quality
- For classification, accuracy/F1 may be preferred over cross-entropy loss

**Example:**
```
Epoch 10: val_loss=0.045, r2=0.72  ← checkpoint saved (lowest loss)
Epoch 15: val_loss=0.048, r2=0.78  ← NOT saved, but better R²!
Epoch 20: val_loss=0.044, r2=0.65  ← checkpoint saved (overfitting)
```

**Solution:**
Add `--best-metric` flag to select which metric determines "best" model:
```python
# CLI flags
--best-metric r2          # For regression (higher is better)
--best-metric val_loss    # Current behavior (lower is better)
--best-metric accuracy    # For classification (higher is better)

# In Trainer.train():
if is_better(current_metric, best_metric, mode=metric_mode):
    save_checkpoint()
```

**Acceptance Criteria:**
- [x] `--best-metric` flag added to train.py with choices: val_loss, r2, mae, accuracy, f1
- [x] Support both "higher is better" and "lower is better" modes
- [x] Default to current behavior (val_loss) for backwards compatibility
- [x] Checkpoint filename or metadata indicates which metric was used
- [x] Tests for metric-based model selection

**Completed:**
- Added `METRIC_MODES` dict defining min/max modes for each metric
- Added `is_metric_better()` helper function for metric comparison
- Added `best_metric` parameter to `Trainer.__init__()` with validation
- Modified `train()` to use configurable metric for best model selection
- Added `best_metric` and `best_metric_value` fields to checkpoint
- Added `best_metric_value` parameter to `save_checkpoint()`
- Updated `load_checkpoint()` to return `best_metric_value`
- Added fallback to val_loss when selected metric not in validation results
- Added 14 unit tests for metric selection functionality
- Added 3 CLI tests for `--best-metric` flag

**Files:**
- `aam/cli/train.py` - Add `--best-metric` flag
- `aam/training/trainer.py` - Modify best model selection logic
- `tests/test_trainer.py` - Add tests for metric-based selection
- `tests/test_cli.py` - Add CLI flag tests

**Dependencies:** None

---

### PYT-BUG-3: Count Loss Has No Configurable Weight
**Priority:** MEDIUM | **Effort:** 1-2 hours | **Status:** COMPLETE

Count loss is added to total loss without a penalty weight, unlike all other loss components.

**Problem:**
```python
# In MultiTaskLoss.forward():
total_loss = (
    losses["target_loss"] * self.target_penalty
    + losses["count_loss"]                        # ← No penalty!
    + losses["unifrac_loss"] * self.penalty
    + losses["nuc_loss"] * self.nuc_penalty
)
```

All other losses have configurable weights:
- `--penalty` for UniFrac loss
- `--nuc-penalty` for nucleotide loss
- `--target-penalty` for target loss
- **Nothing** for count loss

This causes:
1. Cannot disable count loss (useful when count prediction is not needed)
2. Cannot downweight if count loss dominates total loss
3. Erratic total loss behavior if count scale differs from other losses

**Solution:**
Add `--count-penalty` flag with default 1.0 for backwards compatibility:

```python
# MultiTaskLoss.__init__
def __init__(self, ..., count_penalty: float = 1.0):
    self.count_penalty = count_penalty

# MultiTaskLoss.forward()
total_loss = (
    losses["target_loss"] * self.target_penalty
    + losses["count_loss"] * self.count_penalty   # ← Add weight
    + losses["unifrac_loss"] * self.penalty
    + losses["nuc_loss"] * self.nuc_penalty
)
```

**Acceptance Criteria:**
- [x] Add `count_penalty` parameter to `MultiTaskLoss.__init__()`
- [x] Apply weight in total loss computation
- [x] Add `--count-penalty` flag to train.py (default 1.0)
- [x] Document in README with other penalty flags
- [x] Tests for count penalty behavior

**Completed:**
- Added `count_penalty` parameter to `MultiTaskLoss.__init__()` with default 1.0
- Applied weight in total loss computation
- Added `--count-penalty` flag to train.py and pretrain.py
- Documented in README Loss Weights table
- Added 5 unit tests for count_penalty in test_losses.py
- Added 4 CLI tests for --count-penalty flag in test_cli.py

**Files:**
- `aam/training/losses.py` - Add count_penalty parameter and apply in forward()
- `aam/cli/train.py` - Add `--count-penalty` flag
- `aam/cli/pretrain.py` - Add `--count-penalty` flag
- `tests/test_losses.py` - Test count penalty weighting
- `tests/test_cli.py` - Test CLI flag existence and defaults

**Dependencies:** None

---

### PYT-BUG-4: Distributed Validation Prediction Plots Show Only Local GPU Data
**Priority:** MEDIUM | **Effort:** 2-4 hours | **Status:** COMPLETE

TensorBoard prediction plots only show validation samples from rank 0, not the full validation set.

**Problem:**
With DDP/FSDP training using multiple GPUs:
1. Each GPU processes ~25% of validation samples (via DistributedSampler)
2. `StreamingRegressionMetrics.get_plot_data()` only captures local samples
3. `log_figures_to_tensorboard()` only runs on rank 0
4. Result: TensorBoard scatter plot shows ~25% of true predictions

**Example (nproc=4, 200 val samples):**
```
GPU 0: samples 0-49   → Only these 50 points shown in TensorBoard
GPU 1: samples 50-99  → Discarded
GPU 2: samples 100-149 → Discarded
GPU 3: samples 150-199 → Discarded
```

**Root Cause:**
`Evaluator.validate_epoch()` returns predictions from `StreamingRegressionMetrics` which only tracks local GPU data. Unlike PYT-BUG-1 which fixed metric synchronization, the raw predictions/targets for plotting are never gathered.

**Solution:**
Gather predictions and targets from all ranks before creating plots:

```python
# In Evaluator.validate_epoch(), when return_predictions=True:
if dist.is_initialized() and dist.get_world_size() > 1:
    # Gather plot data from all ranks
    for key in ["target", "unifrac", "count"]:
        if key in all_preds:
            gathered_preds = gather_predictions_for_plot(all_preds[key])
            gathered_targs = gather_predictions_for_plot(all_targs[key])
            if is_main_process():
                all_preds[key] = gathered_preds
                all_targs[key] = gathered_targs

def gather_predictions_for_plot(local_tensor: torch.Tensor) -> torch.Tensor:
    """Gather predictions from all ranks for plotting."""
    world_size = dist.get_world_size()
    # Handle variable sizes across ranks
    local_size = torch.tensor([local_tensor.shape[0]], device=local_tensor.device)
    all_sizes = [torch.zeros(1, device=local_tensor.device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)

    max_size = max(s.item() for s in all_sizes)
    # Pad to max size for gathering
    padded = torch.zeros(max_size, *local_tensor.shape[1:], device=local_tensor.device)
    padded[:local_tensor.shape[0]] = local_tensor

    gathered = [torch.zeros_like(padded) for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    # Concatenate only valid (non-padded) portions
    result = []
    for i, g in enumerate(gathered):
        valid_size = int(all_sizes[i].item())
        result.append(g[:valid_size])
    return torch.cat(result, dim=0)
```

**Best Practices:**
1. Only gather when `return_predictions=True` (avoid unnecessary communication)
2. Handle variable batch sizes across ranks (last batch may differ)
3. Only rank 0 needs the gathered data (others can discard after gather)
4. Consider memory: limit `max_plot_samples` to prevent OOM with large validation sets
5. Use `all_gather_object` for non-tensor metadata if needed

**Acceptance Criteria:**
- [x] TensorBoard prediction plot shows ALL validation samples
- [x] Plot point count matches total validation set size
- [x] Works with both DDP and FSDP
- [x] Handles variable batch sizes across ranks
- [x] No memory explosion (respects `max_plot_samples` limit)
- [x] Only rank 0 creates/logs plots (already the case)

**Completed:**
- Added `gather_predictions_for_plot()` in distributed.py with variable-size handling
- Handles CPU tensors with NCCL backend (moves to CUDA for all_gather, back to CPU after)
- Integrated gathering into `Evaluator.validate_epoch()` when `return_predictions=True`
- Only rank 0 receives gathered data; other ranks get empty tensors
- Optional `max_samples` parameter to prevent OOM with large validation sets
- Added 10 tests for prediction gathering functionality

**Files:**
- `aam/training/evaluation.py` - Add gathering in `validate_epoch()` when `return_predictions=True`
- `aam/training/distributed.py` - Add `gather_predictions_for_plot()` utility
- `tests/test_distributed.py` - Add test for prediction gathering

**Dependencies:** None (independent of PYT-BUG-1 metric sync)

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

## Phase 12: Distributed Training (1 remaining)

### PYT-12.1: FSDP (Fully Sharded Data Parallel)
**Priority:** MEDIUM | **Effort:** 11-16 hours | **Status:** COMPLETE
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
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** COMPLETE

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
- [x] `wrap_model_fsdp()` function added to `distributed.py`
- [x] `--fsdp` flag added to `train.py` (mutually exclusive with `--distributed`)
- [ ] Training loop completes without error on multi-GPU - *requires hardware validation*
- [ ] Memory usage per GPU lower than DDP for same model - *requires hardware validation*
- [x] Unit tests for FSDP wrapping

**Completed:**
- Added `get_fsdp_wrap_policy()` with default TransformerEncoderLayer wrapping
- Added `wrap_model_fsdp()` with configurable sharding strategy, mixed precision, CPU offload
- Added `is_fsdp_model()`, `is_ddp_model()`, `unwrap_model()` helper functions
- Added `--fsdp` flag to `train.py` with distributed setup and dataloader support
- Added 15 tests for FSDP infrastructure
- Added 2 CLI tests for --fsdp flag

**Files:**
- `aam/training/distributed.py` - Add `wrap_model_fsdp()`, FSDP utilities
- `aam/cli/train.py` - Add `--fsdp` flag
- `tests/test_distributed.py` - Add FSDP wrapping tests

**Dependencies:** None

---

#### PYT-12.1b: FSDP Checkpoint Support
**Priority:** MEDIUM | **Effort:** 3-4 hours | **Status:** COMPLETE

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
- [x] FSDP model checkpoints save correctly
- [x] FSDP model can load its own checkpoints
- [x] Non-FSDP model can load FSDP checkpoint (for inference)
- [x] FSDP model can load pre-trained non-FSDP checkpoint
- [x] `--fsdp-sharded-checkpoint` option for large models
- [x] Tests for all checkpoint roundtrip scenarios

**Completed:**
- Added FSDP checkpoint utility functions: `get_fsdp_state_dict()`, `set_fsdp_state_dict()`, `get_fsdp_optimizer_state_dict()`, `set_fsdp_optimizer_state_dict()`
- Updated Trainer to handle FSDP models in `save_checkpoint()` and `load_checkpoint()`
- Added `--fsdp-sharded-checkpoint` flag for large model optimization
- Supports cross-compatibility: non-FSDP checkpoints into FSDP models and vice versa
- Added 17 tests for FSDP checkpoint functions, 2 CLI tests

**Files:**
- `aam/training/trainer.py` - FSDP checkpoint handling
- `aam/training/distributed.py` - Checkpoint utility functions
- `tests/test_distributed.py` - FSDP checkpoint tests

**Dependencies:** PYT-12.1a

---

#### PYT-12.1c: FSDP for Pretraining + ROCm Validation
**Priority:** MEDIUM | **Effort:** 4-6 hours | **Status:** COMPLETE

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
- [x] `--fsdp` flag added to `pretrain.py`
- [x] UniFrac predictions show full variance (not clustered at 0.5)
- [x] Pretraining metrics match DataParallel behavior
- [x] Works on CUDA (if available for testing)
- [ ] Works on ROCm MI300A (pending hardware validation)
- [x] Memory usage documented (vs DataParallel)
- [x] README updated with FSDP usage guidance

**Completed:**
- Added `gather_embeddings_for_unifrac()` for cross-GPU embedding collection
- Added `_gather_target_matrices()` for UniFrac target gathering with block-diagonal mask
- Integrated gathering into `MultiTaskLoss.compute_base_loss()` with `gather_for_distributed` flag
- Added `--fsdp` and `--fsdp-sharded-checkpoint` flags to pretrain.py
- Updated README with comprehensive FSDP documentation
- Added 8 tests for embedding gathering, 5 CLI tests for FSDP pretrain flags
- Added error handling for `dist.all_gather` with context-rich error messages
- Added warnings for misconfigured distributed gathering (not initialized or world_size=1)
- Fixed loss normalization to compute MSE only on valid pairs (prevents dilution from masked pairs)
- Added 4 tests for `_gather_target_matrices()`, 3 tests for masked loss computation

**Files:**
- `aam/training/distributed.py` - `gather_embeddings_for_unifrac()` with error handling
- `aam/training/losses.py` - `_gather_target_matrices()`, gathering integration, loss normalization fix
- `aam/cli/pretrain.py` - `--fsdp` flag with RuntimeError handling
- `README.md` - FSDP documentation
- `tests/test_distributed.py` - Gathering tests
- `tests/test_losses.py` - Target matrix and masked loss tests

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

## Maintenance

### PYT-MAINT-1: CLI Flag Cleanup and Default Optimization
**Priority:** LOW | **Effort:** 2-4 hours | **Status:** Not Started

Review and optimize CLI flags across train.py and pretrain.py for consistency, usability, and sensible defaults.

**Scope:**

1. **Flag Consistency Audit:**
   - Verify all model config parameters saved in checkpoint are loaded in predict.py
   - Ensure flag names are consistent between train.py and pretrain.py
   - Check for any orphaned or deprecated flags

2. **Default Value Optimization:**
   - Review default values based on empirical training results
   - Consider dataset-size-aware defaults (e.g., smaller embedding_dim for small datasets)
   - Document rationale for default choices in README

3. **Flag Cleanup:**
   - Remove any unused or redundant flags
   - Consolidate related flags where appropriate
   - Ensure mutual exclusivity constraints are properly enforced

4. **Documentation:**
   - Update README with complete flag reference table
   - Add examples for common use cases
   - Document flag interactions and dependencies

**Files:**
- `aam/cli/train.py` - Training CLI flags
- `aam/cli/pretrain.py` - Pretraining CLI flags
- `aam/cli/predict.py` - Inference CLI (model config loading)
- `README.md` - Flag documentation

**Acceptance Criteria:**
- [ ] All model config params saved in train.py are loaded in predict.py
- [ ] Flag names consistent between train.py and pretrain.py
- [ ] Default values documented with rationale
- [ ] README has complete flag reference table
- [ ] No orphaned or deprecated flags remain

---

## Summary

| Phase | Remaining | Est. Hours |
|-------|-----------|------------|
| 10 (Performance) | 0 | 0 |
| 12 (Distributed) | 1 | 4-6 |
| 18 (Memory) | 2 | 8-12 |
| 19 (Regression/Categorical) | 3 | 13-18 |
| 13-17 (Future) | ~13 | 50+ |
| **Total** | **6 + backlog** | **25-36 + 50+** |

### PYT-12.1 Sub-ticket Breakdown

| Sub-ticket | Description | Effort | Status |
|------------|-------------|--------|--------|
| PYT-12.1a | FSDP Infrastructure | 4-6h | **COMPLETE** |
| PYT-12.1b | FSDP Checkpoint Support | 3-4h | **COMPLETE** |
| PYT-12.1c | FSDP Pretraining + ROCm | 4-6h | **COMPLETE** |
