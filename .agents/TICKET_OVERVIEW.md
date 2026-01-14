# Ticket Overview

**Last Updated:** 2026-01-14
**Status:** 14 outstanding tickets (~47-70 hours)

## Quick Links
- **Regressor optimization:** `REGRESSOR_OPTIMIZATION_TICKETS.md` ← NEW
- **ROCm optimization:** `COSMOS_ONBOARDING_TICKETS.md`
- **Categorical features:** `CATEGORICAL_FEATURE_TICKETS.md`
- **PyTorch work:** `PYTORCH_PORTING_TICKETS.md`
- **Documentation:** `DOCUMENTATION_TICKETS.md`
- **Completed work:** `ARCHIVED_TICKETS.md`
- **Workflow:** `WORKFLOW.md`

---

## Outstanding Tickets by Priority

### URGENT (0 tickets)

No urgent tickets remaining.

### HIGH (2 tickets, ~7-9 hours)

| Ticket | Description | Effort | Domain |
|--------|-------------|--------|--------|
| **REG-3** | Conditional output scaling | 3-4h | Regressor |
| **REG-4** | FiLM layers (categorical modulation) | 4-5h | Regressor |

### MEDIUM (3 tickets, ~10-15 hours)

| Ticket | Description | Effort | Domain |
|--------|-------------|--------|--------|
| **REG-5** | Quantile regression | 4-6h | Regressor |
| **REG-6** | Asymmetric loss | 2-3h | Regressor |
| **PYT-12.2** | Batch size optimization | 4-6h | PyTorch |

### LOW (9 tickets, ~32-46 hours)

| Ticket | Description | Effort | Domain |
|--------|-------------|--------|--------|
| **REG-7** | Residual regression head | 2-3h | Regressor |
| **REG-8** | Per-output loss config | 3-4h | Regressor |
| **REG-9** | Mixture of Experts | 6-8h | Regressor |
| **COS-9.5** | Kernel profiling with rocprof | 4-6h | Cosmos |
| **COS-9.6** | SLURM job templates | 3-4h | Cosmos |
| **COS-9.7** | ROCm Singularity container | 4-6h | Cosmos |
| **COS-9.8** | ROCm documentation & best practices | 2-3h | Cosmos |
| **PYT-18.5** | Lazy sample embedding computation | 4-6h | PyTorch |
| **PYT-18.6** | Memory-aware dynamic batching | 4-6h | PyTorch |

### BACKLOG (Phases 13-17)

Future enhancement phases (~50+ hours):
- Phase 13: Attention Visualization, Feature Importance, Encoder Types
- Phase 14: Streaming Data, Augmentation
- Phase 15: Experiment Tracking, Hyperparameter Optimization
- Phase 16: Benchmarking, Error Analysis
- Phase 17: Docs, Tutorials, ONNX, Docker

---

## Recently Completed

**PYT-BUG-4: Distributed Validation Plots Show Only Local GPU Data** (2026-01-14) - COMPLETE
- Added `gather_predictions_for_plot()` for cross-GPU prediction gathering
- Handles CPU tensors with NCCL backend (automatic CUDA transfer)
- Integrated into `Evaluator.validate_epoch()` when `return_predictions=True`
- TensorBoard scatter plots now show ALL validation samples across GPUs
- Added 10 tests for prediction gathering functionality

**PYT-BUG-3: Count Loss Has No Configurable Weight** (2026-01-14) - COMPLETE
- Added `--count-penalty` flag to train.py and pretrain.py (default 1.0)
- Allows disabling or downweighting count loss
- Documented in README Loss Weights table

**PYT-BUG-2: Best Model Selection Uses Loss Instead of Primary Metric** (2026-01-14) - COMPLETE
- Added `--best-metric` flag with choices: val_loss, r2, mae, accuracy, f1
- Added `is_metric_better()` helper for min/max mode comparison
- Checkpoint now stores `best_metric` and `best_metric_value`
- Default behavior unchanged (val_loss) for backwards compatibility
- Added 14 trainer tests and 3 CLI tests

**PYT-BUG-1: Distributed Validation Metrics Not Synchronized** (2026-01-14) - COMPLETE
- Added `all_reduce()` for distributed metric synchronization
- Validation metrics now aggregated across all GPUs in DDP/FSDP
- TensorBoard and console logs show true global performance

**REG-2: Per-Category Target Normalization** (2026-01-13) - COMPLETE
- Added `--normalize-targets-by` flag for per-category z-score normalization
- CategoryNormalizer computes per-category mean/std from training data
- Unseen categories fall back to global statistics with warning
- Mutually exclusive with `--normalize-targets` (global normalization)
- Statistics saved in checkpoint for inference
- Added 26 unit tests for normalization roundtrip

**REG-1: MLP Regression Head** (2026-01-13) - COMPLETE
- Added `--regressor-hidden-dims` flag for configurable MLP (e.g., `64,32`)
- Added `--regressor-dropout` flag for dropout between MLP layers
- Default behavior unchanged (single linear layer)
- Works with all existing output transforms (sigmoid, softplus, etc.)
- Added 25 unit tests for MLP configurations

**DOC-1: README & Installation Modernization** (2026-01-13) - COMPLETE
- Replaced conda/mamba installation with pip-only workflow
- Added Python version requirements (3.9-3.12) and PyTorch installation instructions
- Added Quick Start section with included test data (781 samples)
- Updated test count from 679 to 919
- Updated CLAUDE.md with complete test data file list

**PYT-12.1c: FSDP Pretraining + ROCm Validation** (2026-01-13) - COMPLETE
- Added `gather_embeddings_for_unifrac()` for cross-GPU embedding collection
- Added `_gather_target_matrices()` for UniFrac target gathering
- Integrated gathering into `MultiTaskLoss.compute_base_loss()` with `gather_for_distributed` flag
- Added `--fsdp` and `--fsdp-sharded-checkpoint` flags to pretrain.py
- Updated README with comprehensive FSDP documentation
- Added 8 tests for embedding gathering, 5 CLI tests for FSDP pretrain flags
- PYT-12.1 FSDP Implementation now complete (all 3 sub-tickets done)

**PYT-12.1b: FSDP Checkpoint Support** (2026-01-13) - COMPLETE
- Added FSDP checkpoint utility functions: `get_fsdp_state_dict()`, `set_fsdp_state_dict()`, `get_fsdp_optimizer_state_dict()`, `set_fsdp_optimizer_state_dict()`
- Updated Trainer to handle FSDP models in `save_checkpoint()` and `load_checkpoint()`
- Added `--fsdp-sharded-checkpoint` flag for large model optimization
- Supports cross-compatibility: non-FSDP checkpoints into FSDP models and vice versa
- Added 17 tests for FSDP checkpoint functions, 2 CLI tests

**PYT-12.1a: FSDP Infrastructure** (2026-01-13) - COMPLETE
- Added `wrap_model_fsdp()` with configurable sharding strategy, mixed precision, CPU offload
- Added `get_fsdp_wrap_policy()` for transformer layer auto-wrapping
- Added `--fsdp` flag to train.py (mutually exclusive with `--distributed`)
- Added helper functions: `is_fsdp_model()`, `is_ddp_model()`, `unwrap_model()`
- Added 17 tests for FSDP infrastructure and CLI

**CAT-7: Documentation and Testing** (2026-01-13) - COMPLETE
- Added best practices to README for embedding dim selection and rare categories
- Added TestCategoricalIntegration with 3 integration tests
- All categorical feature work now complete

**CAT-6: Checkpoint Compatibility and Transfer Learning** (2026-01-12) - COMPLETE
- Verified pretrained encoder loads into model with categorical features
- Categorical weights preserved (random init) when loading encoder
- `--freeze-base` correctly includes categorical embedder in optimization
- Added comprehensive test suite (6 tests) verifying staged training workflow

**COS-9.3: Memory Profiling and Optimization** (2026-01-12) - COMPLETE
- Added `--memory-profile` flag to pretrain command
- Created `MemoryProfiler` class for memory logging at key points
- Hardware testing completed on MI300A

**COS-9.9: PyTorch 2.7 SDPA Fix Verified** (2026-01-12) - COMPLETE
- Confirmed ROCm 6.3 + PyTorch 2.7.1 fixes `mem_efficient` SDPA
- Masked attention max_diff: 1.10e-06 (was ~1.73 on ROCm 6.2)
- Performance: 3.76x faster, 4.4x less memory than `math` backend
- `--attn-implementation math` no longer required on ROCm 6.3

**COS-9.2: Fix torch.compile() on ROCm** (2026-01-12) - COMPLETE
- Added ROCm detection via `torch.version.hip`
- `--compile-model` now skips gracefully on ROCm with warning
- Updated README with ROCm limitations table

**COS-9.1: ROCm Attention Investigation** (2026-01-12) - COMPLETE
- Root cause: `mem_efficient` SDPA produces wrong results WITH attention masks on ROCm
- Without masks: max_diff=7e-7 (fine). With masks: max_diff=1.73 (broken)
- Flash Attention for ROCm incompatible with ROCm 6.2+ (build fails)
- Created diagnostic tool: `python -m aam.tools.rocm_attention_diagnostic`
- **Resolution:** Use `--attn-implementation math` (required for correct results)

**COS-8.2: ROCm Numerical Divergence** (2026-01-08) - RESOLVED
- `mem_efficient` SDPA produced incorrect results (42% vs 70% nuc accuracy)
- **Resolution:** Use `--attn-implementation math --no-gradient-checkpointing`
- Trade-off: Higher memory, slower iteration rate
- Renumbered remaining optimization work to COS-9.x series

**PYT-10.7: DataParallel for Pretraining** (2026-01-08)
- Added `--data-parallel` flag for single-node multi-GPU pretraining
- DataParallel preserves full pairwise UniFrac comparisons (unlike DDP)

**CAT-1 through CAT-5: Categorical Features** (2026-01-05)
- Schema definition, dataset encoding, embedder module
- SequencePredictor integration with concat/add fusion
- CLI flags: `--categorical-columns`, `--categorical-embed-dim`, `--categorical-fusion`

---

## Recommended Next Steps

### 1. Regressor Optimization (HIGH Priority)
Focus on categorical compensation for multi-environment/season data:
- **REG-3** - Conditional output scaling (complements REG-2 ✓)
- **REG-4** - FiLM layers (most expressive modulation, requires REG-1 ✓)

### 2. Loss Function Improvements (MEDIUM)
- **REG-5** - Quantile regression for uncertainty estimation
- **REG-6** - Asymmetric loss (if directional errors matter)

### 3. Batch Size Optimization (MEDIUM)
- **PYT-12.2** - Batch size optimization for efficient training

### 4. Infrastructure (As Needed)
- **COS-9.6** - SLURM templates for Cosmos
- **COS-9.8** - ROCm documentation

### 5. Memory Optimization (LOW)
- **PYT-18.5** - Lazy sample embedding computation
- **PYT-18.6** - Memory-aware dynamic batching

### 6. Advanced Regressor (LOW)
- **REG-7, 8, 9** - Residual head, per-output loss, Mixture of Experts

---

## Current ROCm Configuration

**ROCm 6.3 + PyTorch 2.7.1 (Recommended):**
```bash
# No special flags needed - defaults work correctly
aam pretrain --data-parallel --batch-size 32 \
  --table data.biom --unifrac-matrix unifrac.npy --output-dir output/
```

**ROCm 6.2 + PyTorch 2.5.1 (Legacy):**
```bash
aam pretrain \
  --attn-implementation math \
  --no-gradient-checkpointing \
  --data-parallel \
  # ... other flags
```

**Known limitations:**
- `--compile-model` not supported on any ROCm version (Triton bug)
- ROCm 6.2: `mem_efficient` attention broken, requires `--attn-implementation math`
- ROCm 6.3: `mem_efficient` attention works correctly (fixed in aotriton 0.8.2)
