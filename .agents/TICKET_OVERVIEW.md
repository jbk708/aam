# Ticket Overview

**Last Updated:** 2026-01-13
**Status:** 10 outstanding tickets (~32-47 hours)

## Quick Links
- **ROCm optimization:** `COSMOS_ONBOARDING_TICKETS.md`
- **Categorical features:** `CATEGORICAL_FEATURE_TICKETS.md`
- **PyTorch work:** `PYTORCH_PORTING_TICKETS.md`
- **Completed work:** `ARCHIVED_TICKETS.md`
- **Workflow:** `WORKFLOW.md`

---

## Outstanding Tickets by Priority

### HIGH (0 tickets)

All high priority tickets complete.

### MEDIUM (2 tickets, ~16-22 hours)

| Ticket | Description | Effort | Domain |
|--------|-------------|--------|--------|
| **PYT-12.1** | FSDP (if needed for large models) | 12-16h | PyTorch |
| **PYT-12.2** | Batch size optimization | 4-6h | PyTorch |

### LOW (6 tickets, ~21-31 hours)

| Ticket | Description | Effort | Domain |
|--------|-------------|--------|--------|
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

### 1. Infrastructure (As Needed)
- **COS-9.6** - SLURM templates for Cosmos
- **COS-9.8** - ROCm documentation

### 2. Distributed Training (MEDIUM)
- **PYT-12.1** - FSDP for large models (if needed)
- **PYT-12.2** - Batch size optimization

### 3. Memory Optimization (LOW)
- **PYT-18.5** - Lazy sample embedding computation
- **PYT-18.6** - Memory-aware dynamic batching

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
