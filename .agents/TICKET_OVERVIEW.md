# Ticket Overview

**Last Updated:** 2026-01-08
**Status:** 14 outstanding tickets (~48-68 hours)

## Quick Links
- **ROCm optimization:** `COSMOS_ONBOARDING_TICKETS.md`
- **Categorical features:** `CATEGORICAL_FEATURE_TICKETS.md`
- **PyTorch work:** `PYTORCH_PORTING_TICKETS.md`
- **Completed work:** `ARCHIVED_TICKETS.md`
- **Workflow:** `WORKFLOW.md`

---

## Outstanding Tickets by Priority

### HIGH - ROCm Performance (3 tickets, ~12-20 hours)

| Ticket | Description | Effort | Domain |
|--------|-------------|--------|--------|
| **COS-9.1** | ROCm-optimized attention implementation | 6-10h | Cosmos |
| **COS-9.2** | Fix torch.compile() on ROCm/Triton | 2-4h | Cosmos |
| **COS-9.3** | Memory profiling and optimization | 4-6h | Cosmos |

**COS-9.1:** The `math` attention works correctly but is ~30% slower with higher memory usage than `mem_efficient`. Need to fix `mem_efficient`/`flash` on ROCm or optimize `math` path.

**COS-9.2:** `--compile-model` fails with Triton type mismatch. **Workaround:** omit `--compile-model` on ROCm.

**COS-9.3:** Profile memory hotspots, establish baseline for optimization work.

### MEDIUM (6 tickets, ~21-30 hours)

| Ticket | Description | Effort | Domain |
|--------|-------------|--------|--------|
| **COS-9.4** | MI300A unified memory optimization | 4-6h | Cosmos |
| **COS-9.5** | Kernel profiling with rocprof | 4-6h | Cosmos |
| **CAT-6** | Checkpoint compatibility & transfer learning | 3-4h | Categorical |
| **CAT-7** | Documentation and testing | 3-4h | Categorical |
| **PYT-12.1** | FSDP (if needed for large models) | 12-16h | PyTorch |
| **PYT-12.2** | Batch size optimization | 4-6h | PyTorch |

### LOW (5 tickets, ~17-25 hours)

| Ticket | Description | Effort | Domain |
|--------|-------------|--------|--------|
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

### 1. ROCm Performance Optimization (HIGH)
- **COS-9.3** → **COS-9.1** → **COS-9.2** (profile first, then optimize attention, then compile)

### 2. Complete Categorical Features
- **CAT-6** → **CAT-7** (checkpoint compatibility, then docs/testing)

### 3. Infrastructure (As Needed)
- **COS-9.6** - SLURM templates for Cosmos
- **COS-9.8** - Documentation

---

## Current ROCm Configuration

Working configuration for Cosmos MI300A:
```bash
aam pretrain \
  --attn-implementation math \
  --no-gradient-checkpointing \
  --data-parallel \
  # ... other flags
```

**Known limitations:**
- `--compile-model` not supported (Triton bug)
- `mem_efficient` attention produces incorrect results
- Higher memory usage than optimized CUDA path
