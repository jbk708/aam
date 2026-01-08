# Ticket Overview

**Last Updated:** 2026-01-08
**Status:** 15 outstanding tickets (~64-94 hours)

## Quick Links
- **Categorical features:** `CATEGORICAL_FEATURE_TICKETS.md`
- **PyTorch work:** `PYTORCH_PORTING_TICKETS.md`
- **Cosmos onboarding:** `COSMOS_ONBOARDING_TICKETS.md`
- **Completed work:** `ARCHIVED_TICKETS.md`
- **Workflow:** `WORKFLOW.md`

---

## Outstanding Tickets by Priority

### HIGH (1 ticket, ~2-4 hours)

| Ticket | Description | Effort | Domain |
|--------|-------------|--------|--------|
| **COS-8.1** | Fix torch.compile() on ROCm/Triton | 2-4h | Cosmos |

`--compile-model` fails with Triton type mismatch error. **Workaround:** omit `--compile-model` on ROCm.

### MEDIUM (8 tickets, ~25-38 hours)

| Ticket | Description | Effort | Domain |
|--------|-------------|--------|--------|
| **CAT-6** | Checkpoint compatibility & transfer learning | 3-4h | Categorical |
| **CAT-7** | Documentation and testing | 3-4h | Categorical |
| **COS-2.2** | Unified memory optimization for MI300A | 4-6h | Cosmos |
| **COS-3.2** | Data management scripts | 2-3h | Cosmos |
| **COS-5.1** | ROCm CI/CD pipeline | 4-6h | Cosmos |
| **COS-6.1** | MI300A performance profiling | 4-6h | Cosmos |
| **COS-6.2** | Flash Attention for ROCm | 4-6h | Cosmos |
| **COS-7.2** | Cosmos best practices guide | 2-3h | Cosmos |

### LOW (6 tickets, ~36-52 hours)

| Ticket | Description | Effort | Domain |
|--------|-------------|--------|--------|
| **COS-1.1** | ROCm Singularity container | 4-6h | Cosmos |
| **COS-3.1** | SLURM job scripts | 3-4h | Cosmos |
| **PYT-12.1** | FSDP (consolidated with COS-4.2) | 12-16h | PyTorch |
| **PYT-12.2** | Batch size optimization | 4-6h | PyTorch |
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

**PYT-10.7: DataParallel for Pretraining** (2026-01-08)
- Added `--data-parallel` flag for single-node multi-GPU pretraining
- DataParallel preserves full pairwise UniFrac comparisons (unlike DDP)
- Documented DP vs DDP trade-offs in README

**CAT-1 through CAT-5: Categorical Features** (2026-01-05)
- Schema definition, dataset encoding, embedder module
- SequencePredictor integration with concat/add fusion
- CLI flags: `--categorical-columns`, `--categorical-embed-dim`, `--categorical-fusion`

**PYT-21: Transfer Learning** (2025-12-19)
- Regressor head optimization (unbounded regression, LayerNorm)
- Pretrained encoder loading fixes
- Skip nucleotide predictions during fine-tuning

**COS-5.2: Numerical Validation** (2025-12-19)
- Golden file validation infrastructure for CUDA vs ROCm

**CLN-1 through CLN-6: Code Cleanup** (2025-12-19)
- Removed deprecated UniFrac modules (~4165 lines)
- Extracted CLI package and trainer validation logic
- Fixed all type errors

---

## Recommended Next Steps

### 1. Fix ROCm Blocker (HIGH)
- **COS-8.1** - Fix torch.compile() on ROCm (or document workaround)

### 2. Complete Categorical Features
- **CAT-6** → **CAT-7** (checkpoint compatibility, then docs/testing)

### 3. Cosmos Onboarding
- **COS-5.1** - ROCm CI/CD pipeline
- **COS-3.2** - Data management scripts

### 4. Performance (As Needed)
- **PYT-12.1** - FSDP if needed for large models

---

## Consolidation Notes

**FSDP tickets merged:** COS-4.2 consolidated into PYT-12.1. Implement once, validate on both CUDA and ROCm. May be unnecessary given MI300A's 128GB unified memory.
